"""
PLY sequence to GIF renderer compatible with headless HPC cluster environments.

Uses pyrender with EGL (PYOPENGL_PLATFORM=egl), which works without a hardware
GPU or display server. Interactive camera setup uses a matplotlib 3D window and
requires a working DISPLAY (X11 forwarding). Headless GIF generation requires
no display at all.

Usage (headless, auto-camera):
    python ply_to_gif.py --input /path/to/ply_folder
Usage (with interactive camera, needs DISPLAY):
    DISPLAY=:0 python ply_to_gif.py  # prompts for folder, opens matplotlib window
Usage (reuse a saved camera):
    python ply_to_gif.py --input /path/to/ply_folder --cam_json /path/to/camera_config.json
"""

import os

# Must be set before any OpenGL / pyrender imports.
# Forces pyrender to use the EGL software backend, which works without a GPU.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import json
import re
from pathlib import Path

import imageio
import numpy as np
import pyrender
import trimesh

# ==========================================
# SETTINGS
# ==========================================
FPS = 10
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
_SLOT_PATTERN = re.compile(r"slot(\d{2})_.*?_epoch(\d{5})\.ply")


def _extract_vertex_colors(geo: object) -> np.ndarray | None:
    """Return per-vertex RGBA colors as float32 (N, 4) in [0, 1], or None.

    Args:
        geo: A trimesh.PointCloud or trimesh.Trimesh.

    Returns:
        (N, 4) float32 RGBA array, or None if no color data is present.
    """
    colors: np.ndarray | None = None
    if isinstance(geo, trimesh.PointCloud):
        if geo.colors is not None and len(geo.colors) > 0:
            colors = np.asarray(geo.colors, dtype=np.float32)
    elif isinstance(geo, trimesh.Trimesh):
        visual = getattr(geo, "visual", None)
        if visual is not None:
            vc = getattr(visual, "vertex_colors", None)
            if vc is not None:
                vc = np.asarray(vc, dtype=np.float32)
                if vc.ndim == 2 and vc.shape[0] == len(geo.vertices):
                    colors = vc

    if colors is None:
        return None
    if colors.max() > 1.0:
        colors = colors / 255.0
    if colors.shape[1] == 3:
        colors = np.hstack([colors, np.ones((len(colors), 1), dtype=np.float32)])
    return colors.clip(0.0, 1.0).astype(np.float32)


def _trimesh_to_pyrender(geo: object) -> pyrender.Mesh | None:
    """Convert one trimesh geometry to a pyrender.Mesh, preserving vertex colors.

    For solid meshes (has faces), delegates to pyrender.Mesh.from_trimesh which
    carries ColorVisuals through automatically. For point clouds (no faces),
    builds a POINTS primitive with per-vertex colors extracted from the geometry.

    Args:
        geo: A trimesh.Trimesh or trimesh.PointCloud.

    Returns:
        pyrender.Mesh ready to add to a scene, or None if geometry is empty.
    """
    if isinstance(geo, trimesh.Trimesh) and len(geo.faces) > 0:
        return pyrender.Mesh.from_trimesh(geo, smooth=False)

    # Point-cloud path: Trimesh with zero faces, or PointCloud
    if isinstance(geo, (trimesh.Trimesh, trimesh.PointCloud)):
        points = np.asarray(geo.vertices, dtype=np.float32)
    else:
        return None

    if len(points) == 0:
        return None

    colors = _extract_vertex_colors(geo)
    if colors is None or len(colors) != len(points):
        # No stored colors: neutral white, visible on black background
        colors = np.ones((len(points), 4), dtype=np.float32)

    return pyrender.Mesh(primitives=[pyrender.Primitive(positions=points, color_0=colors, mode=pyrender.GLTF.POINTS)])


def _load_pyrender_geometries(fpath: Path) -> list[pyrender.Mesh]:
    """Load all geometries from a PLY as separate pyrender.Mesh objects.

    Returns one Mesh per trimesh sub-geometry so that each retains its own
    vertex colors. Handles solid mesh, point cloud, and Scene files without
    triggering Open3D geometry-type warnings.

    Args:
        fpath: Path to the PLY file.

    Returns:
        List of pyrender.Mesh objects (may be empty if file has no geometry).
    """
    loaded = trimesh.load(str(fpath), process=False)

    if isinstance(loaded, trimesh.Scene):
        geos: list[object] = list(loaded.geometry.values())
    elif isinstance(loaded, (trimesh.Trimesh, trimesh.PointCloud)):
        geos = [loaded]
    else:
        return []

    result = []
    for geo in geos:
        mesh = _trimesh_to_pyrender(geo)
        if mesh is not None:
            result.append(mesh)
    return result


def _get_vertices(fpath: Path) -> np.ndarray | None:
    """Return all vertex positions from a PLY file as an (N, 3) array."""
    loaded = trimesh.load(str(fpath), process=False)

    if isinstance(loaded, trimesh.Scene):
        parts = [np.asarray(g.vertices) for g in loaded.geometry.values() if hasattr(g, "vertices") and len(g.vertices) > 0]
        return np.vstack(parts) if parts else None

    if hasattr(loaded, "vertices") and len(loaded.vertices) > 0:
        return np.asarray(loaded.vertices)

    return None


def _compute_camera_pose(vertices: np.ndarray) -> np.ndarray:
    """Compute a stable camera pose that frames all vertices.

    The camera is placed along +Z from the geometry centroid at a
    distance proportional to the bounding-box diagonal, looking back
    toward the centroid.

    Args:
        vertices: (N, 3) array of vertex positions.

    Returns:
        4x4 camera-to-world pose matrix (OpenGL convention: camera
        looks along its local -Z axis).
    """
    centroid = vertices.mean(axis=0)
    diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if diag < 1e-6:
        diag = 1.0

    cam_pos = centroid + np.array([0.0, 0.0, diag * 2.0])
    forward = centroid - cam_pos  # direction camera looks toward
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # OpenGL: camera looks along -Z, so camera_Z_world = -forward
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = cam_pos
    return pose


def _load_camera_pose(cam_json: Path) -> np.ndarray | None:
    """Load a 4x4 camera-to-world pose matrix from a JSON file.

    The JSON must contain a key "pose" with a 4x4 nested list. This
    format is written by _try_interactive_camera().

    Args:
        cam_json: Path to the JSON file.

    Returns:
        4x4 pose matrix, or None if loading fails.
    """
    try:
        with open(cam_json) as f:
            data = json.load(f)
        return np.array(data["pose"], dtype=np.float64)
    except Exception as exc:
        print(f"Warning: Could not load camera from {cam_json}: {exc}. Auto-computing.")
        return None


def _render_frame(
    renderer: pyrender.OffscreenRenderer,
    fpath: Path,
    cam_pose: np.ndarray,
) -> np.ndarray:
    """Render one PLY file to an (H, W, 3) uint8 RGB image.

    A fresh scene is created per frame so camera and geometry do not
    accumulate across frames.

    Args:
        renderer: Shared OffscreenRenderer (reused across all frames for speed).
        fpath: PLY file to render.
        cam_pose: 4x4 camera-to-world pose matrix.

    Returns:
        (WINDOW_HEIGHT, WINDOW_WIDTH, 3) uint8 image array.
    """
    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 1.0],
        ambient_light=[0.4, 0.4, 0.4, 1.0],
    )
    camera = pyrender.PerspectiveCamera(
        yfov=np.pi / 4.0,
        aspectRatio=WINDOW_WIDTH / WINDOW_HEIGHT,
    )
    scene.add(camera, pose=cam_pose)
    # Directional light co-located with camera so the object is always lit
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)

    for geo in _load_pyrender_geometries(fpath):
        scene.add(geo)

    color, _ = renderer.render(scene)
    return color


def _build_camera_pose_from_angles(
    centroid: np.ndarray,
    dist: float,
    azim_deg: float,
    elev_deg: float,
) -> np.ndarray:
    """Build a 4x4 camera-to-world pose from azimuth and elevation angles (matplotlib convention).

    Args:
        centroid: World position that the camera looks toward.
        dist: Distance from centroid to camera.
        azim_deg: Azimuth angle in degrees.
        elev_deg: Elevation angle in degrees.

    Returns:
        4x4 camera-to-world pose matrix (OpenGL convention).
    """
    azim_rad = np.radians(azim_deg)
    elev_rad = np.radians(elev_deg)
    cx = dist * np.cos(elev_rad) * np.cos(azim_rad)
    cy = dist * np.cos(elev_rad) * np.sin(azim_rad)
    cz = dist * np.sin(elev_rad)
    cam_pos = centroid + np.array([cx, cy, cz])

    forward = centroid - cam_pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # OpenGL convention: camera looks along its local -Z axis
    cam_pose = np.eye(4, dtype=np.float64)
    cam_pose[:3, 0] = right
    cam_pose[:3, 1] = up
    cam_pose[:3, 2] = -forward
    cam_pose[:3, 3] = cam_pos
    return cam_pose


def _try_interactive_camera(first_ply: Path, output_dir: Path) -> Path | None:
    """Open a matplotlib 3D window for interactive camera positioning.

    Displays the first PLY geometry so the user can rotate to the desired
    viewpoint. Reads back azim/elev and converts to a pyrender camera pose
    using Z-as-up convention (matching matplotlib's default orientation).

    Falls back to None (auto-compute) if a display window cannot be opened
    or any other error occurs.

    Args:
        first_ply: Sample PLY file to display.
        output_dir: Directory in which to write camera_config.json.

    Returns:
        Path to the saved camera JSON, or None on failure.
    """
    try:
        import matplotlib  # noqa: PLC0415 — deferred: not needed on headless nodes

        # Switch to an interactive backend before pyplot is imported.
        # Try in order of preference; the first successful use() wins.
        for _backend in ("TkAgg", "Qt5Agg", "GTK3Agg", "WXAgg"):
            try:
                matplotlib.use(_backend)
                break
            except Exception:
                continue

        import matplotlib.pyplot as plt  # noqa: PLC0415
        from mpl_toolkits.mplot3d import Axes3D  # noqa: PLC0415, F401

        verts = _get_vertices(first_ply)
        if verts is None:
            print("Warning: No vertices in sample file. Auto-computing camera.")
            return None

        centroid = verts.mean(axis=0)
        diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
        if diag < 1e-6:
            diag = 1.0
        dist = diag * 2.0

        # Down-sample for interactive performance
        n_display = min(len(verts), 8000)
        if len(verts) > n_display:
            rng = np.random.default_rng()
            idx = rng.choice(len(verts), n_display, replace=False)
        else:
            idx = np.arange(len(verts))
        pts = verts[idx]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.0, c="cyan", alpha=0.7)
        ax.set_title(
            f"Rotate to the desired view angle, then CLOSE this window.\nCurrent: azim={ax.azim:.1f}°  elev={ax.elev:.1f}°"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        print("INSTRUCTIONS: Rotate the 3D view to your desired camera angle, then CLOSE the window.")
        plt.show()  # blocks until the user closes the window

        azim_deg = float(ax.azim)
        elev_deg = float(ax.elev)

        # Build camera-to-world pose using the helper function
        cam_pose = _build_camera_pose_from_angles(centroid, dist, azim_deg, elev_deg)

        cam_json_path = output_dir / "camera_config.json"
        with open(cam_json_path, "w") as f:
            json.dump(
                {
                    "pose": cam_pose.tolist(),
                    "azim_deg": azim_deg,
                    "elev_deg": elev_deg,
                },
                f,
                indent=2,
            )
        print(f"Camera saved (azim={azim_deg:.1f}°, elev={elev_deg:.1f}°): {cam_json_path}")
    except Exception as exc:
        print(f"Warning: Interactive camera setup failed ({exc}). Auto-computing camera.")
        return None
    else:
        return cam_json_path


def interactive_setup() -> tuple[Path | None, Path | None, Path | None]:
    """Prompt for an input folder and optionally run an interactive camera setup.

    If DISPLAY is set, attempts to open an Open3D window for camera positioning.
    If not (e.g., headless cluster node), skips to auto-computed camera.

    Returns:
        (input_path, output_path, cam_json_path) — cam_json_path may be None.
    """
    print("\n" + "=" * 40)
    print(" INTERACTIVE GIF SETUP ")
    print("=" * 40)

    input_str = input("Enter the path to the folder containing PLY files: ").strip()
    input_path = Path(input_str).resolve()

    if not input_path.is_dir():
        print(f"Error: Directory '{input_path}' does not exist.")
        return None, None, None

    output_path = input_path.parent / "pretraining_latent_gifs"
    first_ply = next(input_path.glob("*.ply"), None)
    if first_ply is None:
        print(f"Error: No PLY files found in {input_path}.")
        return None, None, None

    print(f"Found sample file: {first_ply.name}")

    cam_json_path: Path | None = None
    if os.environ.get("DISPLAY"):
        cam_json_path = _try_interactive_camera(first_ply, input_path)
    else:
        print("No DISPLAY detected. Camera will be auto-computed from geometry bounds.")

    return input_path, output_path, cam_json_path


def generate_gifs(
    input_path: Path,
    output_path: Path,
    cam_json: Path | None = None,
) -> None:
    """Render PLY sequences to convergence GIF files.

    Groups PLY files by slot number (slot<2d>_*_epoch<5d>.ply pattern),
    sorts each group by epoch, and renders a GIF per slot. Camera pose
    is fixed for the entire slot sequence (computed from the union of all
    frame vertices) to guarantee a smooth, jitter-free animation.

    Args:
        input_path: Directory containing PLY files.
        output_path: Directory where GIF files are written.
        cam_json: Optional path to a camera_config.json from a prior
            interactive setup. When provided and valid, it overrides
            auto-computed camera for all slots.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    slots: dict[str, list[tuple[int, Path]]] = {}
    for filepath in input_path.glob("*.ply"):
        match = _SLOT_PATTERN.search(filepath.name)
        if match:
            slot_num = match.group(1)
            epoch_num = int(match.group(2))
            slots.setdefault(slot_num, []).append((epoch_num, filepath))

    if not slots:
        raise ValueError(
            f"No PLY files matching 'slot<2d>_*_epoch<5d>.ply' found in {input_path}. "
            "Check that file names follow the expected naming convention."
        )

    cam_pose_override: np.ndarray | None = None
    if cam_json is not None and cam_json.exists():
        cam_pose_override = _load_camera_pose(cam_json)

    renderer = pyrender.OffscreenRenderer(WINDOW_WIDTH, WINDOW_HEIGHT)
    try:
        for slot, files in sorted(slots.items()):
            files.sort(key=lambda x: x[0])
            file_paths = [fp for _, fp in files]
            print(f"\nRendering Slot {slot} ({len(files)} frames)...")

            if cam_pose_override is not None:
                cam_pose = cam_pose_override
            else:
                # Collect vertices from every frame so the camera frames the full
                # trajectory, not just one snapshot.
                all_verts_list = [v for v in (_get_vertices(fp) for fp in file_paths) if v is not None]
                if not all_verts_list:
                    print(f"  Warning: No vertex data for slot {slot}. Skipping.")
                    continue
                cam_pose = _compute_camera_pose(np.vstack(all_verts_list))

            frames = [_render_frame(renderer, fp, cam_pose) for fp in file_paths]
            gif_path = output_path / f"slot{slot}_convergence.gif"
            imageio.mimsave(str(gif_path), frames, fps=FPS)  # type: ignore[arg-type]
            print(f"  Saved: {gif_path}")
    finally:
        renderer.delete()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PLY file sequences to GIFs. Cluster-safe (EGL headless).")
    parser.add_argument("--input", type=str, help="Input folder containing PLY files.")
    parser.add_argument("--output", type=str, help="Output folder for GIF files.")
    parser.add_argument(
        "--cam_json",
        type=str,
        help="Path to camera_config.json from a prior interactive setup.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    in_p: Path | None
    out_p: Path | None
    c_j: Path | None
    if args.input:
        in_p = Path(args.input).resolve()
        out_p = Path(args.output).resolve() if args.output else in_p.parent / "pretraining_latent_gifs"
        cam_p: Path | None = Path(args.cam_json).resolve() if args.cam_json else None
        generate_gifs(in_p, out_p, cam_json=cam_p)
    else:
        in_p, out_p, c_j = interactive_setup()
        if in_p is not None and out_p is not None:
            generate_gifs(in_p, out_p, cam_json=c_j)
