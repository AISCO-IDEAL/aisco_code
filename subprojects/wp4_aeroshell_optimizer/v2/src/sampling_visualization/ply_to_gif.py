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
import time
from pathlib import Path

import imageio
import numpy as np
import pyrender
import trimesh
import yaml
from PIL import Image, ImageDraw, ImageFont

_CFG = yaml.safe_load((Path(__file__).parents[2] / "configs" / "vis_config.yaml").read_text())
FPS: int = _CFG["fps"]
WINDOW_WIDTH: int = _CFG["window_width"]
WINDOW_HEIGHT: int = _CFG["window_height"]
CAMERA_DIRECTION: str = _CFG["camera_direction"]
CAMERA_DISTANCE_MULTIPLIER: float = _CFG["camera_distance_multiplier"]
CAMERA_ELEVATION_ANGLE: float = _CFG["camera_elevation_angle"]
POINT_SIZE: int = _CFG["point_size"]
BACKGROUND_COLOR: list[float] = _CFG["background_color"]

# Per-component color overrides for the combined bbox+SDF PLY format.
# Order matches the alpha-sorted split: [inner_bbox, sdf/aeroshell, outer_bbox].
# Each entry is (rgb_or_None, alpha_or_None); None means keep the baked PLY color.
_ASSEMBLY_CFG = _CFG.get("visual_types", {}).get("mesh_assembly_v4", {})
_COMPONENT_COLORS: list[tuple[list[float] | None, float | None]] = [
    (_ASSEMBLY_CFG.get("inner_color"), None),
    (_ASSEMBLY_CFG.get("sdf_color"), _ASSEMBLY_CFG.get("sdf_alpha")),
    (_ASSEMBLY_CFG.get("outer_color"), _ASSEMBLY_CFG.get("outer_alpha")),
]

_SLOT_PATTERN = re.compile(r"slot(\d{2})_.*?_epoch(\d{5})\.ply")
_FLAT_PATTERN = re.compile(r"^epoch_?(\d{5})\.ply$")


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


def _trimesh_to_pyrender(geo: object, wireframe: bool = False) -> pyrender.Mesh | None:
    """Convert one trimesh geometry to a pyrender.Mesh, preserving vertex colors.

    For solid meshes (has faces), delegates to pyrender.Mesh.from_trimesh which
    carries ColorVisuals through automatically. For point clouds (no faces),
    builds a POINTS primitive with per-vertex colors extracted from the geometry.

    Args:
        geo: A trimesh.Trimesh or trimesh.PointCloud.
        wireframe: If True, render the mesh as a wireframe (no face fill, no Z-buffer write).

    Returns:
        pyrender.Mesh ready to add to a scene, or None if geometry is empty.
    """
    if isinstance(geo, trimesh.Trimesh) and len(geo.faces) > 0:
        return pyrender.Mesh.from_trimesh(geo, smooth=False, wireframe=wireframe)

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


def _split_combined_bbox_mesh(
    geo: trimesh.Trimesh,
) -> list[tuple[trimesh.Trimesh, bool]] | None:
    """Detect and split the combined inner+outer+SDF PLY format.

    ``export_combined_bbox_mesh_ply`` merges three components into one trimesh
    with distinct per-vertex alpha values:

    * inner bbox  → alpha ≈ 255  (opaque)
    * aeroshell   → alpha ≈ 128  (50 % opacity)
    * outer bbox  → alpha ≈ 51   (20 % opacity)

    Returns ``None`` when fewer than 3 distinct alpha groups are present (i.e.
    this is a normal single-component PLY).  Otherwise returns a list of
    ``(sub_mesh, wireframe)`` tuples in the correct pyrender render order::

        [(inner, False), (aeroshell, False), (outer, True)]

    Render order matters because pyrender writes to the depth buffer even in
    BLEND mode.  Placing the outer bbox last as a wireframe prevents it from
    occluding the aeroshell via Z-buffer writes.
    """
    colors = np.asarray(geo.visual.vertex_colors, dtype=np.uint8)
    if colors.ndim != 2 or colors.shape[1] < 4:
        return None

    alpha = colors[:, 3]
    # Sort highest first: [inner≈255, aeroshell≈128, outer≈51]
    unique_alphas = sorted(set(alpha.tolist()), reverse=True)
    if len(unique_alphas) < 3:
        return None

    verts = np.asarray(geo.vertices)
    faces = np.asarray(geo.faces, dtype=np.int64)

    result: list[tuple[trimesh.Trimesh, bool]] = []
    for i, alpha_val in enumerate(unique_alphas[:3]):
        wireframe = i == 2  # only the lowest-alpha group (outer bbox) is wireframe
        v_mask = alpha == alpha_val
        v_idx = np.where(v_mask)[0]
        if len(v_idx) == 0:
            continue
        v_min, v_max = int(v_idx.min()), int(v_idx.max())
        f_mask = np.all((faces >= v_min) & (faces <= v_max), axis=1)
        if not f_mask.any():
            continue
        sub = trimesh.Trimesh(
            vertices=verts[v_min : v_max + 1],
            faces=faces[f_mask] - v_min,
            vertex_colors=colors[v_min : v_max + 1],
            process=False,
        )
        # Apply color overrides from vis_config.yaml if configured.
        if i < len(_COMPONENT_COLORS):
            cfg_rgb, cfg_alpha = _COMPONENT_COLORS[i]
            if cfg_rgb is not None:
                r, g, b = (int(c * 255) for c in cfg_rgb)
                a = int(cfg_alpha * 255) if cfg_alpha is not None else alpha_val
                sub.visual.vertex_colors = np.full((len(sub.vertices), 4), [r, g, b, a], dtype=np.uint8)
        result.append((sub, wireframe))

    return result if len(result) >= 2 else None


def _load_pyrender_geometries(fpath: Path) -> list[pyrender.Mesh]:
    """Load all geometries from a PLY as separate pyrender.Mesh objects.

    Returns one Mesh per trimesh sub-geometry so that each retains its own
    vertex colors. Handles solid mesh, point cloud, and Scene files without
    triggering Open3D geometry-type warnings.  The combined bbox+SDF format
    produced by ``export_combined_bbox_mesh_ply`` is detected automatically
    and split into three separate meshes rendered in the correct depth order
    (inner opaque → aeroshell transparent → outer bbox wireframe).

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
        # Detect the combined inner+outer+SDF format and split it so each
        # component can be rendered with the correct depth/wireframe settings.
        if isinstance(geo, trimesh.Trimesh) and len(geo.faces) > 0:
            split = _split_combined_bbox_mesh(geo)
            if split is not None:
                for sub_geo, wf in split:
                    mesh = _trimesh_to_pyrender(sub_geo, wireframe=wf)
                    if mesh is not None:
                        result.append(mesh)
                continue
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


def _get_camera_offset(diag: float, direction: str = "Z", distance_mult: float = 2.0) -> np.ndarray:
    """Return the camera offset vector based on viewing direction."""
    dist = diag * distance_mult
    offset_map = {
        "X": np.array([dist, 0.0, 0.0]),
        "Y": np.array([0.0, dist, 0.0]),
        "Z": np.array([0.0, 0.0, dist]),
    }
    return offset_map.get(direction, offset_map["Z"])


def _compute_camera_pose(
    vertices: np.ndarray,
    direction: str = "Z",
    distance_mult: float = 2.0,
    elevation_angle: float = 0.0,
) -> np.ndarray:
    """Compute camera pose that frames vertices and always looks at centroid."""
    centroid = vertices.mean(axis=0)
    diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if diag < 1e-6:
        diag = 1.0

    base_offset = _get_camera_offset(diag, direction, distance_mult)

    world_up = np.array([0.0, 0.0, 1.0]) if direction in {"X", "Y"} else np.array([0.0, 1.0, 0.0])

    elev_rad = np.radians(elevation_angle)
    right_vec = np.cross(base_offset, world_up)
    if np.linalg.norm(right_vec) < 1e-6:
        world_up = np.array([1.0, 0.0, 0.0]) if direction != "X" else np.array([0.0, 1.0, 0.0])
        right_vec = np.cross(base_offset, world_up)
    right_vec = right_vec / np.linalg.norm(right_vec)
    tilt_vec = np.cross(right_vec, base_offset)
    tilt_vec = tilt_vec / np.linalg.norm(tilt_vec)

    tilted_offset = base_offset * np.cos(elev_rad) + tilt_vec * np.linalg.norm(base_offset) * np.sin(elev_rad)
    cam_pos = centroid + tilted_offset

    forward = centroid - cam_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([1.0, 0.0, 0.0]) if direction != "X" else np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

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


def _add_text_overlay(img_array: np.ndarray, epoch: int, camera_dir: str) -> np.ndarray:
    """Add epoch number and camera direction text overlay."""
    img = Image.fromarray(img_array, "RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()  # type: ignore[assignment]
    text = f"Epoch: {epoch:05d}  |  View: {camera_dir}-axis"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = float(bbox[2] - bbox[0])
    text_height = float(bbox[3] - bbox[1])
    padding = 10.0
    bg_coords = (10.0, 10.0, 10.0 + text_width + 2 * padding, 10.0 + text_height + 2 * padding)
    draw.rectangle(bg_coords, fill=(0, 0, 0, 180))
    draw.text((10.0 + padding, 10.0 + padding), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def _render_frame(
    renderer: pyrender.OffscreenRenderer,
    fpath: Path,
    cam_pose: np.ndarray,
    camera_direction: str = CAMERA_DIRECTION,
    add_overlay: bool = True,
) -> np.ndarray:
    """Render one PLY file to an (H, W, 3) uint8 RGB image with optional text overlay."""
    scene = pyrender.Scene(
        bg_color=BACKGROUND_COLOR,
        ambient_light=[0.4, 0.4, 0.4, 1.0],
    )
    camera = pyrender.PerspectiveCamera(
        yfov=np.pi / 4.0,
        aspectRatio=WINDOW_WIDTH / WINDOW_HEIGHT,
    )
    scene.add(camera, pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)
    for geo in _load_pyrender_geometries(fpath):
        scene.add(geo)
    color, _ = renderer.render(scene)
    match = re.search(r"epoch_?(\d{5})", fpath.name)
    epoch = int(match.group(1)) if match else 0
    if add_overlay:
        color = _add_text_overlay(color, epoch, camera_direction)
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
    print(f"GIFs will be saved to: {output_path}")

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

    renderer = pyrender.OffscreenRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, point_size=POINT_SIZE)
    try:
        for slot, files in sorted(slots.items()):
            files.sort(key=lambda x: x[0])
            file_paths = [fp for _, fp in files]
            print(f"\nRendering Slot {slot} ({len(files)} frames)...")

            if cam_pose_override is not None:
                cam_pose = cam_pose_override
            else:
                all_verts_list = [v for v in (_get_vertices(fp) for fp in file_paths) if v is not None]
                if not all_verts_list:
                    print(f"  Warning: No vertex data for slot {slot}. Skipping.")
                    continue
                cam_pose = _compute_camera_pose(
                    np.vstack(all_verts_list),
                    direction=CAMERA_DIRECTION,
                    distance_mult=CAMERA_DISTANCE_MULTIPLIER,
                    elevation_angle=CAMERA_ELEVATION_ANGLE,
                )
            frames = [
                _render_frame(renderer, fp, cam_pose, camera_direction=CAMERA_DIRECTION, add_overlay=True) for fp in file_paths
            ]
            gif_path = output_path / f"slot{slot}_convergence.gif"
            imageio.mimsave(str(gif_path), frames, fps=FPS)  # type: ignore[arg-type]
            print(f"  Saved: {gif_path}")
    finally:
        renderer.delete()


def generate_gifs_flat(
    input_dir: Path,
    output_path: Path | None = None,
    cam_json: Path | None = None,
    max_frames: int | None = None,
) -> None:
    """Render a flat ``epoch_XXXXX.ply`` sequence to a single animation GIF.

    Scans ``input_dir`` for files matching ``epoch_XXXXX.ply``, sorts them
    by epoch number, optionally downsamples to ``max_frames``, and renders
    every frame with the same camera pose computed from the union of all
    vertex positions.

    Args:
        input_dir: Directory containing ``epoch_XXXXX.ply`` files.
        output_path: Destination path for the GIF.  Defaults to
            ``input_dir / "animation.gif"``.
        cam_json: Optional path to a ``camera_config.json`` from a prior
            interactive setup.  Overrides auto-computed camera when valid.
        max_frames: Maximum number of frames to include in the GIF.
            Frames are drawn uniformly from the sorted sequence.  If
            ``None``, the value from ``_CFG["gif"]["max_frames"]`` is used;
            if that is also ``None`` or missing, no downsampling is applied.

    Raises:
        ValueError: If no ``epoch_XXXXX.ply`` files are found in
            ``input_dir``.
    """
    t_start = time.monotonic()

    input_dir = Path(input_dir)
    if output_path is None:
        output_path = input_dir / "animation.gif"

    # Collect and sort matching files
    matches: list[tuple[int, Path]] = []
    for fpath in input_dir.iterdir():
        m = _FLAT_PATTERN.match(fpath.name)
        if m:
            matches.append((int(m.group(1)), fpath))

    if not matches:
        raise ValueError(
            f"No epoch_XXXXX.ply files found in {input_dir}.  Expected files named epoch_00000.ply, epoch_00001.ply, etc."
        )

    matches.sort(key=lambda x: x[0])
    file_paths = [fp for _, fp in matches]

    # Determine effective max_frames
    if max_frames is None:
        gif_cfg = _CFG.get("gif", {})
        max_frames = gif_cfg.get("max_frames")

    if max_frames is not None and len(file_paths) > max_frames:
        # Uniformly downsample: pick max_frames evenly-spaced indices
        indices = np.round(np.linspace(0, len(file_paths) - 1, max_frames)).astype(int)
        file_paths = [file_paths[i] for i in indices]

    print(
        json.dumps(
            {
                "event": "gif_flat_render_started",
                "input_dir": str(input_dir),
                "output_path": str(output_path),
                "n_frames": len(file_paths),
            }
        )
    )

    # Build camera pose
    cam_pose_override: np.ndarray | None = None
    if cam_json is not None and Path(cam_json).exists():
        cam_pose_override = _load_camera_pose(Path(cam_json))

    if cam_pose_override is not None:
        cam_pose = cam_pose_override
    else:
        all_verts = [v for v in (_get_vertices(fp) for fp in file_paths) if v is not None]
        if not all_verts:
            raise ValueError(f"No vertex data could be read from PLY files in {input_dir}.")
        cam_pose = _compute_camera_pose(
            np.vstack(all_verts),
            direction=CAMERA_DIRECTION,
            distance_mult=CAMERA_DISTANCE_MULTIPLIER,
            elevation_angle=CAMERA_ELEVATION_ANGLE,
        )

    gif_fps: int = _CFG.get("gif", {}).get("fps", FPS)
    renderer = pyrender.OffscreenRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, point_size=POINT_SIZE)
    try:
        frames = [_render_frame(renderer, fp, cam_pose, camera_direction=CAMERA_DIRECTION, add_overlay=True) for fp in file_paths]
    finally:
        renderer.delete()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=gif_fps)  # type: ignore[arg-type]

    elapsed = time.monotonic() - t_start
    print(
        json.dumps(
            {
                "event": "gif_flat_render_completed",
                "output_path": str(output_path),
                "n_frames": len(frames),
                "elapsed_s": round(elapsed, 2),
            }
        )
    )


_EPOCH_DIR_PATTERN = re.compile(r"^epoch_(\d{5})$")
_SLOT_SUBDIR_PATTERN = re.compile(r"^slot_(\d{2})$")


def generate_gifs_slotted(base_dir: Path, max_frames: int | None = None) -> None:
    """Generate per-slot animation GIFs from an epoch-first directory layout.

    Scans ``base_dir`` for subdirectories named ``epoch_XXXXX/``, collects
    ``slot_XX/mesh.ply`` files from each epoch, and writes
    ``epoch_LAST/slot_XX/animation.gif`` into the last epoch's slot directory,
    alongside its ``mesh.ply``.

    This is the counterpart of :func:`generate_gifs_flat` for the epoch-first
    diversity-training layout produced when ``set_active_slot()`` is active.

    Args:
        base_dir: Directory whose immediate subdirs follow ``epoch_XXXXX/``
            and contain ``slot_XX/mesh.ply`` files.
        max_frames: Maximum GIF frames per slot.  ``None`` falls back to the
            vis_config ``gif.max_frames`` setting, then no limit.

    Raises:
        ValueError: If no ``epoch_XXXXX/`` subdirectories are found.
    """
    base_dir = Path(base_dir)
    epoch_subdirs = sorted(
        (d for d in base_dir.iterdir() if d.is_dir() and _EPOCH_DIR_PATTERN.match(d.name)),
        key=lambda d: d.name,
    )
    if not epoch_subdirs:
        raise ValueError(f"No epoch_XXXXX/ subdirectories found in {base_dir}.")

    slot_indices: set[int] = set()
    for ep_dir in epoch_subdirs:
        for sd in ep_dir.iterdir():
            m = _SLOT_SUBDIR_PATTERN.match(sd.name)
            if m and sd.is_dir() and (sd / "mesh.ply").exists():
                slot_indices.add(int(m.group(1)))

    if not slot_indices:
        raise ValueError(f"No slot_XX/mesh.ply files found under epoch subdirectories in {base_dir}.")

    if max_frames is None:
        gif_cfg = _CFG.get("gif", {})
        max_frames = gif_cfg.get("max_frames")

    last_epoch_dir = epoch_subdirs[-1]

    for slot_idx in sorted(slot_indices):
        slot_name = f"slot_{slot_idx:02d}"
        slot_files: list[Path] = [
            ep_dir / slot_name / "mesh.ply"
            for ep_dir in epoch_subdirs
            if (ep_dir / slot_name / "mesh.ply").exists()
        ]
        if not slot_files:
            continue

        if max_frames is not None and len(slot_files) > max_frames:
            indices = np.round(np.linspace(0, len(slot_files) - 1, max_frames)).astype(int)
            slot_files = [slot_files[i] for i in indices]

        print(
            json.dumps(
                {
                    "event": "gif_slotted_render_started",
                    "base_dir": str(base_dir),
                    "slot": slot_name,
                    "n_frames": len(slot_files),
                }
            )
        )

        all_verts = [v for v in (_get_vertices(fp) for fp in slot_files) if v is not None]
        if not all_verts:
            continue
        cam_pose = _compute_camera_pose(
            np.vstack(all_verts),
            direction=CAMERA_DIRECTION,
            distance_mult=CAMERA_DISTANCE_MULTIPLIER,
            elevation_angle=CAMERA_ELEVATION_ANGLE,
        )

        gif_fps: int = _CFG.get("gif", {}).get("fps", FPS)
        renderer = pyrender.OffscreenRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, point_size=POINT_SIZE)
        try:
            frames = [
                _render_frame(renderer, fp, cam_pose, camera_direction=CAMERA_DIRECTION, add_overlay=True)
                for fp in slot_files
            ]
        finally:
            renderer.delete()

        # Write animation.gif alongside the PLY in the last epoch's slot dir.
        output_path = last_epoch_dir / slot_name / "animation.gif"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(output_path), frames, fps=gif_fps)  # type: ignore[arg-type]

        print(
            json.dumps(
                {
                    "event": "gif_slotted_render_completed",
                    "output_path": str(output_path),
                    "n_frames": len(frames),
                }
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PLY file sequences to GIFs. Cluster-safe (EGL headless).")
    parser.add_argument("--input", type=str, help="Input folder containing PLY files.")
    parser.add_argument("--output", type=str, help="Output folder for GIF files.")
    parser.add_argument("--cam_json", type=str, help="Path to camera_config.json from a prior interactive setup.")
    parser.add_argument(
        "--camera-direction",
        type=str,
        default=CAMERA_DIRECTION,
        choices=["X", "Y", "Z"],
        help=f"Camera view direction (default: {CAMERA_DIRECTION}).",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=CAMERA_DISTANCE_MULTIPLIER,
        help=f"Camera distance multiplier (default: {CAMERA_DISTANCE_MULTIPLIER}).",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=CAMERA_ELEVATION_ANGLE,
        help=f"Camera elevation angle in degrees (default: {CAMERA_ELEVATION_ANGLE}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CAMERA_DIRECTION = args.camera_direction
    CAMERA_DISTANCE_MULTIPLIER = args.camera_distance
    CAMERA_ELEVATION_ANGLE = args.camera_elevation
    in_p: Path | None
    out_p: Path | None
    c_j: Path | None
    if args.input:
        in_p = Path(args.input).resolve()
        cam_p: Path | None = Path(args.cam_json).resolve() if args.cam_json else None
        # Auto-detect: flat epoch_*.ply takes priority over slot pattern
        flat_files = [f for f in in_p.glob("*.ply") if _FLAT_PATTERN.match(f.name)]
        if flat_files:
            gif_out = Path(args.output).resolve() if args.output else in_p / "animation.gif"
            generate_gifs_flat(in_p, output_path=gif_out, cam_json=cam_p)
        else:
            out_p = Path(args.output).resolve() if args.output else in_p.parent / "pretraining_latent_gifs"
            generate_gifs(in_p, out_p, cam_json=cam_p)
    else:
        in_p, out_p, c_j = interactive_setup()
        if in_p is not None and out_p is not None:
            flat_files = [f for f in in_p.glob("*.ply") if _FLAT_PATTERN.match(f.name)]
            if flat_files:
                generate_gifs_flat(in_p, cam_json=c_j)
            else:
                generate_gifs(in_p, out_p, cam_json=c_j)
