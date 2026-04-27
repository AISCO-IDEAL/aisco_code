"""
PLY sequence to GIF renderer compatible with headless HPC cluster environments.

Uses pyrender with EGL (PYOPENGL_PLATFORM=egl), which works without a hardware
GPU or display server. Open3D is only imported when an interactive camera window
is requested and a DISPLAY is detected.

Usage (headless):
    python ply_to_gif.py --input /path/to/ply_folder
Usage (with interactive camera):
    DISPLAY=:0 python ply_to_gif.py  # prompts for folder, opens O3D window
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


def _load_pyrender_mesh(fpath: Path) -> pyrender.Mesh | None:
    """Load a PLY file and return a pyrender Mesh.

    Handles solid mesh, point cloud, and mixed files without triggering
    Open3D geometry-type warnings.

    Args:
        fpath: Path to the PLY file.

    Returns:
        A pyrender.Mesh ready to add to a scene, or None if the file
        contains no usable geometry.
    """
    loaded = trimesh.load(str(fpath), process=False)

    if isinstance(loaded, trimesh.Scene):
        solid_meshes = [
            g for g in loaded.geometry.values()
            if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0
        ]
        if solid_meshes:
            merged = trimesh.util.concatenate(solid_meshes)
            return pyrender.Mesh.from_trimesh(merged, smooth=False)
        # Scene with no solid faces — fall through to point cloud path
        parts = list(loaded.geometry.values())
        if not parts:
            return None
        loaded = parts[0]

    if isinstance(loaded, trimesh.Trimesh):
        if len(loaded.faces) > 0:
            return pyrender.Mesh.from_trimesh(loaded, smooth=False)
        # Trimesh with zero faces is a vertex-only (point cloud) PLY
        points = np.asarray(loaded.vertices, dtype=np.float32)
    elif isinstance(loaded, trimesh.PointCloud):
        points = np.asarray(loaded.vertices, dtype=np.float32)
    else:
        return None

    if len(points) == 0:
        return None

    colors = np.tile(
        np.array([0.2, 0.55, 0.85, 1.0], dtype=np.float32), (len(points), 1)
    )
    primitive = pyrender.Primitive(
        positions=points,
        color_0=colors,
        mode=pyrender.GLTF.POINTS,
    )
    return pyrender.Mesh(primitives=[primitive])


def _get_vertices(fpath: Path) -> np.ndarray | None:
    """Return all vertex positions from a PLY file as an (N, 3) array."""
    loaded = trimesh.load(str(fpath), process=False)

    if isinstance(loaded, trimesh.Scene):
        parts = [
            np.asarray(g.vertices)
            for g in loaded.geometry.values()
            if hasattr(g, "vertices") and len(g.vertices) > 0
        ]
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
        bg_color=[1.0, 1.0, 1.0, 1.0],
        ambient_light=[0.4, 0.4, 0.4, 1.0],
    )
    camera = pyrender.PerspectiveCamera(
        yfov=np.pi / 4.0,
        aspectRatio=WINDOW_WIDTH / WINDOW_HEIGHT,
    )
    scene.add(camera, pose=cam_pose)
    # Directional light co-located with camera so the object is always lit
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)

    geo = _load_pyrender_mesh(fpath)
    if geo is not None:
        scene.add(geo)

    color, _ = renderer.render(scene)
    return color


def _try_interactive_camera(first_ply: Path, output_dir: Path) -> Path | None:
    """Open an Open3D window so the user can set the camera, then save the pose.

    Only called when DISPLAY is set. Any failure returns None gracefully
    so the caller can fall back to auto-computed camera.

    Args:
        first_ply: Sample PLY file to display in the window.
        output_dir: Directory in which to write camera_config.json.

    Returns:
        Path to the saved camera JSON, or None on failure.
    """
    try:
        import open3d as o3d  # noqa: PLC0415 — conditional import, not available on all nodes

        vis = o3d.visualization.Visualizer()
        if not vis.create_window("Set Camera Angle", width=WINDOW_WIDTH, height=WINDOW_HEIGHT):
            print("Warning: Could not open visual window. Auto-computing camera.")
            return None

        pcd = o3d.io.read_point_cloud(str(first_ply))
        mesh = o3d.io.read_triangle_mesh(str(first_ply))
        if len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            vis.add_geometry(mesh)
        elif len(pcd.points) > 0:
            vis.add_geometry(pcd)

        render_opt = vis.get_render_option()
        render_opt.point_size = 3.0
        render_opt.background_color = np.array([1.0, 1.0, 1.0])

        print("INSTRUCTIONS: Position the camera, then CLOSE the window.")
        vis.run()

        # convert_to_pinhole_camera_parameters returns world-to-camera (extrinsic)
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        extrinsic = np.array(param.extrinsic)
        cam_pose = np.linalg.inv(extrinsic).tolist()  # camera-to-world for pyrender
        vis.destroy_window()

        cam_json_path = output_dir / "camera_config.json"
        with open(cam_json_path, "w") as f:
            json.dump({"pose": cam_pose}, f, indent=2)
        print(f"Camera saved to: {cam_json_path}")
        return cam_json_path

    except Exception as exc:
        print(f"Warning: Interactive camera setup failed ({exc}). Auto-computing camera.")
        return None


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
                all_verts_list = [
                    v for v in (_get_vertices(fp) for fp in file_paths)
                    if v is not None
                ]
                if not all_verts_list:
                    print(f"  Warning: No vertex data for slot {slot}. Skipping.")
                    continue
                cam_pose = _compute_camera_pose(np.vstack(all_verts_list))

            frames = [_render_frame(renderer, fp, cam_pose) for fp in file_paths]
            gif_path = output_path / f"slot{slot}_convergence.gif"
            imageio.mimsave(str(gif_path), frames, fps=FPS)
            print(f"  Saved: {gif_path}")
    finally:
        renderer.delete()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render PLY file sequences to GIFs. Cluster-safe (EGL headless)."
    )
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
    if args.input:
        in_p = Path(args.input).resolve()
        out_p = (
            Path(args.output).resolve()
            if args.output
            else in_p.parent / "pretraining_latent_gifs"
        )
        cam_p = Path(args.cam_json).resolve() if args.cam_json else None
        generate_gifs(in_p, out_p, cam_json=cam_p)
    else:
        in_p, out_p, c_j = interactive_setup()
        if in_p is not None:
            generate_gifs(in_p, out_p, cam_json=c_j)
