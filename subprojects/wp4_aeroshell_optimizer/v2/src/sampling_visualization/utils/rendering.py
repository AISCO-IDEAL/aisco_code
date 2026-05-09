from __future__ import annotations

import os

# Must be set before any OpenGL / pyrender imports.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from pathlib import Path

import imageio
import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw, ImageFont


def compute_camera_pose(
    verts: np.ndarray,
    direction: str = "X",
    distance_mult: float = 2.5,
    elevation_angle: float = 20.0,
) -> np.ndarray:
    """Compute a camera pose that frames the given vertices and looks at their centroid.

    Args:
        verts: Vertex positions, shape (V, 3).
        direction: Primary viewing axis — "X", "Y", or "Z".
        distance_mult: Camera distance as a multiple of the bounding-box diagonal.
        elevation_angle: Vertical tilt above the primary viewing plane, in degrees.

    Returns:
        4x4 camera-to-world pose matrix (OpenGL convention).
    """
    centroid = verts.mean(axis=0)
    diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    if diag < 1e-6:
        diag = 1.0

    dist = diag * distance_mult
    offset_map = {
        "X": np.array([dist, 0.0, 0.0]),
        "Y": np.array([0.0, dist, 0.0]),
        "Z": np.array([0.0, 0.0, dist]),
    }
    base_offset = offset_map.get(direction, offset_map["Z"])
    world_up = np.array([0.0, 0.0, 1.0]) if direction in {"X", "Y"} else np.array([0.0, 1.0, 0.0])

    elev_rad = np.radians(elevation_angle)
    right_vec = np.cross(base_offset, world_up)
    if np.linalg.norm(right_vec) < 1e-6:
        world_up = np.array([1.0, 0.0, 0.0]) if direction != "X" else np.array([0.0, 1.0, 0.0])
        right_vec = np.cross(base_offset, world_up)
    right_vec /= np.linalg.norm(right_vec)
    tilt_vec = np.cross(right_vec, base_offset)
    tilt_vec /= np.linalg.norm(tilt_vec)

    tilted_offset = base_offset * np.cos(elev_rad) + tilt_vec * np.linalg.norm(base_offset) * np.sin(elev_rad)
    cam_pos = centroid + tilted_offset
    forward = (centroid - cam_pos) / np.linalg.norm(centroid - cam_pos)

    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([1.0, 0.0, 0.0]) if direction != "X" else np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = cam_pos
    return pose


def render_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    vertex_rgb: np.ndarray,
    cam_pose: np.ndarray,
    width: int,
    height: int,
    *,
    bg_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    renderer: pyrender.OffscreenRenderer | None = None,
) -> np.ndarray:
    """Render a single mesh with vertex colors to a uint8 RGB image.

    Args:
        verts: Vertex positions, shape (V, 3).
        faces: Face indices, shape (F, 3).
        vertex_rgb: Per-vertex RGB colors, float32 in [0, 1], shape (V, 3).
        cam_pose: 4x4 camera-to-world pose matrix (OpenGL convention).
        width: Output image width in pixels.
        height: Output image height in pixels.
        bg_color: RGBA background color, float in [0, 1].
        renderer: Optional long-lived OffscreenRenderer for batch efficiency.
            When None, a renderer is created and deleted per call.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    rgb_f32 = np.asarray(vertex_rgb, dtype=np.float32)
    rgba_uint8 = (np.hstack([rgb_f32, np.ones((len(rgb_f32), 1), dtype=np.float32)]) * 255).astype(np.uint8)

    tmesh = trimesh.Trimesh(
        vertices=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int32),
        process=False,
    )
    tmesh.visual = trimesh.visual.ColorVisuals(mesh=tmesh, vertex_colors=rgba_uint8)
    pr_mesh = pyrender.Mesh.from_trimesh(tmesh, smooth=False)

    scene = pyrender.Scene(bg_color=bg_color, ambient_light=[0.4, 0.4, 0.4, 1.0])
    scene.add(pr_mesh)
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=width / height), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)

    own_renderer = renderer is None
    if own_renderer:
        renderer = pyrender.OffscreenRenderer(width, height)
    assert renderer is not None  # for mypy
    try:
        color, _ = renderer.render(scene)
    finally:
        if own_renderer:
            renderer.delete()
    return color


def render_ply(
    ply_path: Path | str,
    cam_pose: np.ndarray,
    width: int,
    height: int,
    *,
    bg_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    renderer: pyrender.OffscreenRenderer | None = None,
) -> np.ndarray:
    """Load a PLY file and render it to a uint8 RGB image.

    Handles mixed PLY scenes (solid meshes and/or point clouds) and preserves
    vertex colors stored in the file.

    Args:
        ply_path: Path to the PLY file.
        cam_pose: 4x4 camera-to-world pose matrix (OpenGL convention).
        width: Output image width in pixels.
        height: Output image height in pixels.
        bg_color: RGBA background color, float in [0, 1].
        renderer: Optional long-lived OffscreenRenderer for batch efficiency.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    loaded = trimesh.load(str(ply_path), process=False)
    geos: list[object]
    if isinstance(loaded, trimesh.Scene):
        geos = list(loaded.geometry.values())
    elif isinstance(loaded, (trimesh.Trimesh, trimesh.PointCloud)):
        geos = [loaded]
    else:
        geos = []

    scene = pyrender.Scene(bg_color=bg_color, ambient_light=[0.4, 0.4, 0.4, 1.0])
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=width / height), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)

    for geo in geos:
        pr_mesh = _geo_to_pyrender(geo)
        if pr_mesh is not None:
            scene.add(pr_mesh)

    own_renderer = renderer is None
    if own_renderer:
        renderer = pyrender.OffscreenRenderer(width, height)
    assert renderer is not None  # for mypy
    try:
        color, _ = renderer.render(scene)
    finally:
        if own_renderer:
            renderer.delete()
    return color


def frames_to_gif(
    frames: list[np.ndarray],
    output_path: Path | str,
    fps: int = 10,
) -> Path:
    """Write a list of (H, W, 3) uint8 RGB frames to an animated GIF.

    Args:
        frames: List of RGB images, each (H, W, 3) uint8.
        output_path: Output GIF path.
        fps: Frames per second.

    Returns:
        Path to the written GIF.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), frames, fps=fps)  # type: ignore[arg-type]
    return out


def add_epoch_overlay(img_rgb: np.ndarray, epoch: int, label: str = "") -> np.ndarray:
    """Add an epoch number and optional label as a text overlay in the top-left corner.

    Args:
        img_rgb: (H, W, 3) uint8 RGB image.
        epoch: Epoch number to display.
        label: Optional extra label string appended after the epoch number.

    Returns:
        (H, W, 3) uint8 RGB image with text overlay burned in.
    """
    img = Image.fromarray(img_rgb, "RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()  # type: ignore[assignment]
    text = f"Epoch: {epoch:05d}" + (f"  |  {label}" if label else "")
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
    pad = 10.0
    draw.rectangle((10.0, 10.0, 10.0 + tw + 2 * pad, 10.0 + th + 2 * pad), fill=(0, 0, 0, 180))
    draw.text((10.0 + pad, 10.0 + pad), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def _geo_to_pyrender(geo: object) -> pyrender.Mesh | None:
    """Convert a trimesh geometry to a pyrender.Mesh, preserving vertex colors."""
    if isinstance(geo, trimesh.Trimesh) and len(geo.faces) > 0:
        return pyrender.Mesh.from_trimesh(geo, smooth=False)

    if isinstance(geo, (trimesh.Trimesh, trimesh.PointCloud)):
        points = np.asarray(geo.vertices, dtype=np.float32)
    else:
        return None

    if len(points) == 0:
        return None

    colors: np.ndarray | None = None
    if isinstance(geo, trimesh.PointCloud) and geo.colors is not None and len(geo.colors) > 0:
        colors = np.asarray(geo.colors, dtype=np.float32)
    elif isinstance(geo, trimesh.Trimesh):
        visual = getattr(geo, "visual", None)
        vc = getattr(visual, "vertex_colors", None) if visual is not None else None
        if vc is not None:
            vc = np.asarray(vc, dtype=np.float32)
            if vc.ndim == 2 and vc.shape[0] == len(geo.vertices):
                colors = vc

    if colors is None or len(colors) != len(points):
        colors = np.ones((len(points), 4), dtype=np.float32)
    else:
        if colors.max() > 1.0:
            colors = colors / 255.0
        if colors.shape[1] == 3:
            colors = np.hstack([colors, np.ones((len(colors), 1), dtype=np.float32)])
        colors = colors.clip(0.0, 1.0)

    return pyrender.Mesh(primitives=[pyrender.Primitive(positions=points, color_0=colors, mode=pyrender.GLTF.POINTS)])
