from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.utils.coloring import solid_rgb

_DARK_BLUE: tuple[float, float, float] = (0.0, 0.0, 0.546)


def save_mesh_ply(
    path: Path | str,
    verts: np.ndarray,
    faces: np.ndarray | None = None,
    vertex_rgb: np.ndarray | None = None,
) -> Path:
    """Save a mesh or point cloud to a binary PLY file with vertex colors.

    Args:
        path: Output file path. Parent directory must exist.
        verts: Vertex positions, shape (V, 3). Accepts torch tensors.
        faces: Optional face indices, shape (F, 3).
        vertex_rgb: Per-vertex RGB colors, float32 in [0, 1], shape (V, 3).
            Falls back to a uniform dark-blue when None.

    Returns:
        Path to the written PLY file.
    """
    if hasattr(verts, "detach"):
        verts_np = verts.detach().cpu().numpy().astype(np.float64)
    else:
        verts_np = np.asarray(verts, dtype=np.float64)

    rgb = np.asarray(
        vertex_rgb if vertex_rgb is not None else solid_rgb(len(verts_np), _DARK_BLUE),
        dtype=np.float64,
    )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_np)
    if faces is not None:
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(rgb)

    out_path = Path(path)
    o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=False)
    return out_path


def save_scene_glb(
    path: Path | str,
    meshes: list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None]],
) -> Path:
    """Export multiple named meshes to a single GLB file.

    Each mesh becomes a named object in the GLB scene, which maps to a named
    child in three.js — enabling per-mesh visibility and opacity controls in
    the localhost viewer.

    Args:
        path: Output .glb file path.
        meshes: List of (name, verts (V,3), faces (F,3), vertex_rgb (V,3) float32 | None).
            Pass None for vertex_rgb to use a neutral grey material.

    Returns:
        Path to the written GLB file.
    """
    scene = trimesh.Scene()
    for name, verts, faces, vertex_rgb in meshes:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(verts, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int32),
            process=False,
        )
        if vertex_rgb is not None:
            rgb_f32 = np.asarray(vertex_rgb, dtype=np.float32)
            alpha = np.ones((len(rgb_f32), 1), dtype=np.float32)
            rgba_uint8 = (np.hstack([rgb_f32, alpha]) * 255).astype(np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba_uint8)
        scene.add_geometry(mesh, geom_name=name)

    out_path = Path(path)
    scene.export(str(out_path))
    return out_path
