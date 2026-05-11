"""Module-level picklable task functions for the vis manager thread/process pools.

Must live at module scope so ProcessPoolExecutor can pickle them.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]


def write_ply_task(
    out_path: Path,
    verts_np: np.ndarray,
    faces_np: np.ndarray | None,
    rgb_np: np.ndarray,
    add_coord_frame: bool,
    coord_frame_scale: float,
) -> tuple[bool, str, str | None]:
    try:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_np)
        if faces_np is not None:
            mesh.triangles = o3d.utility.Vector3iVector(faces_np)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgb_np.astype(np.float64))
        if add_coord_frame:
            diag = float(np.linalg.norm(verts_np.max(axis=0) - verts_np.min(axis=0)))
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=coord_frame_scale * max(diag, 1e-6), origin=[0.0, 0.0, 0.0]
            )
            mesh = mesh + frame
        o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=False)
        return (True, str(out_path), None)
    except Exception as exc:
        return (False, str(out_path), str(exc))


def render_gif_task(type_dir: Path) -> tuple[bool, str, str | None]:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.ply_to_gif import (  # noqa: PLC0415
            generate_gifs_flat,
        )

        generate_gifs_flat(type_dir)
        return (True, str(type_dir), None)
    except Exception as exc:
        return (False, str(type_dir), str(exc))
