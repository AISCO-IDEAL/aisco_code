"""Utility for exporting a combined inner bbox + outer bbox + SDF mesh as one vertex-coloured PLY.

The three mesh components are merged into a single trimesh.Trimesh with per-vertex
RGBA colours, then exported via trimesh to a binary PLY file.  This is purely an
I/O helper — no rendering, no CadQuery, no kaolin at runtime.

Typical call site: MeshAssemblyModeV4.update()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh


def export_combined_bbox_mesh_ply(
    *,
    inner_verts: torch.Tensor,
    inner_faces: torch.Tensor,
    outer_verts: torch.Tensor,
    outer_faces: torch.Tensor,
    sdf_verts: torch.Tensor,
    sdf_faces: torch.Tensor,
    output_path: Path,
    inner_color: tuple[float, float, float],
    outer_color: tuple[float, float, float],
    sdf_color: tuple[float, float, float],
    inner_alpha: float = 1.0,
    outer_alpha: float = 1.0,
    sdf_alpha: float = 1.0,
) -> None:
    """Export inner bbox + outer bbox + SDF mesh as a single vertex-coloured PLY.

    All three meshes are concatenated, each component given a uniform per-vertex
    RGBA colour, and written to *output_path* via trimesh.  Tensors may reside on
    any device and may carry gradients — they are detached and moved to CPU
    automatically.

    The alpha channel is encoded in the PLY vertex-color data.  Most mesh viewers
    (e.g. MeshLab, Blender, ParaView) honour it when rendering with transparency.

    Args:
        inner_verts: ``[N, 3]`` float32 tensor — inner bounding-box vertices.
        inner_faces: ``[F, 3]`` int64 tensor  — inner bounding-box face indices.
        outer_verts: ``[M, 3]`` float32 tensor — outer bounding-box vertices.
        outer_faces: ``[G, 3]`` int64 tensor  — outer bounding-box face indices.
        sdf_verts:   ``[P, 3]`` float32 tensor — SDF mesh vertices.
        sdf_faces:   ``[Q, 3]`` int64 tensor  — SDF mesh face indices.
        output_path: Destination file; parent directory must exist.
        inner_color: RGB triple in [0, 1] for the inner bbox component.
        outer_color: RGB triple in [0, 1] for the outer bbox component.
        sdf_color:   RGB triple in [0, 1] for the SDF mesh component.
        inner_alpha: Opacity of the inner bbox component in [0, 1]. Default: 1.0 (opaque).
        outer_alpha: Opacity of the outer bbox component in [0, 1]. Default: 1.0 (opaque).
        sdf_alpha:   Opacity of the SDF mesh component in [0, 1]. Default: 1.0 (opaque).

    Raises:
        ValueError: If *inner*, *outer*, or *sdf* mesh contains zero faces.
    """
    # --- Validate inputs -------------------------------------------------------
    if inner_faces.shape[0] == 0:
        msg = "inner mesh has zero faces. Ensure the inner bounding-box solid was tessellated successfully before calling export."
        raise ValueError(msg)
    if outer_faces.shape[0] == 0:
        msg = "outer mesh has zero faces. Ensure the outer bounding-box solid was tessellated successfully before calling export."
        raise ValueError(msg)
    if sdf_faces.shape[0] == 0:
        msg = "sdf mesh has zero faces. Ensure the SDF model produced a non-empty mesh before calling export."
        raise ValueError(msg)

    # --- Move all tensors to CPU numpy -----------------------------------------
    def _to_np_verts(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().float().numpy()

    def _to_np_faces(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().long().numpy()

    iv = _to_np_verts(inner_verts)
    i_f = _to_np_faces(inner_faces)
    ov = _to_np_verts(outer_verts)
    o_f = _to_np_faces(outer_faces)
    sv = _to_np_verts(sdf_verts)
    s_f = _to_np_faces(sdf_faces)

    # --- Merge vertices with face-index offsets --------------------------------
    all_verts = np.concatenate([iv, ov, sv], axis=0)

    inner_offset = iv.shape[0]
    sdf_offset = iv.shape[0] + ov.shape[0]
    all_faces = np.concatenate(
        [
            i_f,
            o_f + inner_offset,
            s_f + sdf_offset,
        ],
        axis=0,
    )

    # --- Build per-vertex RGBA colours (uint8) ---------------------------------
    def _color_block(n: int, rgb: tuple[float, float, float], alpha: float) -> np.ndarray:
        r, g, b = (round(c * 255) for c in rgb)
        a = round(alpha * 255)
        block = np.empty((n, 4), dtype=np.uint8)
        block[:] = [r, g, b, a]
        return block

    vertex_colors = np.concatenate(
        [
            _color_block(iv.shape[0], inner_color, inner_alpha),
            _color_block(ov.shape[0], outer_color, outer_alpha),
            _color_block(sv.shape[0], sdf_color, sdf_alpha),
        ],
        axis=0,
    )

    # --- Assemble and export ---------------------------------------------------
    mesh = trimesh.Trimesh(
        vertices=all_verts,
        faces=all_faces,
        vertex_colors=vertex_colors,
        process=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))
