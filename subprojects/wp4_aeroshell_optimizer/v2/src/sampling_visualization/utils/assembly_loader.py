"""Load and tessellate the inner/outer bbox assembly meshes from WP2 regulations.

Called once at CentralVisManagerV2 init when mesh_assembly_v4 is enabled.
Raises on any failure so the caller can log and fall back gracefully.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.custom_modes.bounding_box_mode_utils import (
    load_bb_assembly_id,
    load_inner_outer_solids,
    merge_meshes,
    solid_to_kaolin_mesh,
)


def _cpu(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.from_numpy(np.asarray(x))


def load_assembly_meshes(
    tessellation_tolerance: float = 2.0,
    angular_tolerance: float = 0.4,
) -> tuple:
    """Tessellate inner/outer bbox solids and return a stored assembly tuple.

    Returns:
        ``(inn_v, inn_f, out_v, out_f, sdf_color, inner_color, outer_color,
        sdf_alpha, outer_alpha)`` where mesh tensors are CPU torch.Tensors.

    Raises:
        Any exception from WP2 (missing assembly_id, CAD failure, etc.) —
        the caller is responsible for catching and logging.
    """
    assembly_id = load_bb_assembly_id()
    inner_solids, outer_solid = load_inner_outer_solids(assembly_id)

    inner_meshes = [
        solid_to_kaolin_mesh(s, device="cpu", tolerance=tessellation_tolerance, angular_tolerance=angular_tolerance)
        for s in inner_solids
    ]
    inner_verts, inner_faces, _ = merge_meshes(inner_meshes)
    outer_verts, outer_faces = solid_to_kaolin_mesh(
        outer_solid, device="cpu", tolerance=tessellation_tolerance, angular_tolerance=angular_tolerance
    )

    return (
        _cpu(inner_verts), _cpu(inner_faces),
        _cpu(outer_verts), _cpu(outer_faces),
        (0.2, 1.0, 1.0),   # sdf_color  — cyan
        (1.0, 0.5, 0.0),   # inner_color — orange
        (1.0, 0.5, 0.0),   # outer_color — orange
        0.5,                # sdf_alpha
        0.2,                # outer_alpha
    )
