from __future__ import annotations

from typing import Any

import numpy as np
import torch

from subprojects.wp2_assembler.version4.components.nbz import call_assembly
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config


def load_bb_assembly_id(default_id: int = 0) -> int:
    general_config = load_config(cfg_name="general_config.yaml")
    losses_cfg_name = general_config["active_configs"]["losses"]
    losses_cfg = load_config(cfg_name=losses_cfg_name)
    return int(losses_cfg.get("bb_config", {}).get("assembly_id", default_id))


def load_inner_outer_solids(assembly_id: int):
    inner_assy, outer_solid, __ = call_assembly(assembly_id)
    inner_solids = list(inner_assy.toCompound().Solids())
    return inner_solids, outer_solid


def solid_to_kaolin_mesh(
    solid: Any,
    device: str,
    tolerance: float = 2.0,
    angular_tolerance: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    vertices_cq, faces_cq = solid.tessellate(tolerance, angular_tolerance)
    vertices_np = np.array([[v.x, v.y, v.z] for v in vertices_cq], dtype=np.float32)
    faces_np = np.array(faces_cq, dtype=np.int64)
    vertices = torch.from_numpy(vertices_np).to(device=device)
    faces = torch.from_numpy(faces_np).to(device=device)
    return vertices, faces


def merge_meshes(meshes: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    all_vertices: list[torch.Tensor] = []
    all_faces: list[torch.Tensor] = []
    per_mesh_face_counts: list[int] = []

    v_offset = 0
    for vertices, faces in meshes:
        all_vertices.append(vertices)
        all_faces.append(faces + v_offset)
        per_mesh_face_counts.append(int(faces.shape[0]))
        v_offset += int(vertices.shape[0])

    return torch.cat(all_vertices, dim=0), torch.cat(all_faces, dim=0), per_mesh_face_counts
