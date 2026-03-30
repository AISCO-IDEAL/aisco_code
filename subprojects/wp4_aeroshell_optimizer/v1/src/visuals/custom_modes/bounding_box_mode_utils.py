from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from subprojects.wp2_assembler.version4.components.nbz import call_assembly
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config


def load_bb_assembly_id(default_id: int | None = None) -> int:
    # Highest-priority override for reproducible runs.
    env_assembly_id = os.getenv("AISCO_ASSEMBLY_ID", "").strip()
    if env_assembly_id:
        return int(env_assembly_id)

    # Fallback to active losses config (bb_config.assembly_id).
    try:
        general_cfg = load_config(cfg_name="general_config.yaml")
        losses_cfg_name = general_cfg.get("active_configs", {}).get("losses")
        if losses_cfg_name:
            losses_cfg = load_config(cfg_name=losses_cfg_name)
            configured_assembly_id = losses_cfg.get("bb_config", {}).get("assembly_id")
            if configured_assembly_id is not None:
                return int(configured_assembly_id)
    except Exception:
        # Keep renderer usable even if config loading fails in edge environments.
        pass

    if default_id is not None:
        return int(default_id)

    msg = "Unable to resolve assembly_id from AISCO_ASSEMBLY_ID or losses config (bb_config.assembly_id)."
    raise ValueError(msg)


def load_inner_outer_solids(assembly_id: int):
    inner_assy, outer_solid, __ = call_assembly(id_number=assembly_id)
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


def get_mesh_from_optimization_state(optimization_state: dict[str, Any]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    vertices = optimization_state.get("vertices")
    if vertices is None:
        vertices = optimization_state.get("verts")

    faces = optimization_state.get("faces")
    if faces is None:
        faces = optimization_state.get("faces_in")

    if vertices is None or faces is None:
        return None, None

    return vertices.detach(), faces.detach()


def alpha_blend(base_rgb: np.ndarray, overlay_rgb: np.ndarray, mask_2d: np.ndarray, alpha: float) -> np.ndarray:
    m = np.clip(mask_2d, 0.0, 1.0)[..., None] * float(alpha)
    out = overlay_rgb * m + base_rgb * (1.0 - m)
    return np.clip(out, 0.0, 1.0)


def silhouette_edge_mask(mask_2d: np.ndarray) -> np.ndarray:
    m = (mask_2d > 0.5).astype(np.uint8)
    edge = np.zeros_like(m, dtype=np.uint8)
    edge[1:, :] |= m[1:, :] != m[:-1, :]
    edge[:, 1:] |= m[:, 1:] != m[:, :-1]
    edge[:-1, :] |= m[:-1, :] != m[1:, :]
    edge[:, :-1] |= m[:, :-1] != m[:, 1:]
    return edge.astype(np.float32)


def overlay_edges(
    base_rgb: np.ndarray, edge_mask_2d: np.ndarray, edge_color: tuple[float, float, float], strength: float
) -> np.ndarray:
    edge_rgb = np.array(edge_color, dtype=np.float32)[None, None, :]
    w = np.clip(edge_mask_2d, 0.0, 1.0)[..., None] * float(strength)
    out = base_rgb * (1.0 - w) + edge_rgb * w
    return np.clip(out, 0.0, 1.0)
