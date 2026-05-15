"""
MRRT_SolarLoss: K solar directions with mesh raytracing per direction.

Soft budget mask is determined by average quality across all K directions (same as MR).
Raytracing and power accumulation loop over each direction individually; results are
averaged over K before returning.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from subprojects.wp3_solar_estimator.raytracing.diffable.version4.mesh_ray_visibility_v4 import (
    mesh_visibility_v4 as _default_mesh_visibility,
)
from subprojects.wp3_solar_estimator.utils.rollout_utils.mesh_area_normals import compute_face_areas_and_normals
from subprojects.wp3_solar_estimator.utils.solar_utils import averaging_solar_quality
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import NullVisManager
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import CentralVisManagerV2


def _efficient_cumsum(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    # torch.cumsum has no deterministic CUDA kernel — run on CPU and transfer back.
    return torch.cumsum(x.cpu(), dim=dim).to(x.device)


def solar_loss_mrrt(  # noqa: PLR0915
    vertices: torch.Tensor,
    faces: torch.Tensor,
    solar_vectors: torch.Tensor,
    area_budget_mm2: float,
    raytracing_cfg: dict[str, Any],
    effective_irradiance_W_m2: float = 250.0,
    sigmoid_sharpness: float = 10.0,
    soft_mask_selection_threshold: float = 0.5,
    soft_mask_full_selection_threshold: float = 0.95,
    shading_penalty_weight: float = 0.0,
    kappa: float = 1.0,
    _lambda: float = 1.0,
    epsilon: float = 1e-12,
    skip_backface_receivers: bool = False,
    vis_manager: CentralVisManagerV2 | NullVisManager = NullVisManager(),
    epoch_name: int | None = None,
    mesh_visibility_fn: Callable | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Multi-ray solar power objective with per-direction mesh raytracing.

    The soft budget mask uses mean quality across all K directions. Power and visibility are
    accumulated per direction then divided by K.

    Returns raw power in watts — caller is responsible for any further scaling.

    Args:
        vertices: (N, 3) mesh vertices in mm
        faces: (M, 3) triangle face indices
        solar_vectors: (K, 3) sun direction vectors (normalised internally)
        area_budget_mm2: maximum selectable area in mm²
        raytracing_cfg: dict with a "mesh" sub-dict passed directly to mesh_visibility_v4
        effective_irradiance_W_m2: electrical power output per m² of perfectly-aligned area
        sigmoid_sharpness: steepness of the soft budget boundary sigmoid
        soft_mask_selection_threshold: mask threshold for "selected" faces in diagnostics
        soft_mask_full_selection_threshold: mask threshold for "fully selected" faces in diagnostics
        shading_penalty_weight: scales penalty for occluded illuminated faces; 0 disables it
        kappa: exponent in numerical_factor = 1 - λ * (1 + ε - visibility)^(1/κ)
        _lambda: scale in numerical_factor formula
        epsilon: numerical stability floor
        skip_backface_receivers: when True, filters receiver faces to sun-facing-only per direction
            (where face_normal · sun_dir > 0); requires full occluder coverage
        vis_manager: pass NullVisManager() to skip PLY export
        epoch_name: epoch index forwarded to vis_manager; skipped when None
        mesh_visibility_fn: override default mesh_visibility_v4

    Returns:
        (power_watts, diagnostics) where diagnostics keys:
            total_power_watts, selected_area_m2, total_area_m2, total_faces,
            selected_faces, fully_selected_faces, transition_zone_faces,
            budget_crossover_idx, mean_visibility, shading_penalty, num_solar_vectors
    """
    _vis_fn = mesh_visibility_fn if mesh_visibility_fn is not None else _default_mesh_visibility

    if faces.numel() == 0 or vertices.numel() == 0:
        print("Warning: empty mesh in solar_loss_mrrt, returning zero power")
        zero = torch.zeros((), device=vertices.device, dtype=vertices.dtype)
        return zero, {
            "total_power_watts": 0.0,
            "selected_area_m2": 0.0,
            "total_area_m2": 0.0,
            "total_faces": 0,
            "selected_faces": 0,
            "fully_selected_faces": 0,
            "transition_zone_faces": 0,
            "budget_crossover_idx": 0,
            "mean_visibility": 0.0,
            "shading_penalty": 0.0,
            "num_solar_vectors": int(solar_vectors.shape[0]),
        }

    _face_area_mm2, face_area_m2, face_normals_unit = compute_face_areas_and_normals(vertices, faces, epsilon)

    solar_vectors_unit = solar_vectors / torch.linalg.norm(solar_vectors, dim=-1, keepdim=True).clamp_min(epsilon)
    K = solar_vectors_unit.shape[0]
    mesh_cfg = raytracing_cfg["mesh"]

    avg_quality = averaging_solar_quality(face_normals_unit, solar_vectors_unit)  # (F,)

    quality_order_desc = torch.argsort(avg_quality, descending=True)
    sorted_areas_m2 = face_area_m2[quality_order_desc]

    cumulative_area_m2 = _efficient_cumsum(sorted_areas_m2, dim=0)
    area_budget_m2 = area_budget_mm2 * 1e-6

    distance_to_budget = (area_budget_m2 - cumulative_area_m2) / max(area_budget_m2, epsilon)
    soft_mask_sorted = torch.sigmoid(sigmoid_sharpness * distance_to_budget)

    original_order = torch.argsort(quality_order_desc)
    soft_mask_original = soft_mask_sorted[original_order]

    power_acc = torch.zeros((), device=vertices.device, dtype=vertices.dtype)
    visibility_acc = torch.zeros(faces.shape[0], device=vertices.device, dtype=vertices.dtype)
    penalty_acc = torch.zeros((), device=vertices.device, dtype=vertices.dtype)
    sun_facing_acc = (
        torch.zeros(faces.shape[0], device=vertices.device, dtype=vertices.dtype) if skip_backface_receivers else None
    )

    for k in range(K):
        sun_k = solar_vectors_unit[k]
        quality_k = (face_normals_unit * sun_k).sum(-1)

        sun_facing_indices_k = torch.where(quality_k > 0)[0] if skip_backface_receivers else None

        if sun_facing_indices_k is not None:
            vis_rt, _, _ = _vis_fn(
                vertices,
                faces[sun_facing_indices_k],
                sun_k,
                mesh_cfg=mesh_cfg,
                normal_source="mesh",
                occluder_faces=faces,
                receiver_face_indices=sun_facing_indices_k,
            )
            vis_k = torch.zeros(faces.shape[0], device=vertices.device, dtype=vertices.dtype)
            vis_k[sun_facing_indices_k] = vis_rt
        else:
            vis_k, _, _ = _vis_fn(vertices, faces, sun_k, mesh_cfg=mesh_cfg, normal_source="mesh")

        nf_k = 1 - _lambda * torch.pow(1 + epsilon - vis_k, 1 / kappa)
        p_k = face_area_m2 * torch.clamp(quality_k, min=0.0) * nf_k * effective_irradiance_W_m2
        power_acc = power_acc + (p_k * soft_mask_original).sum()

        vis_k_det = vis_k.detach()
        visibility_acc = visibility_acc + vis_k_det
        if shading_penalty_weight > 0:
            nf_k_diag = 1 - _lambda * torch.pow(1 + epsilon - vis_k_det, 1 / kappa)
            penalty_acc = penalty_acc + (
                shading_penalty_weight * ((1 - nf_k_diag) * torch.clamp(quality_k.detach(), min=0.0) * soft_mask_original).sum()
            )

        if sun_facing_acc is not None:
            sun_facing_acc = sun_facing_acc + (quality_k > 0).float()

    power_watts = power_acc / K
    if shading_penalty_weight > 0:
        power_watts = power_watts - penalty_acc / K

    mean_visibility = visibility_acc / K

    if epoch_name is not None:
        vis_manager.vis_solar_quality(int(epoch_name), vertices, faces, avg_quality)
        vis_manager.vis_solar_selection(int(epoch_name), vertices, faces, soft_mask_original)
        if skip_backface_receivers and sun_facing_acc is not None:
            vis_manager.vis_solar_receiver_mask(int(epoch_name), vertices, faces, sun_facing_acc / K)
        vis_manager.vis_solar_shading(int(epoch_name), vertices, faces, mean_visibility)
        numerical_factor_mean = 1 - _lambda * torch.pow(1 + epsilon - mean_visibility, 1 / kappa)
        vis_manager.vis_solar_numerical_factor(int(epoch_name), vertices, faces, numerical_factor_mean)
        vis_manager.vis_solar_penalty(
            int(epoch_name),
            vertices,
            faces,
            (1 - numerical_factor_mean) * torch.clamp(avg_quality, min=0.0) * soft_mask_original,
        )

    faces_near_boundary = (soft_mask_sorted > 0.1) & (soft_mask_sorted < 0.9)
    budget_crossover_idx = (
        torch.argmax((cumulative_area_m2 >= area_budget_m2).float()).item()
        if (cumulative_area_m2 >= area_budget_m2).any()
        else len(cumulative_area_m2)
    )

    diagnostics: dict[str, Any] = {
        "total_power_watts": power_watts.detach().item(),
        "selected_area_m2": (sorted_areas_m2 * soft_mask_sorted).sum().detach().item(),
        "total_area_m2": face_area_m2.sum().detach().item(),
        "total_faces": int(faces.shape[0]),
        "selected_faces": int((soft_mask_sorted > soft_mask_selection_threshold).sum().item()),
        "fully_selected_faces": int((soft_mask_sorted > soft_mask_full_selection_threshold).sum().item()),
        "transition_zone_faces": int(faces_near_boundary.sum().item()),
        "budget_crossover_idx": int(budget_crossover_idx),
        "mean_visibility": float(mean_visibility.mean().item()),
        "shading_penalty": float((penalty_acc / K).item()) if shading_penalty_weight > 0 else 0.0,
        "num_solar_vectors": K,
    }

    return power_watts, diagnostics
