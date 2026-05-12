"""
SRRT_SolarLoss: single fixed sun direction with mesh raytracing (v3).

Extends SR by multiplying each face's power contribution by a differentiable
visibility factor computed via Möller-Trumbore ray-triangle intersection.

Note on back-face culling in mesh_visibility_v3: the raytracer tests all faces
as potential blockers without filtering by face_normal · sun_dir < 0 first. For
a watertight shell this is generally correct, but if you observe near-zero
visibility on all upward-facing faces, check that mesh_cfg["origin_offset_eps"]
is large enough to prevent self-intersection (the ray origin is nudged along the
face normal by this amount before casting toward the sun).
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from subprojects.wp3_solar_estimator.raytracing.diffable.version3.mesh_ray_visibility_v3 import mesh_visibility_v3
from subprojects.wp3_solar_estimator.utils.rollout_utils.mesh_area_normals import compute_face_areas_and_normals
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import NullVisManager
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import CentralVisManagerV2


def _efficient_cumsum(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    # torch.cumsum has no deterministic CUDA kernel — run on CPU and transfer back.
    return torch.cumsum(x.cpu(), dim=dim).to(x.device)


def solar_loss_srrt(  # noqa: PLR0912, PLR0915
    vertices: torch.Tensor,
    faces: torch.Tensor,
    sun_dir: torch.Tensor,
    area_budget_mm2: float,
    raytracing_cfg: dict[str, Any],
    solar_irradiance_W_m2: float = 1000.0,
    efficiency: float = 0.25,
    sigmoid_sharpness: float = 10.0,
    soft_mask_selection_threshold: float = 0.5,
    soft_mask_full_selection_threshold: float = 0.95,
    shading_penalty_weight: float = 0.0,
    kappa: float = 1.0,
    _lambda: float = 1.0,
    epsilon: float = 1e-12,
    use_activation_checkpointing: bool = True,
    vis_manager: CentralVisManagerV2 | NullVisManager = NullVisManager(),  # noqa: B008
    epoch_name: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Single-ray solar power objective with mesh raytracing visibility.

    Returns raw power in watts — caller is responsible for any further scaling.
    No wandb calls, no rendering, no epoch counters.

    Args:
        vertices: (N, 3) mesh vertices in mm
        faces: (M, 3) triangle face indices
        sun_dir: (3,) sun direction vector (normalised internally)
        area_budget_mm2: maximum selectable area in mm²
        raytracing_cfg: dict with a "mesh" sub-dict passed directly to mesh_visibility_v3
        solar_irradiance_W_m2: solar irradiance at surface in W/m²
        efficiency: system efficiency factor (0–1)
        sigmoid_sharpness: steepness of the soft budget boundary sigmoid
        soft_mask_selection_threshold: mask threshold for counting "selected" faces in diagnostics
        soft_mask_full_selection_threshold: mask threshold for counting "fully selected" faces in diagnostics
        shading_penalty_weight: scales a penalty term that reduces the objective by occluded power; 0 disables it
        kappa: exponent in numerical_factor = 1 - λ * (1 + ε - visibility)^(1/κ)
        _lambda: scale in numerical_factor formula above
        epsilon: numerical stability floor (also used in visibility formula as ε)
        use_activation_checkpointing: when True and vertices.requires_grad, wraps the visibility+loss
            compute in torch.utils.checkpoint to trade memory for recomputation, then runs a second
            detached no_grad pass for diagnostics/vis
        vis_manager: CentralVisManagerV2 or NullVisManager; pass NullVisManager() to skip export
        epoch_name: epoch index forwarded to vis_manager methods; skipped when None

    Returns:
        (power_watts, diagnostics) where diagnostics is a dict of detached scalar metrics:
            total_power_watts, selected_area_m2, total_area_m2, total_faces,
            selected_faces, fully_selected_faces, transition_zone_faces,
            budget_crossover_idx, mean_visibility, shading_penalty
    """
    if faces.numel() == 0 or vertices.numel() == 0:
        print("Warning: empty mesh in solar_loss_srrt, returning zero power")
        zero = torch.zeros((), device=vertices.device, dtype=vertices.dtype)
        return zero, {
            "total_power_watts": 0.0, "selected_area_m2": 0.0, "total_area_m2": 0.0,
            "total_faces": 0, "selected_faces": 0, "fully_selected_faces": 0,
            "transition_zone_faces": 0, "budget_crossover_idx": 0,
            "mean_visibility": 0.0, "shading_penalty": 0.0,
        }

    face_area_mm2, face_area_m2, face_normals_unit = compute_face_areas_and_normals(vertices, faces, epsilon)

    sun_dir_unit = sun_dir / torch.linalg.norm(sun_dir).clamp_min(epsilon)
    illumination_quality = (face_normals_unit * sun_dir_unit).sum(-1)

    quality_order_desc = torch.argsort(illumination_quality, descending=True)
    sorted_areas_m2 = face_area_m2[quality_order_desc]
    sorted_quality = illumination_quality[quality_order_desc]

    cumulative_area_m2 = _efficient_cumsum(sorted_areas_m2, dim=0)
    area_budget_m2 = area_budget_mm2 * 1e-6

    distance_to_budget = (area_budget_m2 - cumulative_area_m2) / max(area_budget_m2, epsilon)
    soft_mask_sorted = torch.sigmoid(sigmoid_sharpness * distance_to_budget)

    original_order = torch.argsort(quality_order_desc)
    soft_mask_original_order = soft_mask_sorted[original_order]

    mesh_cfg = raytracing_cfg["mesh"]

    def _compute_power(vtx: torch.Tensor) -> torch.Tensor:
        visibility, _, _ = mesh_visibility_v3(vtx, faces, sun_dir_unit, mesh_cfg=mesh_cfg, normal_source="mesh")
        numerical_factor = 1 - _lambda * torch.pow(1 + epsilon - visibility, 1 / kappa)
        power_per_face = (
            face_area_m2
            * torch.clamp(illumination_quality, min=0.0)
            * numerical_factor
            * solar_irradiance_W_m2
            * efficiency
        )
        return (power_per_face * soft_mask_original_order).sum()

    if use_activation_checkpointing and vertices.requires_grad:
        power_watts = checkpoint(_compute_power, vertices, use_reentrant=False)
        with torch.no_grad():
            visibility_for_vis, _, _ = mesh_visibility_v3(
                vertices, faces, sun_dir_unit, mesh_cfg=mesh_cfg, normal_source="mesh"
            )
            numerical_factor_for_vis = 1 - _lambda * torch.pow(1 + epsilon - visibility_for_vis, 1 / kappa)
            power_per_face_for_vis = (
                face_area_m2
                * torch.clamp(illumination_quality, min=0.0)
                * numerical_factor_for_vis
                * solar_irradiance_W_m2
                * efficiency
            )
    else:
        visibility_for_vis, _, _ = mesh_visibility_v3(
            vertices, faces, sun_dir_unit, mesh_cfg=mesh_cfg, normal_source="mesh"
        )
        numerical_factor_for_vis = 1 - _lambda * torch.pow(1 + epsilon - visibility_for_vis, 1 / kappa)
        power_per_face_for_vis = (
            face_area_m2
            * torch.clamp(illumination_quality, min=0.0)
            * numerical_factor_for_vis
            * solar_irradiance_W_m2
            * efficiency
        )
        power_watts = (power_per_face_for_vis * soft_mask_original_order).sum()
        visibility_for_vis = visibility_for_vis.detach()
        power_per_face_for_vis = power_per_face_for_vis.detach()

    penalty_scalar = torch.zeros((), device=vertices.device, dtype=vertices.dtype)
    if shading_penalty_weight > 0:
        # Penalty reduces the objective when illuminated faces are occluded.
        # (1 - numerical_factor) is large when visibility is low.
        penalty_scalar = shading_penalty_weight * (
            (1 - numerical_factor_for_vis.detach())
            * torch.clamp(illumination_quality, min=0.0)
            * soft_mask_original_order
        ).sum()
        power_watts = power_watts - penalty_scalar

    if epoch_name is not None:
        vis_manager.vis_solar_quality(int(epoch_name), vertices, faces, sorted_quality[original_order])
        vis_manager.vis_solar_selection(int(epoch_name), vertices, faces, soft_mask_original_order)
        vis_manager.vis_solar_shading(int(epoch_name), vertices, faces, visibility_for_vis)
        vis_manager.vis_solar_penalty(
            int(epoch_name), vertices, faces,
            (power_per_face_for_vis * soft_mask_original_order.detach()),
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
        "mean_visibility": float(visibility_for_vis.mean().item()),
        "shading_penalty": float(penalty_scalar.item()) if shading_penalty_weight > 0 else 0.0,
    }

    return power_watts, diagnostics
