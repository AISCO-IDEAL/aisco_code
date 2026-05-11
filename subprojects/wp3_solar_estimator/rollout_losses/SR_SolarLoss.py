"""
SR_SolarLoss: single fixed sun direction, no raytracing.

Computes differentiable solar power (watts) via soft face selection:
  power = Σ area_m2 * clamp(cos(sun_angle), 0) * irradiance * efficiency * soft_mask
"""

from __future__ import annotations

from typing import Any

import torch

from subprojects.wp3_solar_estimator.utils.rollout_utils.mesh_area_normals import compute_face_areas_and_normals
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import NullVisManager
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import CentralVisManagerV2


def _efficient_cumsum(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    # torch.cumsum has no deterministic CUDA kernel — run on CPU (deterministic) and transfer back.
    # Gradient flows correctly through the device transfer. O(N) memory vs the O(N²) triangular matmul fallback.
    return torch.cumsum(x.cpu(), dim=dim).to(x.device)


def solar_loss_sr(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    sun_dir: torch.Tensor,
    area_budget_mm2: float,
    solar_irradiance_W_m2: float = 1000.0,
    efficiency: float = 0.25,
    sigmoid_sharpness: float = 10.0,
    soft_mask_selection_threshold: float = 0.5,
    soft_mask_full_selection_threshold: float = 0.95,
    epsilon: float = 1e-12,
    vis_manager: CentralVisManagerV2 | NullVisManager = NullVisManager(),  # noqa: B008
    epoch_name: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Single-ray solar power objective (no raytracing).

    Returns raw power in watts — caller is responsible for any further scaling.
    No wandb calls, no rendering, no epoch counters.

    Args:
        vertices: (N, 3) mesh vertices in mm
        faces: (M, 3) triangle face indices
        sun_dir: (3,) sun direction vector (normalised internally)
        area_budget_mm2: maximum selectable area in mm²
        solar_irradiance_W_m2: solar irradiance at surface in W/m²
        efficiency: system efficiency factor (0–1)
        sigmoid_sharpness: steepness of the soft budget boundary sigmoid
        soft_mask_selection_threshold: mask threshold for counting "selected" faces in diagnostics
        soft_mask_full_selection_threshold: mask threshold for counting "fully selected" faces in diagnostics
        epsilon: numerical stability floor
        vis_manager: CentralVisManagerV2 or NullVisManager; pass NullVisManager() to skip PLY export
        epoch_name: epoch index forwarded to vis_manager.save_ply; PLY skipped when None

    Returns:
        (power_watts, diagnostics) where diagnostics is a dict of detached scalar metrics:
            total_power_watts, selected_area_m2, total_area_m2, total_faces,
            selected_faces, fully_selected_faces, transition_zone_faces, budget_crossover_idx
    """
    if faces.numel() == 0 or vertices.numel() == 0:
        print("Warning: empty mesh in solar_loss_sr, returning zero power")
        zero = torch.zeros((), device=vertices.device, dtype=vertices.dtype)
        return zero, {"total_power_watts": 0.0, "selected_area_m2": 0.0, "total_area_m2": 0.0,
                      "total_faces": 0, "selected_faces": 0, "fully_selected_faces": 0,
                      "transition_zone_faces": 0, "budget_crossover_idx": 0}

    _, face_area_m2, face_normals_unit = compute_face_areas_and_normals(vertices, faces, epsilon)

    sun_dir_unit = sun_dir / torch.linalg.norm(sun_dir).clamp_min(epsilon)
    illumination_quality = (face_normals_unit * sun_dir_unit).sum(-1)

    quality_order_desc = torch.argsort(illumination_quality, descending=True)
    sorted_areas_m2 = face_area_m2[quality_order_desc]
    sorted_quality = illumination_quality[quality_order_desc]

    cumulative_area_m2 = _efficient_cumsum(sorted_areas_m2, dim=0)
    area_budget_m2 = area_budget_mm2 * 1e-6

    distance_to_budget = (area_budget_m2 - cumulative_area_m2) / max(area_budget_m2, epsilon)
    soft_mask = torch.sigmoid(sigmoid_sharpness * distance_to_budget)

    power_per_face = sorted_areas_m2 * torch.clamp(sorted_quality, min=0.0) * solar_irradiance_W_m2 * efficiency
    power_watts = (power_per_face * soft_mask).sum()

    # Restore soft_mask to original face order for vis
    original_order = torch.argsort(quality_order_desc)
    soft_mask_original_order = soft_mask[original_order]

    if epoch_name is not None:
        vis_manager.save_ply(
            "solar/selection",
            int(epoch_name),
            vertices.detach().cpu().numpy(),
            faces.detach().cpu().numpy(),
            scalar_values=soft_mask_original_order.detach().cpu().numpy(),
        )

    faces_near_boundary = (soft_mask > 0.1) & (soft_mask < 0.9)
    budget_crossover_idx = (
        torch.argmax((cumulative_area_m2 >= area_budget_m2).float()).item()
        if (cumulative_area_m2 >= area_budget_m2).any()
        else len(cumulative_area_m2)
    )

    diagnostics: dict[str, Any] = {
        "total_power_watts": power_watts.detach().item(),
        "selected_area_m2": (sorted_areas_m2 * soft_mask).sum().detach().item(),
        "total_area_m2": face_area_m2.sum().detach().item(),
        "total_faces": int(faces.shape[0]),
        "selected_faces": int((soft_mask > soft_mask_selection_threshold).sum().item()),
        "fully_selected_faces": int((soft_mask > soft_mask_full_selection_threshold).sum().item()),
        "transition_zone_faces": int(faces_near_boundary.sum().item()),
        "budget_crossover_idx": int(budget_crossover_idx),
    }

    return power_watts, diagnostics
