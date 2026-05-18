"""
MR_SolarLoss: K solar directions averaged via dot-product quality, no raytracing.

Extends SR by replacing the single-direction quality score with a mean over K
sun vectors via averaging_solar_quality(), then applies the same soft budget
selection and power computation.
"""

from __future__ import annotations

from typing import Any

import torch

from subprojects.wp3_solar_estimator.utils.rollout_utils.mesh_area_normals import compute_face_areas_and_normals
from subprojects.wp3_solar_estimator.utils.solar_utils import averaging_solar_quality
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import NullVisManager
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import CentralVisManagerV2


def _efficient_cumsum(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    # torch.cumsum has no deterministic CUDA kernel — run on CPU and transfer back.
    return torch.cumsum(x.cpu(), dim=dim).to(x.device)


def solar_loss_mr(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    solar_vectors: torch.Tensor,
    area_budget_mm2: float,
    effective_irradiance_W_m2: float = 250.0,
    sigmoid_sharpness: float = 10.0,
    soft_mask_selection_threshold: float = 0.5,
    soft_mask_full_selection_threshold: float = 0.95,
    epsilon: float = 1e-12,
    vis_manager: CentralVisManagerV2 | NullVisManager = NullVisManager(),
    epoch_name: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Multi-ray solar power objective (no raytracing).

    Face selection order is determined by the mean illumination quality across all K sun
    directions. Power is computed per-direction and averaged over K.

    Returns raw power in watts — caller is responsible for any further scaling.

    Args:
        vertices: (N, 3) mesh vertices in mm
        faces: (M, 3) triangle face indices
        solar_vectors: (K, 3) sun direction vectors (normalised internally)
        area_budget_mm2: maximum selectable area in mm²
        effective_irradiance_W_m2: electrical power output per m² of perfectly-aligned area
        sigmoid_sharpness: steepness of the soft budget boundary sigmoid
        soft_mask_selection_threshold: mask threshold for counting "selected" faces in diagnostics
        soft_mask_full_selection_threshold: mask threshold for counting "fully selected" faces in diagnostics
        epsilon: numerical stability floor
        vis_manager: pass NullVisManager() to skip PLY export
        epoch_name: epoch index forwarded to vis_manager; PLY skipped when None

    Returns:
        (power_watts, diagnostics) where diagnostics keys:
            total_power_watts, selected_area_m2, total_area_m2, total_faces,
            selected_faces, fully_selected_faces, transition_zone_faces,
            budget_crossover_idx, num_solar_vectors
    """
    if faces.numel() == 0 or vertices.numel() == 0:
        print("Warning: empty mesh in solar_loss_mr, returning zero power")
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
            "num_solar_vectors": int(solar_vectors.shape[0]),
        }

    _, face_area_m2, face_normals_unit = compute_face_areas_and_normals(vertices, faces, epsilon)

    solar_vectors_unit = solar_vectors / torch.linalg.norm(solar_vectors, dim=-1, keepdim=True).clamp_min(epsilon)
    avg_quality = averaging_solar_quality(face_normals_unit, solar_vectors_unit)  # (F,)

    quality_order_desc = torch.argsort(avg_quality, descending=True)
    sorted_areas_m2 = face_area_m2[quality_order_desc]
    sorted_avg_quality = avg_quality[quality_order_desc]

    cumulative_area_m2 = _efficient_cumsum(sorted_areas_m2, dim=0)
    area_budget_m2 = area_budget_mm2 * 1e-6

    distance_to_budget = (area_budget_m2 - cumulative_area_m2) / max(area_budget_m2, epsilon)
    soft_mask = torch.sigmoid(sigmoid_sharpness * distance_to_budget)

    power_per_face = sorted_areas_m2 * torch.clamp(sorted_avg_quality, min=0.0) * effective_irradiance_W_m2
    power_watts = (power_per_face * soft_mask).sum()

    original_order = torch.argsort(quality_order_desc)
    soft_mask_original_order = soft_mask[original_order]

    if epoch_name is not None:
        vis_manager.vis_solar_selection(int(epoch_name), vertices, faces, soft_mask_original_order)
        vis_manager.vis_solar_quality(int(epoch_name), vertices, faces, avg_quality)

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
        "num_solar_vectors": int(solar_vectors.shape[0]),
    }

    return power_watts, diagnostics
