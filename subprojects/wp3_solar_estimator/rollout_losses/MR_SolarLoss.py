"""Multi-Ray solar loss (no raytracing). Quality averaged over K sun directions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from subprojects.wp3_solar_estimator.utils.rollout_utils.mesh_area_normals import compute_face_areas_and_normals
from subprojects.wp3_solar_estimator.utils.solar_utils import averaging_solar_quality

if TYPE_CHECKING:
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import NullVisManager
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import CentralVisManagerV2


def _efficient_cumsum(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    # torch.cumsum has no deterministic CUDA kernel; run on CPU and transfer back.
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
    smooth_gradient_iters: int = 0,
    vis_manager: CentralVisManagerV2 | NullVisManager | None = None,
    epoch_name: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute solar power loss averaged over K sun directions, without raytracing.

    Face quality is the mean dot product with all K solar vectors via averaging_solar_quality().
    The soft-selection budget and power formula are identical to SR.

    Args:
        vertices: (N, 3) mesh vertices in mm
        faces: (F, 3) face vertex indices
        solar_vectors: (K, 3) solar direction vectors (normalised internally)
        area_budget_mm2: maximum face area to include in the soft selection (mm²)
        effective_irradiance_W_m2: W/m² scaling for average power output
        sigmoid_sharpness: steepness of sigmoid soft-budget boundary
        soft_mask_selection_threshold: diagnostic threshold for "partially selected" faces
        soft_mask_full_selection_threshold: diagnostic threshold for "fully selected" faces
        epsilon: numerical stability clamp
        smooth_gradient_iters: unused placeholder (reserved)
        vis_manager: optional CVM for visual outputs
        epoch_name: epoch index passed through to vis_manager

    Returns:
        (power_watts, diagnostics) where power_watts is a scalar differentiable tensor
        and diagnostics is a plain dict of floats/ints for logging.
    """
    if faces.numel() == 0 or vertices.numel() == 0:
        empty = torch.zeros((), device=vertices.device, dtype=vertices.dtype)
        return empty, {"power_watts": 0.0, "num_faces": 0, "num_solar_vectors": int(solar_vectors.shape[0])}

    solar_vectors_unit = solar_vectors / torch.linalg.norm(solar_vectors, dim=-1, keepdim=True).clamp(min=epsilon)

    _, face_area_m2, face_normals_unit = compute_face_areas_and_normals(vertices, faces, epsilon)

    # averaging_solar_quality expects normalised inputs and returns the mean dot product over K directions
    avg_quality = averaging_solar_quality(face_normals_unit, solar_vectors_unit)  # (F,)

    quality_order_desc = torch.argsort(avg_quality, descending=True)
    sorted_areas_m2 = face_area_m2[quality_order_desc]
    cumulative_area_m2 = _efficient_cumsum(sorted_areas_m2)

    area_budget_m2 = area_budget_mm2 * 1e-6
    soft_mask_sorted = torch.sigmoid(
        sigmoid_sharpness * (area_budget_m2 - cumulative_area_m2) / max(area_budget_m2, epsilon)
    )
    original_order = torch.argsort(quality_order_desc)
    soft_mask = soft_mask_sorted[original_order]

    power_watts = (face_area_m2 * torch.clamp(avg_quality, min=0.0) * soft_mask * effective_irradiance_W_m2).sum()

    with torch.no_grad():
        soft_mask_det = soft_mask.detach()
        diagnostics: dict[str, Any] = {
            "power_watts": float(power_watts.item()),
            "selected_area_m2": float((face_area_m2 * soft_mask_det).sum().item()),
            "area_budget_m2": area_budget_m2,
            "num_faces": int(faces.shape[0]),
            "num_solar_vectors": int(solar_vectors.shape[0]),
            "num_partially_selected_faces": int((soft_mask_det > soft_mask_selection_threshold).sum().item()),
            "num_fully_selected_faces": int((soft_mask_det > soft_mask_full_selection_threshold).sum().item()),
            "mean_avg_quality": float(avg_quality.mean().item()),
        }

    if vis_manager is not None:
        vis_manager.log_mesh_scalars(
            vertices=vertices.detach(),
            faces=faces,
            scalars=soft_mask_det,
            name="mr_soft_mask",
            epoch=epoch_name,
        )

    return power_watts, diagnostics
