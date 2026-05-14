"""Single-Ray solar loss (no raytracing). One fixed sun direction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from subprojects.wp3_solar_estimator.utils.rollout_utils.mesh_area_normals import compute_face_areas_and_normals

if TYPE_CHECKING:
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import NullVisManager
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import CentralVisManagerV2


def _efficient_cumsum(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    # torch.cumsum has no deterministic CUDA kernel; run on CPU and transfer back.
    return torch.cumsum(x.cpu(), dim=dim).to(x.device)


def solar_loss_sr(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    sun_dir: torch.Tensor,
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
    Compute solar power loss for a single fixed sun direction with no raytracing.

    The loss is the total illuminated power (W) over the soft-selected face budget.
    Higher = better solar geometry; WP4 wrappers maximise this.

    Args:
        vertices: (N, 3) mesh vertices in mm
        faces: (F, 3) face vertex indices
        sun_dir: (3,) sun direction vector (normalised internally)
        area_budget_mm2: maximum face area to include in the soft selection (mm²)
        effective_irradiance_W_m2: W/m² scaling for power output
        sigmoid_sharpness: steepness of sigmoid soft-budget boundary
        soft_mask_selection_threshold: diagnostic threshold for "partially selected" faces
        soft_mask_full_selection_threshold: diagnostic threshold for "fully selected" faces
        epsilon: numerical stability clamp
        smooth_gradient_iters: unused placeholder for gradient smoothing (reserved)
        vis_manager: optional CVM for visual outputs
        epoch_name: epoch index passed through to vis_manager

    Returns:
        (power_watts, diagnostics) where power_watts is a scalar differentiable tensor
        and diagnostics is a plain dict of floats/ints for logging.
    """
    if faces.numel() == 0 or vertices.numel() == 0:
        empty = torch.zeros((), device=vertices.device, dtype=vertices.dtype)
        return empty, {"power_watts": 0.0, "num_faces": 0}

    sun_dir_unit = sun_dir / torch.linalg.norm(sun_dir).clamp(min=epsilon)

    _, face_area_m2, face_normals_unit = compute_face_areas_and_normals(vertices, faces, epsilon)

    illumination_quality = (face_normals_unit * sun_dir_unit).sum(-1)  # (F,)

    quality_order_desc = torch.argsort(illumination_quality, descending=True)
    sorted_areas_m2 = face_area_m2[quality_order_desc]
    cumulative_area_m2 = _efficient_cumsum(sorted_areas_m2)

    area_budget_m2 = area_budget_mm2 * 1e-6
    soft_mask_sorted = torch.sigmoid(
        sigmoid_sharpness * (area_budget_m2 - cumulative_area_m2) / max(area_budget_m2, epsilon)
    )
    original_order = torch.argsort(quality_order_desc)
    soft_mask = soft_mask_sorted[original_order]

    power_watts = (face_area_m2 * torch.clamp(illumination_quality, min=0.0) * soft_mask * effective_irradiance_W_m2).sum()

    with torch.no_grad():
        soft_mask_det = soft_mask.detach()
        num_partially_selected = int((soft_mask_det > soft_mask_selection_threshold).sum().item())
        num_fully_selected = int((soft_mask_det > soft_mask_full_selection_threshold).sum().item())
        selected_area_m2 = float((face_area_m2 * soft_mask_det).sum().item())

        diagnostics: dict[str, Any] = {
            "power_watts": float(power_watts.item()),
            "selected_area_m2": selected_area_m2,
            "area_budget_m2": area_budget_m2,
            "num_faces": int(faces.shape[0]),
            "num_partially_selected_faces": num_partially_selected,
            "num_fully_selected_faces": num_fully_selected,
            "mean_illumination_quality": float(illumination_quality.mean().item()),
        }

    if vis_manager is not None:
        vis_manager.log_mesh_scalars(
            vertices=vertices.detach(),
            faces=faces,
            scalars=soft_mask.detach(),
            name="sr_soft_mask",
            epoch=epoch_name,
        )

    return power_watts, diagnostics
