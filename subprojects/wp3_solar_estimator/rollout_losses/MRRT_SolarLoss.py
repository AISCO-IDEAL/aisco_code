"""Multi-Ray + RayTracing solar loss. K sequential raytracing passes, averaged power."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.checkpoint import checkpoint

from subprojects.wp3_solar_estimator.raytracing.diffable.version3.mesh_ray_visibility_v3 import mesh_visibility_v3
from subprojects.wp3_solar_estimator.utils.rollout_utils.mesh_area_normals import compute_face_areas_and_normals
from subprojects.wp3_solar_estimator.utils.solar_utils import averaging_solar_quality

if TYPE_CHECKING:
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import NullVisManager
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import CentralVisManagerV2


def _efficient_cumsum(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    # torch.cumsum has no deterministic CUDA kernel; run on CPU and transfer back.
    return torch.cumsum(x.cpu(), dim=dim).to(x.device)


def _vis_fn(
    vtx: torch.Tensor,
    faces: torch.Tensor,
    sun_dir: torch.Tensor,
    mesh_cfg: dict[str, Any],
    normal_source: str = "mesh",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return mesh_visibility_v3(vtx, faces, sun_dir, mesh_cfg=mesh_cfg, normal_source=normal_source)


def solar_loss_mrrt(
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
    use_activation_checkpointing: bool = True,
    smooth_gradient_iters: int = 0,
    mesh_visibility_fn: Callable | None = None,
    vis_manager: CentralVisManagerV2 | NullVisManager | None = None,
    epoch_name: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute solar power loss with differentiable raytracing, averaged over K sun directions.

    Runs K sequential visibility passes (one per solar vector) and accumulates power.
    The soft-selection budget mask is built from the mean quality across all K directions
    (same as MR) so the selected face set is consistent with the no-RT variant.

    The checkpoint factory pattern (_make_vis_fn) is required because Python closures
    capture variables by reference: without it, every checkpoint function would recompute
    visibility for solar_vectors[K-1] in backward instead of its own direction.

    Args:
        vertices: (N, 3) mesh vertices in mm
        faces: (F, 3) face vertex indices
        solar_vectors: (K, 3) solar direction vectors (normalised internally)
        area_budget_mm2: maximum face area to include in the soft selection (mm²)
        raytracing_cfg: config dict with key "mesh" containing mesh_visibility_v3 params
        effective_irradiance_W_m2: W/m² scaling for average power output
        sigmoid_sharpness: steepness of sigmoid soft-budget boundary
        soft_mask_selection_threshold: diagnostic threshold for "partially selected" faces
        soft_mask_full_selection_threshold: diagnostic threshold for "fully selected" faces
        shading_penalty_weight: if > 0, adds penalty for shaded high-quality faces (averaged over K)
        kappa: exponent in numerical shading factor
        _lambda: scale of shading suppression in numerical factor
        epsilon: numerical stability clamp
        use_activation_checkpointing: recompute visibility in backward per direction to reduce peak VRAM
        smooth_gradient_iters: unused placeholder (reserved)
        mesh_visibility_fn: injectable visibility function for testing (defaults to mesh_visibility_v3)
        vis_manager: optional CVM for visual outputs (uses mean visibility across K)
        epoch_name: epoch index passed through to vis_manager

    Returns:
        (power_watts, diagnostics) where power_watts is a scalar differentiable tensor
        and diagnostics is a plain dict of floats/ints for logging.
    """
    if faces.numel() == 0 or vertices.numel() == 0:
        empty = torch.zeros((), device=vertices.device, dtype=vertices.dtype)
        return empty, {"power_watts": 0.0, "num_faces": 0, "num_solar_vectors": int(solar_vectors.shape[0])}

    solar_vectors_unit = solar_vectors / torch.linalg.norm(solar_vectors, dim=-1, keepdim=True).clamp(min=epsilon)
    K = solar_vectors_unit.shape[0]
    mesh_cfg = raytracing_cfg.get("mesh", {})
    vis_callable = mesh_visibility_fn if mesh_visibility_fn is not None else _vis_fn

    _, face_area_m2, face_normals_unit = compute_face_areas_and_normals(vertices, faces, epsilon)

    # Soft-selection mask from mean quality across all K directions (consistent with MR)
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

    F = faces.shape[0]
    power_acc = torch.zeros((), device=vertices.device, dtype=vertices.dtype)
    visibility_acc = torch.zeros(F, device=vertices.device, dtype=vertices.dtype)
    penalty_acc = torch.zeros((), device=vertices.device, dtype=vertices.dtype)

    for k in range(K):
        sun_k = solar_vectors_unit[k]
        quality_k = (face_normals_unit * sun_k).sum(-1)  # (F,)

        # Factory captures sun_k by value at call time; without this, every backward
        # pass would use the final iteration's sun_k due to Python's late-binding.
        def _make_vis_fn(s: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
            def _get_vis(vtx: torch.Tensor) -> torch.Tensor:
                vis, _, _ = vis_callable(vtx, faces, s, mesh_cfg=mesh_cfg, normal_source="mesh")
                return vis
            return _get_vis

        if use_activation_checkpointing and vertices.requires_grad:
            vis_k = checkpoint(_make_vis_fn(sun_k), vertices, use_reentrant=False)
            with torch.no_grad():
                vis_k_diag, _, _ = vis_callable(vertices, faces, sun_k, mesh_cfg=mesh_cfg, normal_source="mesh")
                vis_k_diag = vis_k_diag.detach()
        else:
            vis_k_diag, _, _ = vis_callable(vertices, faces, sun_k, mesh_cfg=mesh_cfg, normal_source="mesh")
            vis_k = vis_k_diag
            vis_k_diag = vis_k_diag.detach()

        nf_k = 1 - _lambda * torch.pow(1 + epsilon - vis_k, 1 / kappa)
        p_k = face_area_m2 * torch.clamp(quality_k, min=0.0) * nf_k * effective_irradiance_W_m2
        power_acc = power_acc + (p_k * soft_mask).sum()
        visibility_acc = visibility_acc + vis_k_diag

        if shading_penalty_weight > 0:
            nf_k_diag = 1 - _lambda * torch.pow(1 + epsilon - vis_k_diag, 1 / kappa)
            penalty_acc = penalty_acc + (
                shading_penalty_weight * ((1 - nf_k_diag) * torch.clamp(quality_k.detach(), min=0.0) * soft_mask.detach()).sum()
            )

    power_watts = power_acc / K
    if shading_penalty_weight > 0:
        power_watts = power_watts - penalty_acc / K

    mean_visibility = visibility_acc / K  # (F,) per-face average across K directions

    with torch.no_grad():
        soft_mask_det = soft_mask.detach()
        diagnostics: dict[str, Any] = {
            "power_watts": float(power_watts.item()),
            "selected_area_m2": float((face_area_m2 * soft_mask_det).sum().item()),
            "area_budget_m2": area_budget_m2,
            "num_faces": int(faces.shape[0]),
            "num_solar_vectors": K,
            "num_partially_selected_faces": int((soft_mask_det > soft_mask_selection_threshold).sum().item()),
            "num_fully_selected_faces": int((soft_mask_det > soft_mask_full_selection_threshold).sum().item()),
            "mean_avg_quality": float(avg_quality.mean().item()),
            "mean_visibility": float(mean_visibility.mean().item()),
        }

    if vis_manager is not None:
        vis_manager.log_mesh_scalars(
            vertices=vertices.detach(),
            faces=faces,
            scalars=mean_visibility,
            name="mrrt_mean_visibility",
            epoch=epoch_name,
        )
        vis_manager.log_mesh_scalars(
            vertices=vertices.detach(),
            faces=faces,
            scalars=soft_mask_det,
            name="mrrt_soft_mask",
            epoch=epoch_name,
        )

    return power_watts, diagnostics
