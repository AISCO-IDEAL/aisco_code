"""
Solar loss (v7): soft-selection loss with differentiable shading.

Extends v6 by improving the differentiable shading implementation
--> Better choice of how to integrate visibility, optimisation.

Supports RouteV1 or RouteV2 via config.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import wandb
from matplotlib import pyplot as plt  # noqa: F401
from torch.utils.checkpoint import checkpoint

from subprojects.wp3_solar_estimator.australia.route_v1 import RouteV1
from subprojects.wp3_solar_estimator.australia.route_v2 import RouteV2
from subprojects.wp3_solar_estimator.mesh_extractors.diso.diso_wrapper import DISOExtractor
from subprojects.wp3_solar_estimator.mesh_extractors.kaolin_flexicubes.flexicubes_wrapper import FCExtractor
from subprojects.wp3_solar_estimator.mesh_extractors.mesh_sdf.mesh_sdf_extractor import MeshSDFExtractor
from subprojects.wp3_solar_estimator.raytracing.diffable.version1.mesh_ray_visibility_v1 import mesh_visibility_v1
from subprojects.wp3_solar_estimator.raytracing.diffable.version1.sdf_ray_visibility_v1 import sdf_visibility_v1
from subprojects.wp3_solar_estimator.raytracing.diffable.version2.mesh_ray_visibility_v2 import mesh_visibility_v2
from subprojects.wp3_solar_estimator.raytracing.diffable.version2.sdf_ray_visibility_v2 import sdf_visibility_v2
from subprojects.wp3_solar_estimator.raytracing.diffable.version3.mesh_ray_visibility_v3 import mesh_visibility_v3
from subprojects.wp3_solar_estimator.raytracing.diffable.version3.sdf_ray_visibility_v3 import sdf_visibility_v3
from subprojects.wp3_solar_estimator.utils.solar_utils import averaging_solar_quality
from subprojects.wp4_aeroshell_optimizer.v1.logging_lib.basic_plots import multi_log_in_one
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.hashing import dict_hashing
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.storage_utils import reformat_wandb_name
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.detached_modes.mesh_sdf_grads import MeshSDFGrads
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visual_wrapper import VisualisationManager
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser import Visualiser


def _efficient_cumsum(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Memory-efficient cumulative sum — O(N) memory, O(N) time.

    torch.cumsum_cuda_kernel has no deterministic implementation, so we run on CPU
    (which is deterministic) and transfer back. The device transfer is differentiable,
    so gradients flow correctly.
    """
    return torch.cumsum(x.cpu(), dim=dim).to(x.device)


def _tensor_quantiles(x: torch.Tensor, q_values: list[float]) -> dict[str, float]:
    """Return quantiles as plain floats for stable wandb scalar logging."""
    if x.numel() == 0:
        return {f"p{round(q * 100):02d}": float("nan") for q in q_values}
    q = torch.tensor(q_values, device=x.device, dtype=x.dtype)
    values = torch.quantile(x, q)
    return {f"p{round(float(q_i) * 100):02d}": float(v.item()) for q_i, v in zip(q.tolist(), values, strict=False)}


def _build_ray_diagnostics_metrics(
    *,
    visibility: torch.Tensor,
    sigmoid_arg: torch.Tensor,
    delta_norm: torch.Tensor,
    q_values: list[float],
) -> dict[str, Any]:
    """Build scalar + tensor diagnostics for saturation analysis logging."""
    abs_arg = torch.abs(sigmoid_arg)
    q_sigmoid_arg = _tensor_quantiles(sigmoid_arg, q_values)
    q_delta_norm = _tensor_quantiles(delta_norm, q_values)

    metrics: dict[str, Any] = {
        "ray_diag_enabled": True,
        "ray_diag_sat_abs_gt5_ratio": float((abs_arg > 5.0).float().mean().item()),
        "ray_diag_sat_abs_gt8_ratio": float((abs_arg > 8.0).float().mean().item()),
        "ray_diag_grad_region_abs_le2_ratio": float((abs_arg <= 2.0).float().mean().item()),
        "ray_diag_vis_transition_01_09_ratio": float(((visibility > 0.1) & (visibility < 0.9)).float().mean().item()),
        "ray_diag_vis_transition_001_099_ratio": float(((visibility > 0.01) & (visibility < 0.99)).float().mean().item()),
        "ray_diag_sigmoid_arg": sigmoid_arg,
        "ray_diag_delta_norm": delta_norm,
    }
    for name, value in q_sigmoid_arg.items():
        metrics[f"ray_diag_sigmoid_arg_{name}"] = value
    for name, value in q_delta_norm.items():
        metrics[f"ray_diag_delta_norm_{name}"] = value
    return metrics


def _empty_quantiles(q_values: list[float]) -> dict[str, float]:
    """Return NaN quantiles when diagnostics sampling is disabled."""
    return {f"p{round(q * 100):02d}": float("nan") for q in q_values}


def solar_mesh_loss_v7(  # noqa: PLR0912, PLR0915
    vertices: torch.Tensor,
    faces: torch.Tensor,
    solar_vectors: torch.Tensor,
    area_budget_mm2: float,
    sun_batching_cfg: dict[str, Any],
    raytracing_version: str,
    raytracing_cfg: dict[str, Any],
    sigmoid_sharpness: float = 10.0,
    soft_mask_selection_threshold: float = 0.5,
    soft_mask_full_selection_threshold: float = 0.95,
    budget_penalty_weight: float = 0.1,
    epsilon: float = 1e-12,
    kappa: float = 1.0,
    _lambda: float = 1.0,
    Softplus: torch.nn.Module | None = None,
    enable_wandb_logging: bool = False,
    rendering: bool = False,
    counter: int = 0,
    epoch_name: int | None = None,
    log_group_name: str = "solar_loss_metrics",
    num_epochs: int | None = None,
    rendering_manager: VisualisationManager | None = None,
    shading_render_interval: int = 1,
    selection_threshold_render: float = 0.5,
    sdf_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    output_dir: Path | None = None,
) -> torch.Tensor:
    """
    Compute differentiable solar mesh loss with soft selection and shading.

    Args:
        vertices: (N, 3) mesh vertices in mm
        faces: (M, 3) face indices
        solar_vectors: (K, 3) solar vectors along the route
        area_budget_mm2: maximum selectable area in mm²
        sun_batching_cfg: dict with shading orchestration parameters
        raytracing_cfg: resolved raytracing config selected via raytracing_version
        sigmoid_sharpness: steepness of selection blur (higher = sharper boundary)
        soft_mask_selection_threshold: threshold for soft selection mask (0.0 to 1.0)
        soft_mask_full_selection_threshold: threshold for full selection mask (0.0 to 1.0)
        budget_penalty_weight: weight for unused budget penalty
        shading_penalty_weight: weight for shaded high-quality faces
        use_visibility_multiplier: multiply quality by visibility (if True)
        use_shading_penalty: add an explicit penalty term for shaded high-quality faces
        epsilon: numerical stability
        enable_wandb_logging: whether to log to wandb
        rendering: whether to render intermediate results
        counter: counter for rendering
        epoch_name: name of the current epoch for logging
        log_group_name: name of the log group in wandb
        output_dir: directory to save output files
    Returns:
        scalar loss in m²
    """
    ### Basic checks and solar batching
    if faces.numel() == 0 or vertices.numel() == 0:
        print("Warning: Empty vertices or faces in solar_mesh_loss_v7, returning zero loss")
        return torch.zeros((), device=vertices.device, dtype=vertices.dtype)

    # Optional sun-vector subsampling (reduces compute)
    sun_batch_size = int(sun_batching_cfg.get("sun_batch_size", 1))
    subsample_fraction = float(sun_batching_cfg.get("sun_subsample_fraction", 1.0))
    subsample_min = int(sun_batching_cfg.get("sun_subsample_min", 1))
    subsample_mode = str(sun_batching_cfg.get("sun_subsample_mode", "random")).lower()
    use_activation_checkpointing = bool(sun_batching_cfg.get("use_activation_checkpointing", True))
    memory_safe_mode = bool(sun_batching_cfg.get("memory_safe_mode", False))
    memory_safe_max_sun_vectors = int(sun_batching_cfg.get("memory_safe_max_sun_vectors", 0))
    log_visibility_histograms = bool(sun_batching_cfg.get("log_visibility_histograms", not memory_safe_mode))
    default_hist_interval = 1 if log_visibility_histograms else 0
    visibility_histogram_interval = int(sun_batching_cfg.get("visibility_histogram_interval", default_hist_interval))
    if sun_batch_size <= 0:
        raise ValueError("sun_batch_size must be > 0")
    if visibility_histogram_interval < 0:
        raise ValueError("visibility_histogram_interval must be >= 0")

    # Memory-safe hard cap before optional fractional subsampling.
    if memory_safe_max_sun_vectors > 0 and solar_vectors.shape[0] > memory_safe_max_sun_vectors:
        target_k = max(subsample_min, memory_safe_max_sun_vectors)
        target_k = min(target_k, solar_vectors.shape[0])
        if subsample_mode == "random":
            idx = torch.randperm(solar_vectors.shape[0], device=solar_vectors.device)[:target_k]
        elif subsample_mode == "first":
            idx = torch.arange(target_k, device=solar_vectors.device)
        else:
            raise ValueError(f"Unknown sun_subsample_mode: {subsample_mode}")
        solar_vectors = solar_vectors[idx]

    if 0.0 < subsample_fraction < 1.0 and solar_vectors.shape[0] > 0:
        target_k = max(subsample_min, int(solar_vectors.shape[0] * subsample_fraction))
        target_k = min(target_k, solar_vectors.shape[0])
        if subsample_mode == "random":
            idx = torch.randperm(solar_vectors.shape[0], device=solar_vectors.device)[:target_k]
        elif subsample_mode == "first":
            idx = torch.arange(target_k, device=solar_vectors.device)
        else:
            raise ValueError(f"Unknown sun_subsample_mode: {subsample_mode}")
        solar_vectors = solar_vectors[idx]

    if vertices.device != faces.device:
        raise RuntimeError(
            f"Vertices and faces must be on the same device, but got {vertices.device} and {faces.device}, something really went wrong!"
        )
    ### Setup of variables and mesh precomputations

    # initialise loss and bookkeeping variables as zero for overview
    F = faces.shape[0]
    S = solar_vectors.shape[0]
    # Loss per face, will be aggregated into total loss later.

    loss_sum_mm2 = torch.zeros((), device=vertices.device, dtype=vertices.dtype)
    visual_losses = torch.zeros(F, device=vertices.device, dtype=vertices.dtype)
    visual_aggregate_visibility = torch.zeros(F, device=vertices.device, dtype=vertices.dtype)
    visual_aggregate_penalty = torch.zeros(F, device=vertices.device, dtype=vertices.dtype)
    visual_aggregate_penalty_factor = torch.zeros(F, device=vertices.device, dtype=vertices.dtype)
    visual_quality_loss = torch.zeros(F, device=vertices.device, dtype=vertices.dtype)

    mesh_cfg = raytracing_cfg.get("mesh", {}) if raytracing_cfg.get("method") == "mesh" else {}
    ray_diag_enabled = bool(mesh_cfg.get("diagnostics_enabled", False)) and raytracing_version in ("v2", "v3")
    ray_diag_log_histograms = bool(mesh_cfg.get("diagnostics_log_histograms", True))
    ray_diag_quantiles = [float(q) for q in mesh_cfg.get("diagnostics_quantiles", [0.01, 0.05, 0.5, 0.95, 0.99])]
    ray_diag_sample_limit = int(mesh_cfg.get("diagnostics_sample_limit", 0))
    ray_diag_total_count = 0
    ray_diag_sat_abs_gt5_count = 0.0
    ray_diag_sat_abs_gt8_count = 0.0
    ray_diag_grad_region_abs_le2_count = 0.0
    ray_diag_sigmoid_arg_samples: list[torch.Tensor] = []
    ray_diag_delta_norm_samples: list[torch.Tensor] = []
    ray_diag_sample_count = 0
    # Compute face areas and normals (mesh normals only for quality)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    edge_a, edge_b = v1 - v0, v2 - v0
    face_cross = torch.cross(edge_a, edge_b, dim=-1)
    face_cross_norm = torch.linalg.norm(face_cross, dim=-1)

    face_area_mm2 = 0.5 * face_cross_norm  # (F,)
    face_normals_unit = face_cross / (face_cross_norm).unsqueeze(-1).clamp(min=epsilon)  # (F, 3)
    sun_directions_unit = solar_vectors / (torch.linalg.norm(solar_vectors, dim=-1, keepdim=True))  # (S, 3)

    # First ignore shading and sort by quality to get soft selection mask (Uses F x sun_batch_size Memory)
    aggregate_illumination_quality = averaging_solar_quality(
        face_normals_unit, sun_directions_unit, batch_size=sun_batch_size, debug_print=True
    )  # (F,)
    sorted_quality, sorted_indices = torch.sort(aggregate_illumination_quality, descending=True)
    original_order = torch.argsort(sorted_indices)
    sorted_areas = face_area_mm2[sorted_indices]
    # Deterministic-safe cumulative area
    cumulative_area = _efficient_cumsum(sorted_areas, dim=0)

    # Create soft selection weights using sigmoid with custom sharpness
    distance_to_budget = (area_budget_mm2 - cumulative_area) / max(area_budget_mm2, epsilon)
    soft_weights = torch.sigmoid(sigmoid_sharpness * distance_to_budget)
    selection_weights = soft_weights[original_order]
    selection_weights_detached = selection_weights.detach()

    # Can use the computation before the loop starts
    visual_quality_loss += (soft_weights * aggregate_illumination_quality).detach().clone()

    def _compute_visibility_and_terms(  # noqa: PLR0912
        current_vertices: torch.Tensor,
        sun_dir: torch.Tensor,
        collect_diag: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        """Compute per-sun visibility and loss terms for the current mesh state."""
        ray_diag_local: dict[str, torch.Tensor] | None = None
        if raytracing_cfg["method"] == "mesh":
            if raytracing_version == "v1":
                visibility_local, _, _ = mesh_visibility_v1(
                    current_vertices,
                    faces,
                    sun_dir,
                    normal_source=str(raytracing_cfg["normal_source"]),
                    mesh_cfg=raytracing_cfg["mesh"],
                )
            elif raytracing_version == "v2":
                if collect_diag:
                    results = mesh_visibility_v2(
                        current_vertices,
                        faces,
                        sun_dir,
                        normal_source=str(raytracing_cfg["normal_source"]),
                        mesh_cfg=raytracing_cfg["mesh"],
                        return_diagnostics=True,
                    )  # (F,) in range [0, 1]
                    visibility_local, _, _, ray_diag_local = results
                else:
                    results = mesh_visibility_v2(
                        current_vertices,
                        faces,
                        sun_dir,
                        normal_source=str(raytracing_cfg["normal_source"]),
                        mesh_cfg=raytracing_cfg["mesh"],
                        return_diagnostics=False,
                    )  # (F,) in range [0, 1]
                    visibility_local, _, _, _ = results
            elif raytracing_version == "v3":
                visibility_local, _, _ = mesh_visibility_v3(
                    current_vertices,
                    faces,
                    sun_dir,
                    normal_source=str(raytracing_cfg["normal_source"]),
                    mesh_cfg=raytracing_cfg["mesh"],
                )
            else:
                raise ValueError(f"Unknown raytracing version: {raytracing_version}")
        elif raytracing_cfg["method"] == "sdf":
            assert sdf_fn is not None, "sdf_fn must be provided for sdf_visibility"
            if raytracing_version == "v1":
                visibility_local, _, _ = sdf_visibility_v1(
                    sdf_fn=sdf_fn,
                    vertices=current_vertices,
                    faces=faces,
                    sun_dir=sun_dir,
                    normal_source=str(raytracing_cfg["normal_source"]),
                    sdf_cfg=raytracing_cfg["sdf"],
                )
            elif raytracing_version == "v2":
                visibility_local, _, _ = sdf_visibility_v2(
                    sdf_fn=sdf_fn,
                    vertices=current_vertices,
                    faces=faces,
                    sun_dir=sun_dir,
                    save_dir=output_dir,
                    normal_source=str(raytracing_cfg["normal_source"]),
                    sdf_cfg=raytracing_cfg["sdf"],
                )
            elif raytracing_version == "v3":
                visibility_local, _, _ = sdf_visibility_v3(
                    sdf_fn=sdf_fn,
                    vertices=current_vertices,
                    faces=faces,
                    sun_dir=sun_dir,
                    sdf_cfg=raytracing_cfg["sdf"],
                )
            else:
                raise ValueError(f"Unknown raytracing version: {raytracing_version}")
        else:
            raise ValueError(f"Unknown raytracing method: {raytracing_cfg['method']}")

        qualities_local = sun_dir @ face_normals_unit.t()
        if Softplus is not None:
            qualities_local = Softplus(qualities_local)
        numerical_factor_local = 1 - _lambda * torch.pow(1 + epsilon - visibility_local, 1 / kappa)
        return visibility_local, qualities_local, numerical_factor_local, ray_diag_local

    checkpoint_enabled = use_activation_checkpointing and bool(vertices.requires_grad)

    for sun_step_idx, s in enumerate(range(0, S, sun_batch_size)):
        sun_batch = sun_directions_unit[s : s + sun_batch_size]  # (B, 3)
        if sun_batch_size == 1:
            sun_batch = sun_batch.squeeze(0)  # (3,)
        else:
            raise NotImplementedError(
                f"Batching for multiple (tried {sun_batch_size}) sun directions is not yet implemented in solar_mesh_loss_v7. Please set sun_batch_size to 1."
            )

        if checkpoint_enabled:
            # Checkpoint keeps only inputs and recomputes internals in backward,
            # which avoids retaining all per-sun activations simultaneously.
            def _sun_loss_checkpointed(vtx: torch.Tensor, sun_dir_cp: torch.Tensor = sun_batch) -> torch.Tensor:
                _, qualities_cp, numerical_factor_cp, _ = _compute_visibility_and_terms(
                    current_vertices=vtx,
                    sun_dir=sun_dir_cp,
                    collect_diag=False,
                )
                return (face_area_mm2 * selection_weights * qualities_cp * numerical_factor_cp).sum()

            sun_loss_scalar = checkpoint(_sun_loss_checkpointed, vertices, use_reentrant=False)
            loss_sum_mm2 = loss_sum_mm2 + sun_loss_scalar

            # Detached pass for metrics/rendering keeps behavior while avoiding graph growth.
            with torch.no_grad():
                visibility, qualities, numerical_factor, ray_diag = _compute_visibility_and_terms(
                    current_vertices=vertices,
                    sun_dir=sun_batch,
                    collect_diag=ray_diag_enabled,
                )
                visual_losses += face_area_mm2 * selection_weights_detached * qualities * numerical_factor
        else:
            visibility, qualities, numerical_factor, ray_diag = _compute_visibility_and_terms(
                current_vertices=vertices,
                sun_dir=sun_batch,
                collect_diag=ray_diag_enabled,
            )
            per_face_loss = face_area_mm2 * selection_weights * qualities * numerical_factor
            loss_sum_mm2 = loss_sum_mm2 + per_face_loss.sum()
            visual_losses += per_face_loss.detach()

        if ray_diag_enabled and ray_diag is not None:
            sigmoid_arg_detached = ray_diag["sigmoid_arg"].detach()
            delta_norm_detached = ray_diag["delta_norm"].detach()
            abs_sigmoid_arg = torch.abs(sigmoid_arg_detached)
            ray_diag_total_count += int(sigmoid_arg_detached.numel())
            ray_diag_sat_abs_gt5_count += float((abs_sigmoid_arg > 5.0).sum().item())
            ray_diag_sat_abs_gt8_count += float((abs_sigmoid_arg > 8.0).sum().item())
            ray_diag_grad_region_abs_le2_count += float((abs_sigmoid_arg <= 2.0).sum().item())

            if ray_diag_sample_limit > 0 and ray_diag_sample_count < ray_diag_sample_limit:
                remaining = ray_diag_sample_limit - ray_diag_sample_count
                flat_sigmoid_arg = sigmoid_arg_detached.flatten()
                flat_delta_norm = delta_norm_detached.flatten()
                take = min(remaining, int(flat_sigmoid_arg.numel()))
                if take > 0:
                    if take == flat_sigmoid_arg.numel():
                        sigmoid_sample = flat_sigmoid_arg
                        delta_sample = flat_delta_norm
                    else:
                        sample_idx = torch.randint(
                            low=0,
                            high=flat_sigmoid_arg.numel(),
                            size=(take,),
                            device=flat_sigmoid_arg.device,
                        )
                        sigmoid_sample = flat_sigmoid_arg[sample_idx]
                        delta_sample = flat_delta_norm[sample_idx]
                    ray_diag_sigmoid_arg_samples.append(sigmoid_sample.cpu())
                    ray_diag_delta_norm_samples.append(delta_sample.cpu())
                    ray_diag_sample_count += take

        # visualisation storages, don't track gradients for these
        # Divide by S at the end to get average visibility and penalty across sun directions for visualization

        # make wandb histogram of values in visibility tensor across faces for this sun direction
        if (
            enable_wandb_logging
            and log_visibility_histograms
            and visibility_histogram_interval > 0
            and (sun_step_idx % visibility_histogram_interval == 0)
        ):
            visibility_detached = visibility.detach()
            payload = {
                f"{log_group_name}/visibility_histogram": wandb.Histogram(visibility_detached.cpu().numpy()),
                f"{log_group_name}/visibility_histogram_mean": float(visibility_detached.mean().item()),
                f"{log_group_name}/visibility_histogram_std": float(visibility_detached.std(unbiased=False).item()),
            }
            if epoch_name is not None:
                payload["epoch"] = epoch_name
                wandb.log(payload, step=int(epoch_name))
            else:
                wandb.log(payload)
        visibility_detached = visibility.detach()
        numerical_factor_detached = numerical_factor.detach()
        qualities_detached = qualities.detach()
        visual_aggregate_visibility += visibility_detached
        # The factor that gets subtracted from 1 to get the total solar "power"
        visual_aggregate_penalty_factor += (-numerical_factor_detached) + 1
        # The actual penalty contribution (given as per area penalty)
        visual_aggregate_penalty += (-numerical_factor_detached + 1) * qualities_detached * selection_weights_detached

    ### Finalisation of loss terms

    # Penalty for unused budget
    total_selected_area = (sorted_areas * soft_weights[original_order]).sum()
    unused_ratio = torch.clamp((area_budget_mm2 - total_selected_area) / max(area_budget_mm2, epsilon), min=0.0)
    budget_penalty = budget_penalty_weight * (unused_ratio**2)

    losses = visual_losses / S  # Average per-face loss over sun directions (for logging/rendering)
    # Combined loss over sum of all faces
    loss_mm2 = (loss_sum_mm2 / S) + budget_penalty

    # Sort softmask back to original face order for logging/visualization
    sorted_soft_mask = soft_weights[original_order]

    # Analyze sigmoid transition zone
    transition_zone_faces = ((soft_weights > 0.1) & (soft_weights < 0.9)).sum().item()
    faces_near_boundary = (soft_weights > 0.1) & (soft_weights < 0.9)
    transition_zone_area_mm2 = (sorted_areas[faces_near_boundary]).sum().item() if faces_near_boundary.any() else 0.0

    # Find approximate budget crossover point
    budget_crossover_idx = (
        torch.argmax((cumulative_area >= area_budget_mm2).float()).item()
        if (cumulative_area >= area_budget_mm2).any()
        else len(cumulative_area)
    )

    metrics_log = {
        "face_area_mm2": face_area_mm2,
        "illumination_quality": aggregate_illumination_quality,
        "visibility_mean": visual_aggregate_visibility / S,
        "sorted_areas": sorted_areas,
        "sorted_quality": sorted_quality,
        "soft_weights": soft_weights,
        "sorted_soft_weights": sorted_soft_mask,
        "quality_loss": visual_quality_loss,
        "budget_penalty": budget_penalty,
        "numerical_factor": visual_aggregate_penalty_factor.sum() / S,
        "shading_penalty": visual_aggregate_penalty.sum() / S,
        "soft_mask_selection_threshold": soft_mask_selection_threshold,
        "soft_mask_full_selection_threshold": soft_mask_full_selection_threshold,
        "transition_zone_faces": transition_zone_faces,
        "transition_zone_area_mm2": transition_zone_area_mm2,
        "budget_crossover_idx": budget_crossover_idx,
    }

    if ray_diag_enabled and ray_diag_total_count > 0:
        visibility_mean = visual_aggregate_visibility / S
        denom = max(ray_diag_total_count, 1)
        metrics_log.update(
            {
                "ray_diag_enabled": True,
                "ray_diag_sat_abs_gt5_ratio": ray_diag_sat_abs_gt5_count / denom,
                "ray_diag_sat_abs_gt8_ratio": ray_diag_sat_abs_gt8_count / denom,
                "ray_diag_grad_region_abs_le2_ratio": ray_diag_grad_region_abs_le2_count / denom,
                "ray_diag_vis_transition_01_09_ratio": float(
                    ((visibility_mean > 0.1) & (visibility_mean < 0.9)).float().mean().item()
                ),
                "ray_diag_vis_transition_001_099_ratio": float(
                    ((visibility_mean > 0.01) & (visibility_mean < 0.99)).float().mean().item()
                ),
            }
        )

        if ray_diag_sigmoid_arg_samples and ray_diag_delta_norm_samples:
            ray_diag_sigmoid_arg = torch.cat(ray_diag_sigmoid_arg_samples, dim=0)
            ray_diag_delta_norm = torch.cat(ray_diag_delta_norm_samples, dim=0)
            q_sigmoid_arg = _tensor_quantiles(ray_diag_sigmoid_arg, ray_diag_quantiles)
            q_delta_norm = _tensor_quantiles(ray_diag_delta_norm, ray_diag_quantiles)
            for name, value in q_sigmoid_arg.items():
                metrics_log[f"ray_diag_sigmoid_arg_{name}"] = value
            for name, value in q_delta_norm.items():
                metrics_log[f"ray_diag_delta_norm_{name}"] = value

            metrics_log["ray_diag_log_histograms"] = ray_diag_log_histograms
            if ray_diag_log_histograms:
                metrics_log["ray_diag_sigmoid_arg"] = ray_diag_sigmoid_arg
                metrics_log["ray_diag_delta_norm"] = ray_diag_delta_norm
        else:
            q_sigmoid_arg = _empty_quantiles(ray_diag_quantiles)
            q_delta_norm = _empty_quantiles(ray_diag_quantiles)
            for name, value in q_sigmoid_arg.items():
                metrics_log[f"ray_diag_sigmoid_arg_{name}"] = value
            for name, value in q_delta_norm.items():
                metrics_log[f"ray_diag_delta_norm_{name}"] = value
            metrics_log["ray_diag_log_histograms"] = False

    metrics_render = {
        "vertices": vertices,
        "faces": faces,
        "selection_weights": sorted_soft_mask,
        "threshold": selection_threshold_render,
        "shading_visibility": visual_aggregate_visibility / S,
        "shading_penalty_intensity": visual_aggregate_penalty / S,
        "shading_factor": visual_aggregate_penalty_factor / S,
        "qualities": aggregate_illumination_quality,
        "losses": losses,
    }

    loss = loss_mm2 * 1e-6  # Convert to m² for final loss

    # Rendering
    if rendering and num_epochs is not None:
        assert rendering_manager is not None, "rendering_manager must be provided for rendering"
        do_render = shading_render_interval > 0 and (counter % shading_render_interval == 0)
        if counter <= num_epochs and do_render:
            rendering_manager.update(optimization_state=metrics_render)
        if counter == num_epochs and rendering_manager is not None:
            rendering_manager.finalize()

    # Log to wandb if enabled
    if enable_wandb_logging and epoch_name is not None:
        log_to_wandb(loss, vertices, faces, metrics_log, epoch_name, log_group_name, epsilon)

    return loss


def log_to_wandb(  # noqa: PLR0915 #too many statements since we log a ton of stuff
    loss: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    metrics: dict[str, Any],
    epoch_name: int,
    log_group_name: str = "solar_loss_metrics",
    epsilon: float = 1e-12,
) -> None:
    """
    Log solar loss metrics to wandb using pre-computed metrics.
    """

    def _as_float_scalar(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Expected scalar tensor, got shape={tuple(value.shape)}")
            return float(value.detach().cpu().item())
        return float(value)

    def _summary_stats(value: Any) -> tuple[float, float, float]:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                scalar = float(value.detach().cpu().item())
                return scalar, scalar, scalar
            v = value.detach()
            return float(v.mean().item()), float(v.max().item()), float(v.min().item())
        scalar = float(value)
        return scalar, scalar, scalar

    try:
        if faces.numel() > 0 and vertices.numel() > 0 and metrics:
            required_keys = [
                "face_area_mm2",
                "illumination_quality",
                "visibility_mean",
                "sorted_areas",
                "soft_weights",
                "quality_loss",
                "budget_penalty",
                "shading_penalty",
                "soft_mask_selection_threshold",
                "soft_mask_full_selection_threshold",
                "transition_zone_faces",
                "transition_zone_area_mm2",
                "budget_crossover_idx",
            ]
            missing = [k for k in required_keys if k not in metrics]
            if missing:
                print(f"Warning: Skipping solar wandb logging due to missing metrics keys: {missing}")
                return

            face_area_mm2 = metrics["face_area_mm2"]
            illumination_quality = metrics["illumination_quality"]
            visibility_mean = metrics["visibility_mean"]
            sorted_areas = metrics["sorted_areas"]
            soft_mask = metrics["soft_weights"]
            quality_loss = metrics["quality_loss"]
            budget_penalty = metrics["budget_penalty"]
            shading_penalty = metrics["shading_penalty"]
            soft_mask_selection_threshold = float(metrics["soft_mask_selection_threshold"])
            soft_mask_full_selection_threshold = float(metrics["soft_mask_full_selection_threshold"])
            transition_zone_faces = int(metrics["transition_zone_faces"])
            transition_zone_area_mm2 = float(metrics["transition_zone_area_mm2"])
            budget_crossover_idx = int(metrics["budget_crossover_idx"])

            total_area_mm2 = float(face_area_mm2.sum().item())
            total_num_faces = len(face_area_mm2)
            selected_area_mm2 = float((sorted_areas * soft_mask).sum().item())

            quality_mean, quality_max, quality_min = _summary_stats(illumination_quality)
            visibility_mean_scalar, _, _ = _summary_stats(visibility_mean)
            soft_mask_mean, soft_mask_max, soft_mask_min = _summary_stats(soft_mask)

            selected_faces = int((soft_mask > soft_mask_selection_threshold).sum().item())
            fully_selected_faces = int((soft_mask > soft_mask_full_selection_threshold).sum().item())

            if isinstance(quality_loss, torch.Tensor) and quality_loss.numel() > 1:
                quality_loss_scalar = float(quality_loss.detach().sum().cpu().item())
            else:
                quality_loss_scalar = _as_float_scalar(quality_loss)

            shading_penalty_scalar = _as_float_scalar(shading_penalty)
            budget_penalty_scalar = _as_float_scalar(budget_penalty)

            log_dict = {
                "solar_loss": float(loss.detach().cpu().item()),
                "total_area_m2": total_area_mm2 * 1e-6,
                "selected_area_m2": selected_area_mm2 * 1e-6,
                "selection_ratio": selected_area_mm2 / max(total_area_mm2, epsilon),
                "solar_quality_loss": quality_loss_scalar * 1e-6,
                "solar_budget_penalty": budget_penalty_scalar,
                "solar_shading_penalty": shading_penalty_scalar * 1e-6,
                "solar_quality_mean": quality_mean,
                "solar_quality_max": quality_max,
                "solar_quality_min": quality_min,
                "solar_visibility_mean": visibility_mean_scalar,
                "solar_soft_mask_mean": soft_mask_mean,
                "solar_soft_mask_max": soft_mask_max,
                "solar_soft_mask_min": soft_mask_min,
                "total_faces": total_num_faces,
                "selected_faces": selected_faces,
                "fully_selected_faces": fully_selected_faces,
                "transition_zone_faces": transition_zone_faces,
                "transition_zone_area_m2": transition_zone_area_mm2 * 1e-6,
                "transition_zone_ratio": transition_zone_faces / max(total_num_faces, 1),
                "budget_crossover_idx": budget_crossover_idx,
            }

            if bool(metrics.get("ray_diag_enabled", False)):
                diag_scalar_keys = [
                    "ray_diag_sat_abs_gt5_ratio",
                    "ray_diag_sat_abs_gt8_ratio",
                    "ray_diag_grad_region_abs_le2_ratio",
                    "ray_diag_vis_transition_01_09_ratio",
                    "ray_diag_vis_transition_001_099_ratio",
                ]
                diag_quantile_keys = [
                    key for key in metrics if key.startswith(("ray_diag_sigmoid_arg_p", "ray_diag_delta_norm_p"))
                ]
                for key in diag_scalar_keys + diag_quantile_keys:
                    if key in metrics:
                        log_dict[key] = float(metrics[key])

            multi_log_in_one(log_group_name, log_dict, step=int(epoch_name))

            if bool(metrics.get("ray_diag_enabled", False)) and bool(metrics.get("ray_diag_log_histograms", True)):
                hist_payload = {"epoch": epoch_name}
                if "ray_diag_sigmoid_arg" in metrics:
                    hist_payload[f"{log_group_name}/ray_diag_sigmoid_arg_histogram"] = wandb.Histogram(
                        metrics["ray_diag_sigmoid_arg"].detach().cpu().numpy()
                    )
                if "ray_diag_delta_norm" in metrics:
                    hist_payload[f"{log_group_name}/ray_diag_delta_norm_histogram"] = wandb.Histogram(
                        metrics["ray_diag_delta_norm"].detach().cpu().numpy()
                    )
                if len(hist_payload) > 1:
                    wandb.log(hist_payload, step=int(epoch_name))
    except Exception as e:
        print(f"Warning: Failed to log solar loss metrics to wandb at epoch {epoch_name}: {e}")


class SolarLossEstimator:
    def __init__(self, config: dict[str, Any]) -> None:  # noqa: PLR0915
        defaults: dict[str, Any] = {
            "route_version": "v2",
            "raytracing_version": "v1",
            "method": "mesh_sdf",
            "bbox": [[-5000, 5000], [-5000, 5000], [-5000, 5000]],
            "resolution": 64,
            "area_budget_mm2": 1e6,
            "device": "cuda",
            "batch_size": 4096,
            "eps": 1e-12,
            "sigmoid_sharpness": 10.0,
            "budget_penalty_weight": 0.1,
            "shading_penalty_weight": 10.0,
            "use_visibility_multiplier": True,
            "use_shading_penalty": True,
            "sun_vector_source": "route",
            "fixed_sun_dir": None,
            "wandb_logging": False,
            "soft_mask_selection_threshold": 0.5,
            "soft_mask_full_selection_threshold": 0.95,
            "rendering": False,
            "selection_threshold_render": 0.5,
            "log_group_name": "solar_loss_metrics",
            "sun_batching": {
                "sun_batch_size": 1,
                "sun_subsample_fraction": 1.0,
                "sun_subsample_min": 1,
                "sun_subsample_mode": "random",
                "use_activation_checkpointing": False,
                "shading_render_interval": 1,
                "memory_safe_mode": False,
                "memory_safe_max_sun_vectors": 0,
                "log_visibility_histograms": True,
                "visibility_histogram_interval": 1,
            },
        }

        general_config = load_config(v="v1", cfg_name="general_config.yaml")
        cfg: dict[str, Any] = defaults if config is None else config

        self.config = cfg

        self.route_version = str(cfg.get("route_version", "v2")).lower()
        if "raytracing_version" not in cfg:
            raise ValueError("SolarLossConfigV7 must define 'raytracing_version'")
        self.raytracing_version = str(cfg["raytracing_version"]).lower()
        self.method = cfg["method"]
        self.bbox = cfg["bbox"]
        self.resolution = int(cfg["resolution"])
        self.area_budget_mm2 = float(cfg["area_budget_mm2"])
        self.device = cfg["device"]
        self.batch_size = int(cfg["batch_size"])

        self.log_group_name = str(cfg["log_group_name"])
        self.sigmoid_sharpness = float(cfg["sigmoid_sharpness"])
        self.budget_penalty_weight = float(cfg["budget_penalty_weight"])
        self._lambda = float(cfg["lambda"])
        self.kappa = float(cfg["kappa"])
        self.epsilon = float(cfg["epsilon"])
        self.Softplus: torch.nn.Module | None
        if cfg["apply_softplus"]:
            self.Softplus = torch.nn.Softplus(beta=cfg.get("softplus_beta", 5.2))
        else:
            self.Softplus = None
        self.enable_wandb_logging = bool(cfg["wandb_logging"])
        self.soft_mask_selection_threshold = float(cfg["soft_mask_selection_threshold"])
        self.soft_mask_full_selection_threshold = float(cfg["soft_mask_full_selection_threshold"])
        self.selection_threshold_render = float(cfg["selection_threshold_render"])
        sun_batching_cfg_raw = cfg.get("sun_batching", cfg.get("sunbatching", cfg.get("shading", {})))
        sun_batching_cfg_dict = sun_batching_cfg_raw if isinstance(sun_batching_cfg_raw, dict) else {}
        self.sun_batching_cfg = {**defaults["sun_batching"], **sun_batching_cfg_dict}
        self.raytracing_cfg = self._load_versioned_raytracing_config(self.raytracing_version)
        self.shading_render_interval = int(self.sun_batching_cfg.get("shading_render_interval", 1))
        self.sun_vector_source = str(cfg.get("sun_vector_source", "route")).lower()
        self.fixed_sun_dir = cfg.get("fixed_sun_dir")
        self.counter = 0

        self.validation_sim = bool(cfg.get("validation_sim", False))
        self.validation_interval = int(cfg.get("validation_interval", 0))
        self.simulator = None
        if self.validation_sim:
            from subprojects.wp3_solar_estimator.validation.simulationv1 import SimulationV1  # noqa: PLC0415

            australia_cfg = cfg.get("australia_cfg", "solar_configs/australia.yaml")
            self.simulator = SimulationV1(cfg_path=australia_cfg, area_budget_mm2=self.area_budget_mm2)

        self.rendering = bool(cfg["rendering"])
        self.number_epochs: int | None
        self.model_hash: str | None
        self.wandb_run_name: str | None
        self.rendering_dir: Path | None
        self.visualiser: Visualiser | None
        self.render_grads: MeshSDFGrads | None
        if self.rendering:
            self.visualiser = Visualiser(
                image_size=4500,
                background_color=[1.0, 1.0, 1.0],
                anti_aliasing=True,
                camera_mode="look_at",
                viewing_angle=30.0,
                device=self.device,
                up=[0.0, 0.0, 1.0],
            )

            self.visualisation_manager = VisualisationManager(
                visualiser=self.visualiser, key_for_modes="solar_info_modes", verbose=True
            )
            detached_cfg = load_config(v="v1", cfg_name=general_config["active_configs"]["neural_renderer"])["detached_modes"]
            assert self.visualiser is not None  # for type checker
            self.render_grads = (
                MeshSDFGrads(visualiser=deepcopy(self.visualiser))
                if detached_cfg.get("MeshSDF_grads", {}).get("enabled", False)
                else None
            )

            train_cfg = load_config(v="v1", cfg_name="training_v0_config.yaml")["training_params"]
            self.number_epochs = int(train_cfg["num_epochs"])
            self.model_hash = None
            self.wandb_run_name = wandb.run.name if wandb.run else None
            self.rendering_dir = None
            self.storage_paths = general_config["storage_paths"]
        else:
            self.visualiser = None
            self.renderSolar = None
            self.renderShadingVisibility = None
            self.renderShadingPenalty = None
            self.render_grads = None
            self.number_epochs = None
            self.model_hash = None
            self.wandb_run_name = None
            self.rendering_dir = None

        self.route = None
        if self.sun_vector_source == "fixed" and self.fixed_sun_dir is not None:
            fixed_dir = torch.tensor(self.fixed_sun_dir, dtype=torch.float32).view(-1)
            if fixed_dir.numel() != 3:
                raise ValueError("fixed_sun_dir must contain exactly 3 values")
            fixed_dir = fixed_dir / fixed_dir.norm()
            self.solar_vectors = fixed_dir.view(1, 3).to(self.device)
        else:
            route_cls = RouteV2 if self.route_version == "v2" else RouteV1
            self.route = route_cls(cfg_path=cfg.get("australia_cfg", "solar_configs/australia.yaml"))
            self.solar_vectors = self.route.get_route_solar_vecs()
            self.solar_vectors = self.solar_vectors.view(-1, 3).to(self.device)

        self.extractor: FCExtractor | DISOExtractor | MeshSDFExtractor
        self._init_extractor()

    @staticmethod
    def _require_keys(cfg: dict[str, Any], required_keys: list[str], cfg_name: str) -> None:
        missing = [key for key in required_keys if key not in cfg]
        if missing:
            raise ValueError(f"{cfg_name} is missing required keys: {missing}")

    @classmethod
    def _validate_raytracing_cfg(cls, cfg: dict[str, Any], raytracing_version: str) -> None:
        cls._require_keys(cfg, ["method", "normal_source", "sdf", "mesh"], f"RaytracingConfig{raytracing_version.upper()}")

        method = str(cfg["method"]).lower()
        if method not in {"none", "mesh", "sdf"}:
            raise ValueError(f"RaytracingConfig{raytracing_version.upper()}.method must be one of: none, mesh, sdf")

        normal_source = str(cfg["normal_source"]).lower()
        if normal_source not in {"mesh", "sdf"}:
            raise ValueError(f"RaytracingConfig{raytracing_version.upper()}.normal_source must be one of: mesh, sdf")

        sdf_cfg = cfg["sdf"]
        if not isinstance(sdf_cfg, dict):
            raise TypeError(f"RaytracingConfig{raytracing_version.upper()}.sdf must be a mapping")
        cls._require_keys(
            sdf_cfg,
            [
                "method",
                "alpha",
                "beta",
                "kappa",
                "num_steps",
                "step_size_factor",
                "origin_offset_factor",
                "max_march_factor",
                "point_chunk_size",
            ],
            f"RaytracingConfig{raytracing_version.upper()}.sdf",
        )

        sdf_method = str(sdf_cfg["method"]).lower()
        if sdf_method not in {"sigmoid", "absorption"}:
            raise ValueError(f"RaytracingConfig{raytracing_version.upper()}.sdf.method must be one of: sigmoid, absorption")

        mesh_cfg = cfg["mesh"]
        if not isinstance(mesh_cfg, dict):
            raise TypeError(f"RaytracingConfig{raytracing_version.upper()}.mesh must be a mapping")
        cls._require_keys(
            mesh_cfg,
            ["beta", "gamma", "margin_factor", "origin_offset_eps", "sun_distance_factor", "face_chunk_size", "miss_t"],
            f"RaytracingConfig{raytracing_version.upper()}.mesh",
        )
        if raytracing_version in {"v2", "v3"}:
            cls._require_keys(mesh_cfg, ["blocker_chunk_size"], f"RaytracingConfig{raytracing_version.upper()}.mesh")

    @classmethod
    def _load_versioned_raytracing_config(cls, raytracing_version: str) -> dict[str, Any]:
        rt_cfg = load_config(v="v1", cfg_name="solar_configs/raytracing_config.yaml")
        key = f"RaytracingConfig{raytracing_version.upper()}"
        if key not in rt_cfg:
            available = sorted(rt_cfg.keys())
            raise ValueError(
                f"Unknown raytracing_version '{raytracing_version}'. Expected key '{key}' in raytracing_config.yaml. "
                f"Available versions: {available}"
            )

        resolved_cfg = dict(rt_cfg[key])
        if raytracing_version in {"v1", "v2", "v3"}:
            cls._validate_raytracing_cfg(resolved_cfg, raytracing_version)
        else:
            raise ValueError(f"Unsupported raytracing_version '{raytracing_version}'.")

        return resolved_cfg

    def _init_extractor(self) -> None:
        method = self.method
        if method == "flexicubes":
            self.extractor = FCExtractor(device=self.device, verbose=False, fixed_box=True)
            self.extractor.set_grid(self.bbox, self.resolution)
        elif method in {"diso_mc", "diso_dmc"}:
            extractor_type = "DiffMC" if method == "diso_mc" else "DiffDMC"
            self.extractor = DISOExtractor(
                extractor_type=extractor_type,
                dtype="float32",
                device=self.device,
                verbose=False,
                deform=False,
                fixed_box=True,
            )
            self.extractor.set_grid(self.bbox, self.resolution)
        elif method in {"mesh_sdf", "mc"}:
            bbox = self.bbox
            self.extractor = MeshSDFExtractor(
                resolution=self.resolution,
                bbox=[
                    bbox[0][0],
                    bbox[0][1],
                    bbox[1][0],
                    bbox[1][1],
                    bbox[2][0],
                    bbox[2][1],
                ],
                isovalue=0.0,
                device=self.device,
            )
            self.extractor.set_grid(self.bbox, self.resolution)
        else:
            raise ValueError(f"Unknown method: {method}")

    def extract_mesh(self, model: Callable[[torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.method == "flexicubes":
            assert isinstance(self.extractor, FCExtractor)
            return self.extractor.extract_mesh(model, batch_size=self.batch_size)
        if self.method in {"diso_mc", "diso_dmc"}:
            assert isinstance(self.extractor, DISOExtractor)
            return self.extractor.extract_mesh(model, isovalue=0.0, batch_size=self.batch_size)
        if self.method in {"mesh_sdf", "mc"}:
            assert isinstance(self.extractor, MeshSDFExtractor)
            return self.extractor.extract_mesh(model)

        raise ValueError(f"Unknown method: {self.method}")

    def solar_loss(self, model: Any, epoch_name: int | None = None) -> torch.Tensor:
        """
        Compute the solar loss for the given model.
        """
        if self.rendering and self.model_hash is None:
            self.model_hash = dict_hashing(model.get_model_info_dict())
            run_name = reformat_wandb_name(self.wandb_run_name) if self.wandb_run_name else "no_wandb_run"
            self.rendering_dir = Path(self.storage_paths["storage_dir"]) / self.model_hash / run_name / "solar_loss_rendering"
            self.visualisation_manager.output_dir = self.rendering_dir
            print(f"Rendering enabled: saving to {self.rendering_dir}")
            os.makedirs(self.rendering_dir, exist_ok=True)
            print(f"Model hash: {self.model_hash}")
            print(f"Wandb run name: {self.wandb_run_name}")

        self.counter += 1

        if self.method == "mesh_sdf":

            def wrapper_solar_loss(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
                if self.validation_sim and self.simulator is not None:
                    run_sim = False
                    if (self.validation_interval > 0 and self.counter % self.validation_interval == 0) or (
                        self.number_epochs is not None and self.counter == self.number_epochs
                    ):
                        run_sim = True

                    if run_sim:
                        self.simulator.run(vertices, faces, output_dir=self.rendering_dir, epoch_name=epoch_name)

                        if (
                            self.number_epochs is not None
                            and self.counter == self.number_epochs
                            and self.rendering_dir is not None
                        ):
                            from subprojects.wp4_aeroshell_optimizer.v1.logging_lib.image_uploads import (  # noqa: PLC0415
                                create_and_upload_gif_from_folder,
                            )

                            create_and_upload_gif_from_folder(
                                self.rendering_dir / "sim_renders", pattern="simulation_results_*.png", step=self.counter, fps=10
                            )

                loss = solar_mesh_loss_v7(
                    vertices,
                    faces,
                    self.solar_vectors,
                    self.area_budget_mm2,
                    sun_batching_cfg=self.sun_batching_cfg,
                    raytracing_version=self.raytracing_version,
                    raytracing_cfg=self.raytracing_cfg,
                    sigmoid_sharpness=self.sigmoid_sharpness,
                    soft_mask_selection_threshold=self.soft_mask_selection_threshold,
                    soft_mask_full_selection_threshold=self.soft_mask_full_selection_threshold,
                    budget_penalty_weight=self.budget_penalty_weight,
                    _lambda=self._lambda,
                    kappa=self.kappa,
                    epsilon=self.epsilon,
                    Softplus=self.Softplus,
                    epoch_name=epoch_name,
                    log_group_name=self.log_group_name,
                    enable_wandb_logging=self.enable_wandb_logging,
                    rendering=self.rendering,
                    counter=self.counter,
                    num_epochs=self.number_epochs,
                    rendering_manager=self.visualisation_manager,
                    shading_render_interval=self.shading_render_interval,
                    selection_threshold_render=self.selection_threshold_render,
                    sdf_fn=model,
                    output_dir=self.rendering_dir,
                )
                return loss

            assert isinstance(self.extractor, MeshSDFExtractor)
            vertices, faces, loss, surrogate_loss, dL_dvertices = self.extractor.extract_and_compute_loss(
                sdf_model=model,
                loss_fn=wrapper_solar_loss,
                enable_backward=True,
            )
            if self.rendering and self.render_grads is not None:
                self.render_grads.update(
                    optimization_state={
                        "vertices": vertices,
                        "faces": faces,
                        "vertice_grads": dL_dvertices,
                    }
                )
                if (
                    self.enable_wandb_logging
                    and self.number_epochs is not None
                    and self.counter == self.number_epochs
                    and self.rendering_dir is not None
                ):
                    self.render_grads.finalize(output_dir=self.rendering_dir, file_name="mesh_sdf_grads.gif")

        else:
            vertices, faces = self.extract_mesh(model)

            if self.validation_sim and self.simulator is not None:
                run_sim = False
                if (self.validation_interval > 0 and self.counter % self.validation_interval == 0) or (
                    self.number_epochs is not None and self.counter == self.number_epochs
                ):
                    run_sim = True

                if run_sim:
                    self.simulator.run(vertices, faces, output_dir=self.rendering_dir, epoch_name=epoch_name)

                    if self.number_epochs is not None and self.counter == self.number_epochs and self.rendering_dir is not None:
                        from subprojects.wp4_aeroshell_optimizer.v1.logging_lib.image_uploads import (  # noqa: PLC0415
                            create_and_upload_gif_from_folder,
                        )

                        create_and_upload_gif_from_folder(self.rendering_dir / "sim_renders", step=self.counter, fps=10)

            loss = solar_mesh_loss_v7(
                vertices,
                faces,
                self.solar_vectors,
                self.area_budget_mm2,
                sun_batching_cfg=self.sun_batching_cfg,
                raytracing_version=self.raytracing_version,
                raytracing_cfg=self.raytracing_cfg,
                sigmoid_sharpness=self.sigmoid_sharpness,
                soft_mask_selection_threshold=self.soft_mask_selection_threshold,
                soft_mask_full_selection_threshold=self.soft_mask_full_selection_threshold,
                budget_penalty_weight=self.budget_penalty_weight,
                _lambda=self._lambda,
                epsilon=self.epsilon,
                kappa=self.kappa,
                Softplus=self.Softplus,
                epoch_name=epoch_name,
                log_group_name=self.log_group_name,
                enable_wandb_logging=self.enable_wandb_logging,
                rendering=self.rendering,
                counter=self.counter,
                num_epochs=self.number_epochs,
                rendering_manager=self.visualisation_manager,
                shading_render_interval=self.shading_render_interval,
                selection_threshold_render=self.selection_threshold_render,
                sdf_fn=model,
                output_dir=self.rendering_dir,
            )

        return loss if self.method != "mesh_sdf" else surrogate_loss
