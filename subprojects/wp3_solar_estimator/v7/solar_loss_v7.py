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
from matplotlib import pyplot as plt  # noqa: F401

import wandb
from subprojects.wp3_solar_estimator.australia.route_v1 import RouteV1
from subprojects.wp3_solar_estimator.australia.route_v2 import RouteV2
from subprojects.wp3_solar_estimator.mesh_extractors.diso.diso_wrapper import DISOExtractor
from subprojects.wp3_solar_estimator.mesh_extractors.kaolin_flexicubes.flexicubes_wrapper import FCExtractor
from subprojects.wp3_solar_estimator.mesh_extractors.mesh_sdf.mesh_sdf_extractor import MeshSDFExtractor
from subprojects.wp3_solar_estimator.raytracing.diffable import mesh_visibility, sdf_visibility
from subprojects.wp3_solar_estimator.utils.solar_utils import averaging_solar_quality
from subprojects.wp4_aeroshell_optimizer.v1.logging_lib.basic_plots import multi_log_in_one
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.hashing import dict_hashing
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.storage_utils import reformat_wandb_name
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.detached_modes.mesh_sdf_grads import MeshSDFGrads
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.detached_modes.shading_modes import (
    ShadingPenalty,
    ShadingVisibility,
)
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.detached_modes.solar_v3_selection import SolarV3Selection
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser import Visualiser


def _efficient_cumsum(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Memory-efficient cumulative sum — O(N) memory, O(N) time.

    torch.cumsum_cuda_kernel has no deterministic implementation, so we run on CPU
    (which is deterministic) and transfer back. The device transfer is differentiable,
    so gradients flow correctly.
    """
    return torch.cumsum(x.cpu(), dim=dim).to(x.device)


def solar_mesh_loss_v7(  # noqa: PLR0912, PLR0915
    vertices: torch.Tensor,
    faces: torch.Tensor,
    solar_vectors: torch.Tensor,
    area_budget_mm2: float,
    sun_batching_cfg: dict[str, Any],
    raytracing_cfg: dict[str, Any],
    sigmoid_sharpness: float = 10.0,
    soft_mask_selection_threshold: float = 0.5,
    soft_mask_full_selection_threshold: float = 0.95,
    budget_penalty_weight: float = 0.1,
    epsilon: float = 1e-12,
    kappa: float = 1.0,
    _lambda: float = 1.0,
    enable_wandb_logging: bool = False,
    rendering: bool = False,
    counter: int = 0,
    epoch_name: int | None = None,
    log_group_name: str = "solar_loss_metrics",
    output_dir: Path | None = None,
    num_epochs: int | None = None,
    render_solar: SolarV3Selection | None = None,
    render_shading_visibility: ShadingVisibility | None = None,
    render_shading_penalty: ShadingPenalty | None = None,
    shading_render_interval: int = 1,
    selection_threshold_render: float = 0.5,
    sdf_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
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
    sun_batch_size = 1  #####CHANGE THIS WITH CONFIG PARAM
    # Loss per face, will be aggregated into total loss later.

    losses = torch.zeros(F, device=vertices.device, dtype=vertices.dtype)
    soft_weights = torch.zeros(F, device=vertices.device, dtype=vertices.dtype)
    visual_aggregate_visibility = torch.zeros(F, device=vertices.device, dtype=vertices.dtype)
    visual_aggregate_penalty = torch.zeros(F, device=vertices.device, dtype=vertices.dtype)
    visual_quality_loss = torch.zeros(F, device=vertices.device, dtype=vertices.dtype)
    # Compute face areas and normals (mesh normals only for quality)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    edge_a, edge_b = v1 - v0, v2 - v0
    face_cross = torch.cross(edge_a, edge_b, dim=-1)
    face_cross_norm = torch.linalg.norm(face_cross, dim=-1)

    face_area_mm2 = 0.5 * face_cross_norm  # (F,)
    face_normals_unit = face_cross / (face_cross_norm).unsqueeze(-1)  # (F, 3)
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

    # Can use the computation before the loop starts
    visual_quality_loss += (soft_weights * aggregate_illumination_quality).detach().clone()

    for s in range(0, S, sun_batch_size):
        sun_batch = sun_directions_unit[s : s + sun_batch_size]  # (B, 3)
        if sun_batch_size == 1:
            sun_batch = sun_batch.squeeze(0)  # (3,)
        else:
            raise NotImplementedError(
                f"Batching for multiple (tried {sun_batch_size}) sun directions is not yet implemented in solar_mesh_loss_v7. Please set sun_batch_size to 1."
            )

        ### Raytracing to get visibility
        if raytracing_cfg["method"] == "mesh":
            visibility, _, _ = mesh_visibility(
                vertices, faces, sun_batch, normal_source=str(raytracing_cfg["normal_source"]), mesh_cfg=raytracing_cfg["mesh"]
            )  # (F,) in range [0, 1]
        elif raytracing_cfg["method"] == "sdf":
            assert sdf_fn is not None, "sdf_fn must be provided for sdf_visibility"
            visibility, _, _ = sdf_visibility(
                sdf_fn=sdf_fn, vertices=vertices, faces=faces, sun_dir=sun_batch, sdf_cfg=raytracing_cfg["sdf"]
            )  # (F,) in range [0, 1]
        else:
            raise ValueError(f"Unknown raytracing method: {raytracing_cfg['method']}")

        ### Given visibility for one sun_vector, compute loss for each face

        # simple dot product per face (sun_direction dot normals = quality)
        qualities = sun_batch @ face_normals_unit.t()  # (F,) quality of each face, not accounting shading
        numerical_factor = 1 - _lambda * torch.pow(1 + epsilon - visibility, 1 / kappa)
        # Loss already including penalty term in the numerical factor (take care of correct ordering of tensors!)
        losses += face_area_mm2 * soft_weights[original_order] * qualities * numerical_factor  # 3 times (F,) --> (F,)

        # visualisation storages, don't track gradients for these
        # Divide by S at the end to get average visibility and penalty across sun directions for visualization
        visual_aggregate_visibility += visibility.detach().clone()
        visual_aggregate_penalty += numerical_factor.detach().clone() - 1

    ### Finalisation of loss terms

    # Penalty for unused budget
    total_selected_area = (sorted_areas * soft_weights[original_order]).sum()
    unused_ratio = torch.clamp((area_budget_mm2 - total_selected_area) / max(area_budget_mm2, epsilon), min=0.0)
    budget_penalty = budget_penalty_weight * (unused_ratio**2)

    losses = losses / S  # Average over sun directions
    # Combined loss over sum of all faces
    loss_mm2 = losses.sum() + budget_penalty

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
        "shading_penalty": visual_aggregate_penalty.sum() / S,
        "soft_mask_selection_threshold": soft_mask_selection_threshold,
        "soft_mask_full_selection_threshold": soft_mask_full_selection_threshold,
        "transition_zone_faces": transition_zone_faces,
        "transition_zone_area_mm2": transition_zone_area_mm2,
        "budget_crossover_idx": budget_crossover_idx,
    }

    metrics_render = {
        "vertices": vertices,
        "faces": faces,
        "selection_weights": sorted_soft_mask,
        "threshold": selection_threshold_render,
        "shading_visibility": visual_aggregate_visibility / S,
        "shading_penalty_intensity": visual_aggregate_penalty / S,
    }

    loss = loss_mm2 * 1e-6  # Convert to m² for final loss

    # Rendering
    if rendering and num_epochs is not None:
        do_render = shading_render_interval > 0 and (counter % shading_render_interval == 0)
        if counter <= num_epochs and do_render:
            if render_solar is not None:
                render_solar.update(optimization_state=metrics_render)
            if render_shading_visibility is not None:
                render_shading_visibility.update(optimization_state=metrics_render)
            if render_shading_penalty is not None:
                render_shading_penalty.update(optimization_state=metrics_render)
        if counter == num_epochs and output_dir is not None:
            if render_solar is not None:
                render_solar.finalize(output_dir)
            if render_shading_visibility is not None:
                render_shading_visibility.finalize(output_dir, file_name="shading_visibility.gif")
            if render_shading_penalty is not None:
                render_shading_penalty.finalize(output_dir, file_name="shading_penalty.gif")

    # Log to wandb if enabled
    if enable_wandb_logging and epoch_name is not None:
        log_to_wandb(loss, vertices, faces, metrics_log, epoch_name, log_group_name, epsilon)

    return loss


def log_to_wandb(
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
            multi_log_in_one(log_group_name, log_dict, step=int(epoch_name))
    except Exception as e:
        print(f"Warning: Failed to log solar loss metrics to wandb at epoch {epoch_name}: {e}")


class SolarLossEstimator:
    def __init__(self, config: dict[str, Any]) -> None:  # noqa: PLR0915
        defaults = {
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
            "shading": {
                "sun_batch_size": 1,
                "sun_subsample_fraction": 1.0,
                "sun_subsample_min": 1,
                "sun_subsample_mode": "random",
                "shading_render_interval": 1,
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

        self.enable_wandb_logging = bool(cfg["wandb_logging"])
        self.soft_mask_selection_threshold = float(cfg["soft_mask_selection_threshold"])
        self.soft_mask_full_selection_threshold = float(cfg["soft_mask_full_selection_threshold"])
        self.selection_threshold_render = float(cfg["selection_threshold_render"])
        self.sun_batching_cfg = dict(cfg.get("sunbatching", {}))
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
        self.visualiser: Visualiser | None
        self.renderSolar: SolarV3Selection | None
        self.renderShadingVisibility: ShadingVisibility | None
        self.renderShadingPenalty: ShadingPenalty | None
        self.render_grads: MeshSDFGrads | None
        self.number_epochs: int | None
        self.model_hash: str | None
        self.wandb_run_name: str | None
        self.rendering_dir: Path | None
        if self.rendering:
            self.visualiser = Visualiser(
                image_size=1024,
                background_color=[0.0, 0.66, 0.0],
                anti_aliasing=True,
                camera_mode="look_at",
                viewing_angle=30.0,
                device=self.device,
                up=[0.0, 0.0, 1.0],
            )
            detached_cfg = load_config(v="v1", cfg_name="neural_renderer_v0.yaml").get("detached_modes", {})
            self.renderSolar = (
                SolarV3Selection(visualiser=self.visualiser)
                if detached_cfg.get("solar_v3_selection", {}).get("enabled", False)
                else None
            )
            self.renderShadingVisibility = (
                ShadingVisibility(visualiser=deepcopy(self.visualiser))
                if detached_cfg.get("shading_visibility", {}).get("enabled", False)
                else None
            )
            self.renderShadingPenalty = (
                ShadingPenalty(visualiser=deepcopy(self.visualiser))
                if detached_cfg.get("shading_penalty", {}).get("enabled", False)
                else None
            )
            self.render_grads = (
                MeshSDFGrads(visualiser=self.visualiser) if detached_cfg.get("MeshSDF_grads", {}).get("enabled", False) else None
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
    def _validate_raytracing_v1_cfg(cls, cfg: dict[str, Any]) -> None:
        cls._require_keys(cfg, ["method", "normal_source", "sdf", "mesh"], "RaytracingConfigV1")

        method = str(cfg["method"]).lower()
        if method not in {"none", "mesh", "sdf"}:
            raise ValueError("RaytracingConfigV1.method must be one of: none, mesh, sdf")

        normal_source = str(cfg["normal_source"]).lower()
        if normal_source not in {"mesh", "sdf"}:
            raise ValueError("RaytracingConfigV1.normal_source must be one of: mesh, sdf")

        sdf_cfg = cfg["sdf"]
        if not isinstance(sdf_cfg, dict):
            raise TypeError("RaytracingConfigV1.sdf must be a mapping")
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
            "RaytracingConfigV1.sdf",
        )

        sdf_method = str(sdf_cfg["method"]).lower()
        if sdf_method not in {"sigmoid", "absorption"}:
            raise ValueError("RaytracingConfigV1.sdf.method must be one of: sigmoid, absorption")

        mesh_cfg = cfg["mesh"]
        if not isinstance(mesh_cfg, dict):
            raise TypeError("RaytracingConfigV1.mesh must be a mapping")
        cls._require_keys(
            mesh_cfg,
            ["beta", "gamma", "margin_factor", "origin_offset_eps", "sun_distance_factor", "face_chunk_size", "miss_t"],
            "RaytracingConfigV1.mesh",
        )

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
        if raytracing_version == "v1":
            cls._validate_raytracing_v1_cfg(resolved_cfg)
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
                    raytracing_cfg=self.raytracing_cfg,
                    sigmoid_sharpness=self.sigmoid_sharpness,
                    soft_mask_selection_threshold=self.soft_mask_selection_threshold,
                    soft_mask_full_selection_threshold=self.soft_mask_full_selection_threshold,
                    budget_penalty_weight=self.budget_penalty_weight,
                    _lambda=self._lambda,
                    kappa=self.kappa,
                    epsilon=self.epsilon,
                    epoch_name=epoch_name,
                    log_group_name=self.log_group_name,
                    enable_wandb_logging=self.enable_wandb_logging,
                    rendering=self.rendering,
                    counter=self.counter,
                    output_dir=self.rendering_dir,
                    num_epochs=self.number_epochs,
                    render_solar=self.renderSolar,
                    render_shading_visibility=self.renderShadingVisibility,
                    render_shading_penalty=self.renderShadingPenalty,
                    shading_render_interval=self.shading_render_interval,
                    selection_threshold_render=self.selection_threshold_render,
                    sdf_fn=model,
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
                raytracing_cfg=self.raytracing_cfg,
                sigmoid_sharpness=self.sigmoid_sharpness,
                soft_mask_selection_threshold=self.soft_mask_selection_threshold,
                soft_mask_full_selection_threshold=self.soft_mask_full_selection_threshold,
                budget_penalty_weight=self.budget_penalty_weight,
                _lambda=self._lambda,
                epsilon=self.epsilon,
                kappa=self.kappa,
                epoch_name=epoch_name,
                log_group_name=self.log_group_name,
                enable_wandb_logging=self.enable_wandb_logging,
                rendering=self.rendering,
                counter=self.counter,
                output_dir=self.rendering_dir,
                num_epochs=self.number_epochs,
                render_solar=self.renderSolar,
                render_shading_visibility=self.renderShadingVisibility,
                render_shading_penalty=self.renderShadingPenalty,
                shading_render_interval=self.shading_render_interval,
                selection_threshold_render=self.selection_threshold_render,
                sdf_fn=model,
            )

        return loss if self.method != "mesh_sdf" else surrogate_loss
