"""
Solar loss (v6): soft-selection loss with differentiable shading.

Extends v4/v5 by incorporating differentiable visibility from ray tracing.
Supports RouteV1 or RouteV2 via config.
"""

from __future__ import annotations

import os
from collections.abc import Callable
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


def _compute_visibility_per_sun(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    sun_dirs: torch.Tensor,
    sdf_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    shading_cfg: dict[str, Any],
) -> torch.Tensor:
    """
    Compute per-face visibility for each sun direction.

    Returns:
        visibility: (F, K) tensor in [0, 1]
    """
    method = shading_cfg.get("method", "none")
    if method == "none":
        return torch.ones((faces.shape[0], sun_dirs.shape[0]), device=vertices.device, dtype=vertices.dtype)

    sun_batch_size = int(shading_cfg.get("sun_batch_size", 1))
    normal_source = shading_cfg.get("normal_source", "mesh")

    vis_chunks: list[torch.Tensor] = []
    for start in range(0, sun_dirs.shape[0], sun_batch_size):
        end = min(start + sun_batch_size, sun_dirs.shape[0])
        sun_chunk = sun_dirs[start:end]  # (B, 3)
        vis_per_chunk: list[torch.Tensor] = []

        for i in range(sun_chunk.shape[0]):
            sun_dir = sun_chunk[i]
            if method == "mesh":
                vis, _, _ = mesh_visibility(
                    vertices,
                    faces,
                    sun_dir,
                    sdf_fn=sdf_fn,
                    normal_source=normal_source,
                    beta=float(shading_cfg.get("mesh_beta", 10.0)),
                    gamma=float(shading_cfg.get("mesh_gamma", 20.0)),
                    margin_factor=float(shading_cfg.get("mesh_margin_factor", 1e-3)),
                    origin_offset_eps=float(shading_cfg.get("mesh_origin_offset_eps", 1e-3)),
                    sun_distance_factor=float(shading_cfg.get("mesh_sun_distance_factor", 10.0)),
                    face_chunk_size=int(shading_cfg.get("mesh_face_chunk_size", 512)),
                    miss_t=float(shading_cfg.get("mesh_miss_t", 1e9)),
                )
            elif method == "sdf":
                if sdf_fn is None:
                    raise ValueError("sdf_fn is required when shading method is 'sdf'")
                vis, _, _ = sdf_visibility(
                    sdf_fn,
                    vertices,
                    faces,
                    sun_dir,
                    normal_source=normal_source,
                    method=shading_cfg.get("sdf_method", "sigmoid"),
                    alpha=float(shading_cfg.get("sdf_alpha", 20.0)),
                    beta=float(shading_cfg.get("sdf_beta", 20.0)),
                    kappa=float(shading_cfg.get("sdf_kappa", 2.0)),
                    num_steps=int(shading_cfg.get("sdf_num_steps", 32)),
                    step_size_factor=float(shading_cfg.get("sdf_step_size_factor", 0.05)),
                    origin_offset_factor=float(shading_cfg.get("sdf_origin_offset_factor", 1e-3)),
                    max_march_factor=float(shading_cfg.get("sdf_max_march_factor", 2.0)),
                    point_chunk_size=int(shading_cfg.get("sdf_point_chunk_size", 8192)),
                )
            else:
                raise ValueError(f"Unknown shading method: {method}")

            vis_per_chunk.append(vis)

        vis_chunks.append(torch.stack(vis_per_chunk, dim=1))  # (F, B)

    return torch.cat(vis_chunks, dim=1)  # (F, K)


def solar_mesh_loss_v6(  # noqa: PLR0912, PLR0915
    vertices: torch.Tensor,
    faces: torch.Tensor,
    solar_vectors: torch.Tensor,
    area_budget_mm2: float,
    shading_cfg: dict[str, Any],
    sigmoid_sharpness: float = 10.0,
    soft_mask_selection_threshold: float = 0.5,
    soft_mask_full_selection_threshold: float = 0.95,
    budget_penalty_weight: float = 0.1,
    shading_penalty_weight: float = 10.0,
    use_visibility_multiplier: bool = True,
    use_shading_penalty: bool = True,
    epsilon: float = 1e-12,
    enable_wandb_logging: bool = False,
    rendering: bool = False,
    counter: int = 0,
    epoch_name: int | None = None,
    log_group_name: str = "solar_loss_metrics",
    output_dir: Path | None = None,
    num_epochs: int | None = None,
    render_solar: SolarV3Selection | None = None,
    render_shading_visibility: Any | None = None,
    render_shading_penalty: Any | None = None,
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
        shading_cfg: dict with raytracing parameters
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

    if faces.numel() == 0 or vertices.numel() == 0:
        print("Warning: Empty vertices or faces in solar_mesh_loss_v6, returning zero loss")
        return torch.zeros((), device=vertices.device, dtype=vertices.dtype)

    # Optional sun-vector subsampling (reduces memory/compute)
    subsample_fraction = float(shading_cfg.get("sun_subsample_fraction", 1.0))
    subsample_min = int(shading_cfg.get("sun_subsample_min", 1))
    subsample_mode = str(shading_cfg.get("sun_subsample_mode", "random")).lower()
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

    # Compute face areas and normals (mesh normals only for quality)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    edge_a, edge_b = v1 - v0, v2 - v0
    face_cross = torch.cross(edge_a, edge_b, dim=-1)
    face_cross_norm = torch.linalg.norm(face_cross, dim=-1)
    face_area_mm2 = 0.5 * face_cross_norm

    face_normals_unit = face_cross / (face_cross_norm.clamp_min(epsilon)).unsqueeze(-1)  # (M, 3)
    sun_directions_unit = solar_vectors / (torch.linalg.norm(solar_vectors, dim=-1, keepdim=True).clamp_min(epsilon))

    # Illumination quality per face (average over sun vectors)
    aggregate_illumination_quality = averaging_solar_quality(face_normals_unit, sun_directions_unit)

    # Visibility per face (average over sun vectors)
    visibility_per_sun = _compute_visibility_per_sun(vertices, faces, sun_directions_unit, sdf_fn, shading_cfg)
    aggregate_visibility = visibility_per_sun.mean(dim=1)

    # Combine quality and visibility
    if use_visibility_multiplier:
        combined_quality = aggregate_illumination_quality * aggregate_visibility
    else:
        combined_quality = aggregate_illumination_quality

    # Sort faces by descending combined quality
    quality_order_desc = torch.argsort(combined_quality, descending=True)
    sorted_areas = face_area_mm2[quality_order_desc]
    sorted_quality = combined_quality[quality_order_desc]
    original_order = torch.argsort(quality_order_desc)

    # Deterministic-safe cumulative area
    cumulative_area = _efficient_cumsum(sorted_areas, dim=0)

    # Create soft selection mask using sigmoid
    distance_to_budget = (area_budget_mm2 - cumulative_area) / max(area_budget_mm2, epsilon)
    soft_mask = torch.sigmoid(sigmoid_sharpness * distance_to_budget)

    # Base quality-area term
    weighted_quality_area = sorted_areas * sorted_quality * soft_mask
    quality_loss = weighted_quality_area.sum()

    # Penalty for unused budget
    total_selected_area = (sorted_areas * soft_mask).sum()
    unused_ratio = torch.clamp((area_budget_mm2 - total_selected_area) / max(area_budget_mm2, epsilon), min=0.0)
    budget_penalty = budget_penalty_weight * (unused_ratio**2)

    # Shading penalty (high quality but shaded faces)
    shading_penalty = torch.zeros((), device=vertices.device, dtype=vertices.dtype)
    shading_penalty_per_face = torch.zeros_like(face_area_mm2)
    shading_penalty_unweighted = torch.zeros_like(face_area_mm2)
    shading_penalty_intensity = torch.zeros_like(face_area_mm2)
    if use_shading_penalty:
        sorted_visibility = aggregate_visibility[quality_order_desc]
        sorted_raw_quality = aggregate_illumination_quality[quality_order_desc]
        shading_per_sorted_unweighted = sorted_areas * sorted_raw_quality * (1.0 - sorted_visibility) * soft_mask
        shading_intensity_sorted = sorted_raw_quality * (1.0 - sorted_visibility) * soft_mask
        shading_per_sorted = shading_penalty_weight * shading_per_sorted_unweighted
        shading_penalty = shading_per_sorted.sum()
        shading_penalty_per_face = shading_per_sorted[original_order]
        shading_penalty_unweighted = shading_per_sorted_unweighted[original_order]
        shading_penalty_intensity = shading_intensity_sorted[original_order]

    # Combined loss
    loss_mm2 = quality_loss - budget_penalty - shading_penalty

    # Sort softmask back to original face order for logging/visualization
    sorted_soft_mask = soft_mask[original_order]

    # Analyze sigmoid transition zone
    transition_zone_faces = ((soft_mask > 0.1) & (soft_mask < 0.9)).sum().item()
    faces_near_boundary = (soft_mask > 0.1) & (soft_mask < 0.9)
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
        "visibility_mean": aggregate_visibility,
        "sorted_areas": sorted_areas,
        "sorted_quality": sorted_quality,
        "soft_mask": soft_mask,
        "sorted_soft_mask": sorted_soft_mask,
        "quality_loss": quality_loss,
        "budget_penalty": budget_penalty,
        "shading_penalty": shading_penalty,
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
        "shading_visibility": aggregate_visibility,
        "shading_penalty": shading_penalty_per_face,
        "shading_penalty_unweighted": shading_penalty_unweighted,
        "shading_penalty_intensity": shading_penalty_intensity,
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
    metrics: dict[str, torch.Tensor],
    epoch_name: int,
    log_group_name: str = "solar_loss_metrics",
    epsilon: float = 1e-12,
) -> None:
    """
    Log solar loss metrics to wandb using pre-computed metrics.
    """
    try:
        if faces.numel() > 0 and vertices.numel() > 0 and metrics:
            face_area_mm2 = metrics["face_area_mm2"]
            illumination_quality = metrics["illumination_quality"]
            sorted_areas = metrics["sorted_areas"]
            soft_mask = metrics["soft_mask"]
            quality_loss = metrics["quality_loss"]
            budget_penalty = metrics["budget_penalty"]
            shading_penalty = metrics["shading_penalty"]
            soft_mask_selection_threshold = metrics["soft_mask_selection_threshold"]
            soft_mask_full_selection_threshold = metrics["soft_mask_full_selection_threshold"]
            transition_zone_faces = metrics["transition_zone_faces"]
            transition_zone_area_mm2 = metrics["transition_zone_area_mm2"]
            budget_crossover_idx = metrics["budget_crossover_idx"]
            visibility_mean = metrics["visibility_mean"]

            total_area_mm2 = face_area_mm2.sum().item()
            total_num_faces = len(face_area_mm2)
            selected_area_mm2 = (sorted_areas * soft_mask).sum().item()

            quality_mean = illumination_quality.mean().item()
            quality_max = illumination_quality.max().item()
            quality_min = illumination_quality.min().item()

            soft_mask_mean = soft_mask.mean().item()
            soft_mask_max = soft_mask.max().item()
            soft_mask_min = soft_mask.min().item()

            selected_faces = (soft_mask > soft_mask_selection_threshold).sum().item()
            fully_selected_faces = (soft_mask > soft_mask_full_selection_threshold).sum().item()

            log_dict = {
                "solar_loss": float(loss.detach().cpu().item()),
                "total_area_m2": total_area_mm2 * 1e-6,
                "selected_area_m2": selected_area_mm2 * 1e-6,
                "selection_ratio": selected_area_mm2 / max(total_area_mm2, epsilon),
                "solar_quality_loss": float(quality_loss.detach().cpu().item()) * 1e-6,
                "solar_budget_penalty": float(budget_penalty.detach().cpu().item()),
                "solar_shading_penalty": float(shading_penalty.detach().cpu().item()) * 1e-6,
                "solar_quality_mean": quality_mean,
                "solar_quality_max": quality_max,
                "solar_quality_min": quality_min,
                "solar_visibility_mean": visibility_mean.mean().item(),
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
        print(f"Warning: Failed to log solar loss metrics to wandb: {e}")


class SolarLossEstimator:
    def __init__(self, config: dict[str, Any]) -> None:  # noqa: PLR0915
        defaults = {
            "route_version": "v2",
            "method": "flexicubes",
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
                "method": "sdf",
                "normal_source": "mesh",
                "sun_batch_size": 1,
                "sdf_method": "sigmoid",
                "sdf_alpha": 20.0,
                "sdf_beta": 20.0,
                "sdf_kappa": 2.0,
                "sdf_num_steps": 32,
                "sdf_step_size_factor": 0.05,
                "sdf_origin_offset_factor": 1e-3,
                "sdf_max_march_factor": 2.0,
                "sdf_point_chunk_size": 8192,
                "mesh_beta": 10.0,
                "mesh_gamma": 20.0,
                "mesh_margin_factor": 1e-3,
                "mesh_origin_offset_eps": 1e-3,
                "mesh_sun_distance_factor": 10.0,
                "mesh_face_chunk_size": 512,
                "mesh_miss_t": 1e9,
            },
        }

        general_config = load_config(v="v1", cfg_name="general_config.yaml")
        cfg: dict[str, Any] = defaults if config is None else config

        self.config = cfg

        self.route_version = str(cfg.get("route_version", "v2")).lower()
        self.method = cfg["method"]
        self.bbox = cfg["bbox"]
        self.resolution = int(cfg["resolution"])
        self.area_budget_mm2 = float(cfg["area_budget_mm2"])
        self.device = cfg["device"]
        self.batch_size = int(cfg["batch_size"])
        self.eps = float(cfg["eps"])
        self.log_group_name = str(cfg["log_group_name"])
        self.sigmoid_sharpness = float(cfg["sigmoid_sharpness"])
        self.budget_penalty_weight = float(cfg["budget_penalty_weight"])
        self.shading_penalty_weight = float(cfg["shading_penalty_weight"])
        self.use_visibility_multiplier = bool(cfg.get("use_visibility_multiplier", True))
        self.use_shading_penalty = bool(cfg.get("use_shading_penalty", True))
        self.enable_wandb_logging = bool(cfg["wandb_logging"])
        self.soft_mask_selection_threshold = float(cfg["soft_mask_selection_threshold"])
        self.soft_mask_full_selection_threshold = float(cfg["soft_mask_full_selection_threshold"])
        self.selection_threshold_render = float(cfg["selection_threshold_render"])
        self.shading_cfg = dict(cfg.get("shading", {}))
        self.shading_render_interval = int(self.shading_cfg.get("shading_render_interval", 1))
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
        self.number_epochs: int | None
        self.model_hash: str | None
        self.wandb_run_name: str | None
        self.rendering_dir: Path | None
        if self.rendering and self.enable_wandb_logging:
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
                ShadingVisibility(visualiser=self.visualiser)
                if detached_cfg.get("shading_visibility", {}).get("enabled", False)
                else None
            )
            self.renderShadingPenalty = (
                ShadingPenalty(visualiser=self.visualiser)
                if detached_cfg.get("shading_penalty", {}).get("enabled", False)
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
            self.number_epochs = None
            self.model_hash = None
            self.wandb_run_name = None
            self.rendering_dir = None

        self.route = None
        if self.sun_vector_source == "fixed" and self.fixed_sun_dir is not None:
            fixed_dir = torch.tensor(self.fixed_sun_dir, dtype=torch.float32).view(-1)
            if fixed_dir.numel() != 3:
                raise ValueError("fixed_sun_dir must contain exactly 3 values")
            fixed_dir = fixed_dir / fixed_dir.norm().clamp_min(self.eps)
            self.solar_vectors = fixed_dir.view(1, 3).to(self.device)
        else:
            route_cls = RouteV2 if self.route_version == "v2" else RouteV1
            self.route = route_cls(cfg_path=cfg.get("australia_cfg", "solar_configs/australia.yaml"))
            self.solar_vectors = self.route.get_route_solar_vecs()
            self.solar_vectors = self.solar_vectors.view(-1, 3).to(self.device)

        self.extractor: FCExtractor | DISOExtractor | MeshSDFExtractor
        self._init_extractor()

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
        if self.rendering and self.enable_wandb_logging and self.model_hash is None:
            self.model_hash = dict_hashing(model.get_model_info_dict())
            if self.wandb_run_name is not None:
                self.rendering_dir = (
                    Path(self.storage_paths["storage_dir"])
                    / self.model_hash
                    / reformat_wandb_name(self.wandb_run_name)
                    / "solar_loss_rendering"
                )
                print(f"Rendering enabled: saving to {self.rendering_dir}")
                os.makedirs(self.rendering_dir, exist_ok=True)
            else:
                print("Warning: Wandb logging is enabled but no active run found, solar_loss_rendering will not be saved")
                self.rendering_dir = None
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

                            create_and_upload_gif_from_folder(self.rendering_dir / "sim_renders", step=self.counter, fps=10)

                loss = solar_mesh_loss_v6(
                    vertices,
                    faces,
                    self.solar_vectors,
                    self.area_budget_mm2,
                    shading_cfg=self.shading_cfg,
                    sigmoid_sharpness=self.sigmoid_sharpness,
                    soft_mask_selection_threshold=self.soft_mask_selection_threshold,
                    soft_mask_full_selection_threshold=self.soft_mask_full_selection_threshold,
                    budget_penalty_weight=self.budget_penalty_weight,
                    shading_penalty_weight=self.shading_penalty_weight,
                    use_visibility_multiplier=self.use_visibility_multiplier,
                    use_shading_penalty=self.use_shading_penalty,
                    epsilon=self.eps,
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
            _vertices, _faces, loss, surrogate_loss, _vertice_grads = self.extractor.extract_and_compute_loss(
                sdf_model=model,
                loss_fn=wrapper_solar_loss,
                enable_backward=True,
            )

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

            loss = solar_mesh_loss_v6(
                vertices,
                faces,
                self.solar_vectors,
                self.area_budget_mm2,
                shading_cfg=self.shading_cfg,
                sigmoid_sharpness=self.sigmoid_sharpness,
                soft_mask_selection_threshold=self.soft_mask_selection_threshold,
                soft_mask_full_selection_threshold=self.soft_mask_full_selection_threshold,
                budget_penalty_weight=self.budget_penalty_weight,
                shading_penalty_weight=self.shading_penalty_weight,
                use_visibility_multiplier=self.use_visibility_multiplier,
                use_shading_penalty=self.use_shading_penalty,
                epsilon=self.eps,
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
