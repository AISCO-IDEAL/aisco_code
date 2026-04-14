"""
Solar loss (v3B_2): fully differentiable soft-selection loss with meaningful watt output.

Computes actual solar power (watts) instead of surrogate loss.
Power = Σ(selected_area x illumination_quality) x solar_irradiance x efficiency

Supported methods:
  - mesh_sdf: MeshSDF (non-diff extraction + gradient projection surrogate)
  - flexicubes: Kaolin FlexiCubes (diff)
  - diso_mc: DISO DiffMC (diff)
  - diso_dmc: DISO DiffDMC (diff)
  - mc: Marching Cubes baseline (non-diff, forward only)
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

import wandb
from subprojects.wp3_solar_estimator.mesh_extractors.diso.diso_wrapper import DISOExtractor
from subprojects.wp3_solar_estimator.mesh_extractors.kaolin_flexicubes.flexicubes_wrapper import FCExtractor
from subprojects.wp3_solar_estimator.mesh_extractors.mesh_sdf.mesh_sdf_extractor import MeshSDFExtractor
from subprojects.wp3_solar_estimator.mesh_extractors.sparse_dmc.sparse_dmc_extractor import SparseDMCExtractor
from subprojects.wp4_aeroshell_optimizer import (
    MeshSDFGrads,
    SolarV3Selection,
    Visualiser,
    dict_hashing,
    load_config,
    multi_log_in_one,
    reformat_wandb_name,
)


def _deterministic_cumsum_legacy(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    LEGACY — kept for potential revert only. Use _efficient_cumsum instead.

    Deterministic cumulative sum using matrix multiplication - avoids torch.cumsum entirely.
    This is faster than manual loops but slower than native cumsum, but works with
    deterministic algorithms enabled without any warnings.
    NOTE: Eats n**2 memory, so beware of big meshing sizes
    Args:
        x: (N,) input tensor
        dim: dimension to cumsum along (only supports dim=0 for simplicity)
    Returns:
        (N,) cumulative sum tensor where cumsum[i] = x[0] + x[1] + ... + x[i]
    """
    if dim != 0:
        raise ValueError(f"_deterministic_cumsum_legacy only supports dim=0, got dim={dim}")

    n = x.shape[0]
    if n == 0:
        return x

    # Create LOWER triangular matrix of ones - each row i has ones in positions 0 to i
    # This allows us to compute cumsum via matrix-vector multiplication
    tril_mask = torch.tril(torch.ones(n, n, device=x.device, dtype=x.dtype))  # Changed from triu to tril!

    # Matrix multiplication: tril_mask @ x gives cumulative sums
    # tril_mask[i, :] has ones from 0 to i, so (tril_mask @ x)[i] = sum(x[0:i+1])
    return tril_mask @ x


def _efficient_cumsum(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Memory-efficient cumulative sum — O(N) memory, O(N) time.

    Replaces _deterministic_cumsum_legacy which used an O(N²) triangular matmul
    that caused OOM at high mesh resolutions (e.g. 512³ → ~100K faces → 37 GiB).

    torch.cumsum_cuda_kernel has no deterministic implementation, so we run on CPU
    (which is deterministic) and transfer back. The device transfer is differentiable,
    so gradients flow correctly. For ~100K face tensors the transfer overhead is negligible.

    To revert to the O(N²) deterministic fallback: replace `_efficient_cumsum` with
    `_deterministic_cumsum_legacy` at the call site.
    """
    return torch.cumsum(x.cpu(), dim=dim).to(x.device)


def solar_mesh_loss_v3(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    sun_direction: torch.Tensor,
    area_budget_mm2: float,
    solar_irradiance_W_m2: float = 1000.0,
    efficiency: float = 0.25,
    power_output_scale: float = 1.0,
    sigmoid_sharpness: float = 10.0,
    soft_mask_selection_threshold: float = 0.5,
    soft_mask_full_selection_threshold: float = 0.95,
    epsilon: float = 1e-12,
    enable_wandb_logging: bool = False,
    rendering: bool = False,
    counter: int = 0,
    epoch_name: int | None = None,
    log_group_name: str = "solar_loss_metrics",
    output_dir: Path | None = None,
    num_epochs: int | None = None,
    render_solar: SolarV3Selection | None = None,
    selection_threshold_render: float = 0.5,
) -> torch.Tensor:
    """
    Compute differentiable solar power objective using soft selection.

    Returns positive power in watts.
    Gradients flow backward through the power computation.

    Args:
        vertices: (N, 3) mesh vertices in mm
        faces: (M, 3) face indices
        sun_direction: (3,) sun direction vector
        area_budget_mm2: maximum selectable area in mm²
        solar_irradiance_W_m2: solar irradiance at surface (W/m²), default 1000
        efficiency: system efficiency factor (0-1), default 0.25
        power_output_scale: post-scale factor applied to watts output
        sigmoid_sharpness: steepness of selection blur (higher = sharper boundary)
        soft_mask_selection_threshold: threshold for soft selection mask (0.0 to 1.0)
        soft_mask_full_selection_threshold: threshold for full selection mask (0.0 to 1.0)
        epsilon: numerical stability
        enable_wandb_logging: whether to log to wandb
        rendering: whether to render intermediate results
        counter: counter for rendering
        epoch_name: name of the current epoch for logging
        log_group_name: name of the log group in wandb
    Returns:
        scalar power_watts (tensor with gradients for backprop)
    """

    if faces.numel() == 0 or vertices.numel() == 0:
        print("Warning: Empty vertices or faces in solar_mesh_loss_v3, returning zero power")
        return torch.zeros((), device=vertices.device, dtype=vertices.dtype)

    # Compute face areas and normals
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    edge_a, edge_b = v1 - v0, v2 - v0
    face_cross = torch.cross(edge_a, edge_b, dim=-1)
    face_cross_norm = torch.linalg.norm(face_cross, dim=-1)
    face_area_mm2 = 0.5 * face_cross_norm
    face_area_m2 = face_area_mm2 * 1e-6  # Convert to m²

    # Compute illumination quality (cosine of angle with sun direction, between [-1, 1] due to geometry)
    face_normals_unit = face_cross / (face_cross_norm.clamp_min(epsilon)).unsqueeze(-1)
    sun_direction_unit = sun_direction / (torch.linalg.norm(sun_direction).clamp_min(epsilon))
    illumination_quality = (face_normals_unit * sun_direction_unit).sum(-1)

    # Sort faces by descending quality
    # NOTE: Currently ignores order priority for equal-quality faces (TODO: could optimize face ordering)
    quality_order_desc = torch.argsort(illumination_quality, descending=True)
    sorted_areas_m2 = face_area_m2[quality_order_desc]
    sorted_quality = illumination_quality[quality_order_desc]

    # Deterministic-safe cumulative area (in m²)
    cumulative_area_m2 = _efficient_cumsum(sorted_areas_m2, dim=0)
    area_budget_m2 = area_budget_mm2 * 1e-6

    # Create soft selection mask using sigmoid
    # mask(i) = sigmoid(sharpness * (area_budget - cumsum_area[i]))
    # This smoothly transitions from 1.0 (well within budget) to 0.0 (exceeds budget)
    distance_to_budget = (area_budget_m2 - cumulative_area_m2) / max(area_budget_m2, epsilon)
    soft_mask = torch.sigmoid(sigmoid_sharpness * distance_to_budget)

    # Compute power in watts: power = Σ(area x max(quality, 0)) x irradiance x efficiency
    # max(quality, 0) ensures only sun-facing surfaces contribute
    positive_quality = torch.clamp(sorted_quality, min=0.0)
    power_per_face = sorted_areas_m2 * positive_quality * solar_irradiance_W_m2 * efficiency

    # Apply soft mask: weighted power
    weighted_power = power_per_face * soft_mask
    total_power_watts_unscaled = weighted_power.sum()
    total_power_watts = total_power_watts_unscaled * power_output_scale

    # Return positive power in watts
    loss = total_power_watts

    # Sort softmax back to original face order for reasonable logging/visualization
    original_order = torch.argsort(quality_order_desc)
    sorted_soft_mask = soft_mask[original_order]

    # Analyze sigmoid transition zone (faces significantly affected by smooth boundary)
    transition_zone_faces = ((soft_mask > 0.1) & (soft_mask < 0.9)).sum().item()
    faces_near_boundary = (soft_mask > 0.1) & (soft_mask < 0.9)
    transition_zone_area_m2 = (sorted_areas_m2[faces_near_boundary]).sum().item() if faces_near_boundary.any() else 0.0

    # Find the approximate budget crossover point in the sorted list
    budget_crossover_idx = (
        torch.argmax((cumulative_area_m2 >= area_budget_m2).float()).item()
        if (cumulative_area_m2 >= area_budget_m2).any()
        else len(cumulative_area_m2)
    )

    # Store intermediate metrics for logging and rendering
    metrics_log = {
        "face_area_m2": face_area_m2,
        "illumination_quality": illumination_quality,
        "sorted_areas_m2": sorted_areas_m2,
        "sorted_quality": sorted_quality,
        "soft_mask": soft_mask,
        "sorted_soft_mask": sorted_soft_mask,
        "total_power_watts": total_power_watts,
        "total_power_watts_unscaled": total_power_watts_unscaled,
        "power_output_scale": power_output_scale,
        "soft_mask_selection_threshold": soft_mask_selection_threshold,
        "soft_mask_full_selection_threshold": soft_mask_full_selection_threshold,
        # Sigmoid transition analysis
        "transition_zone_faces": transition_zone_faces,
        "transition_zone_area_m2": transition_zone_area_m2,
        "budget_crossover_idx": budget_crossover_idx,
    }

    metrics_render = {
        "vertices": vertices,
        "faces": faces,
        "selection_weights": sorted_soft_mask,
        "threshold": selection_threshold_render,
    }

    # Rendering (if enabled)
    if rendering and render_solar is not None and num_epochs is not None:
        if counter <= num_epochs:
            render_solar.update(optimization_state=metrics_render)
        if counter == num_epochs and output_dir is not None:
            render_solar.finalize(output_dir)

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

    Args:
        loss: Scalar power value (watts)
        vertices: Mesh vertices (N, 3)
        faces: Face indices (M, 3)
        metrics: Dict of intermediate metrics from solar_mesh_loss_v3
        epoch_name: Current epoch number
        log_group_name: Name of the log group in wandb
    """
    try:
        if faces.numel() > 0 and vertices.numel() > 0 and metrics:
            face_area_m2 = metrics["face_area_m2"]
            illumination_quality = metrics["illumination_quality"]
            sorted_areas_m2 = metrics["sorted_areas_m2"]
            soft_mask = metrics["soft_mask"]
            total_power_watts = metrics["total_power_watts"]
            total_power_watts_unscaled = metrics["total_power_watts_unscaled"]
            power_output_scale = metrics["power_output_scale"]
            soft_mask_selection_threshold = metrics["soft_mask_selection_threshold"]
            soft_mask_full_selection_threshold = metrics["soft_mask_full_selection_threshold"]
            transition_zone_faces = metrics["transition_zone_faces"]
            transition_zone_area_m2 = metrics["transition_zone_area_m2"]
            budget_crossover_idx = metrics["budget_crossover_idx"]

            total_area_m2 = face_area_m2.sum().item()
            total_num_faces = len(face_area_m2)
            selected_area_m2 = (sorted_areas_m2 * soft_mask).sum().item()

            # Compute illumination quality statistics
            quality_mean = illumination_quality.mean().item()
            quality_max = illumination_quality.max().item()
            quality_min = illumination_quality.min().item()

            # Compute soft mask statistics
            soft_mask_mean = soft_mask.mean().item()
            soft_mask_max = soft_mask.max().item()
            soft_mask_min = soft_mask.min().item()

            # Count selected faces (using soft_mask thresholds from config)
            selected_faces = (soft_mask > soft_mask_selection_threshold).sum().item()
            fully_selected_faces = (soft_mask > soft_mask_full_selection_threshold).sum().item()

            # Log metrics
            log_dict = {
                # Main power and areas
                "solar_power_watts": float(total_power_watts.detach().cpu().item()),
                "solar_power_watts_unscaled": float(total_power_watts_unscaled.detach().cpu().item()),
                "solar_power_output_scale": float(power_output_scale),
                "solar_loss": float(loss.detach().cpu().item()),
                "total_area_m2": total_area_m2,
                "selected_area_m2": selected_area_m2,
                "selection_ratio": selected_area_m2 / max(total_area_m2, epsilon),
                # Illumination quality statistics
                "solar_quality_mean": quality_mean,
                "solar_quality_max": quality_max,
                "solar_quality_min": quality_min,
                # Soft mask statistics
                "solar_soft_mask_mean": soft_mask_mean,
                "solar_soft_mask_max": soft_mask_max,
                "solar_soft_mask_min": soft_mask_min,
                # Face statistics
                "total_faces": total_num_faces,
                "selected_faces": selected_faces,
                "fully_selected_faces": fully_selected_faces,
                # Sigmoid transition zone analysis
                "transition_zone_faces": transition_zone_faces,
                "transition_zone_area_m2": transition_zone_area_m2,
                "transition_zone_ratio": transition_zone_faces / max(total_num_faces, 1),
                "budget_crossover_idx": budget_crossover_idx,
            }
            multi_log_in_one(log_group_name, log_dict, step=int(epoch_name))
    except Exception as e:
        print(f"Warning: Failed to log solar loss metrics to wandb: {e}")


class SolarLossEstimator:
    def __init__(self, config: dict[str, Any] | None = None) -> None:  # noqa: PLR0915
        defaults = {
            "method": "flexicubes",
            "bbox": [[-5000, 5000], [-5000, 5000], [-5000, 5000]],
            "resolution": 64,
            "area_budget_mm2": 1e6,
            "sun_dir": [0.0, 1.0, 1.0],
            "device": "cuda",
            "batch_size": 4096,
            "eps": 1e-12,
            "sigmoid_sharpness": 10.0,
            "budget_penalty_weight": 0.1,
            "solar_irradiance_W_m2": 1000.0,
            "efficiency": 0.25,
            "power_output_scale": 1.0,
            "wandb_logging": False,
            "soft_mask_selection_threshold": 0.5,
            "soft_mask_full_selection_threshold": 0.95,
            "rendering": False,
            "selection_threshold_render": 0.5,
            "log_group_name": "solar_loss_metrics",
        }

        general_config = load_config(v="v1", cfg_name="general_config.yaml")
        cfg: dict[str, Any] = defaults if config is None else config

        self.config = cfg
        self.general_config = general_config
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
        self.solar_irradiance_W_m2 = float(cfg.get("solar_irradiance_W_m2", 1000.0))
        self.efficiency = float(cfg.get("efficiency", 0.25))
        self.power_output_scale = float(cfg.get("power_output_scale", 1.0))
        self.enable_wandb_logging = bool(cfg["wandb_logging"])
        self.soft_mask_selection_threshold = float(cfg["soft_mask_selection_threshold"])
        self.soft_mask_full_selection_threshold = float(cfg["soft_mask_full_selection_threshold"])
        self.selection_threshold_render = float(cfg["selection_threshold_render"])
        self.counter = 0  # For logging purposes

        self.validation_sim = bool(cfg.get("validation_sim", False))
        self.validation_interval = int(cfg.get("validation_interval", 0))
        self.simulator = None
        if self.validation_sim:
            from subprojects.wp3_solar_estimator.validation.simulationv1 import SimulationV1  # noqa: I001, PLC0415 Since this saves importing of simulation if not needed

            australia_cfg = cfg.get("australia_cfg", "solar_configs/australia.yaml")
            self.simulator = SimulationV1(cfg_path=australia_cfg, area_budget_mm2=self.area_budget_mm2)

        self.rendering = bool(cfg["rendering"])
        self.visualiser: Visualiser | None
        self.renderSolar: SolarV3Selection | None
        self.render_grads: MeshSDFGrads | None
        self.number_epochs: int | None
        self.model_hash: str | None
        self.wandb_run_name: str | None
        self.rendering_dir: Path | None
        if self.rendering:
            self.visualiser = Visualiser(
                image_size=1024,
                background_color=None,
                anti_aliasing=True,
                camera_mode="look_at",
                viewing_angle=30.0,
                device=self.device,
                up=[0.0, 0.0, 1.0],
            )  # Will not be used if rendering is False
            self.renderSolar = SolarV3Selection(visualiser=self.visualiser)  # Will not be used if rendering is False
            detached_cfg = load_config(v="v1", cfg_name="neural_renderer_v0.yaml").get("detached_modes", {})
            self.render_grads = (
                MeshSDFGrads(visualiser=self.visualiser) if detached_cfg.get("MeshSDF_grads", {}).get("enabled", False) else None
            )
            train_cfg = load_config(v="v1", cfg_name="training_v0_config.yaml")["training_params"]
            self.number_epochs = int(train_cfg["num_epochs"])
            self.model_hash = None  # Will be set when solar_loss is first called with a model
            self.wandb_run_name = wandb.run.name if wandb.run else None
            self.rendering_dir = None  # Will be set when solar_loss is first called with a model
            self.storage_paths = general_config["storage_paths"]
        else:
            self.visualiser = None
            self.renderSolar = None
            self.render_grads = None
            self.number_epochs = None
            self.model_hash = None
            self.wandb_run_name = None
            self.rendering_dir = None

        sun_dir = cfg["sun_dir"]
        if not isinstance(sun_dir, torch.Tensor):
            self.sun_dir = torch.tensor(sun_dir, device=self.device, dtype=torch.float32)
        else:
            self.sun_dir = sun_dir.to(device=self.device, dtype=torch.float32)

        self.extractor: FCExtractor | DISOExtractor | MeshSDFExtractor | SparseDMCExtractor
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
            isovalue = float(self.config.get("isovalue", 0.0))
            auto_iso = bool(self.config.get("auto_isovalue_if_out_of_range", True))
            warn_out_of_range = bool(self.config.get("warn_mesh_extraction", True))
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
                isovalue=isovalue,
                device=self.device,
                auto_isovalue_if_out_of_range=auto_iso,
                warn_out_of_range=warn_out_of_range,
            )
            self.extractor.set_grid(self.bbox, self.resolution)
        elif method == "sparse_dmc":
            self.extractor = SparseDMCExtractor(bbox=self.bbox, resolution=self.resolution, device=self.device)
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
        if self.method == "sparse_dmc":
            assert isinstance(self.extractor, SparseDMCExtractor)
            return self.extractor.extract_mesh(model)
        raise ValueError(f"Unknown method: {self.method}")

    def solar_loss(
        self,
        model: Any,
        epoch_name: int | None = None,
        disable_logging: bool = False,
        disable_rendering: bool = False,
    ) -> torch.Tensor:
        """
        Compute the solar objective for the given model.

        Returns positive solar power in watts.
        For mesh_sdf, backward gradients come from the surrogate path while
        forward value remains the meaningful power value.

        Args:
            model: SDF model (nn.Module)
            epoch_name: Current epoch (for logging)
            disable_logging: Override flag to disable wandb logging for this call
            disable_rendering: Override flag to disable rendering for this call

        Returns:
            Scalar tensor = power_watts
        """

        print(
            f"[SolarLossEstimator] Mesh extraction method for this loss evaluation: "
            f"{self.method} (extractor={type(self.extractor).__name__})"
        )

        # Initialize model_hash and rendering_dir on first call
        if self.rendering and self.model_hash is None:
            self.model_hash = dict_hashing(model.get_model_info_dict())
            run_name = reformat_wandb_name(self.wandb_run_name) if self.wandb_run_name else "no_wandb_run"
            if self.general_config["general_info"].get("run_locally", False).lower() == "true":
                storage_path = self.storage_paths["local_storage_dir"]
            else:
                storage_path = self.storage_paths["storage_dir"]
            self.rendering_dir = Path(storage_path) / self.model_hash / run_name / "solar_loss_rendering"
            print(f"Rendering enabled: saving to {self.rendering_dir}")
            os.makedirs(self.rendering_dir, exist_ok=True)
            print(f"Model hash: {self.model_hash}")
            print(f"Wandb run name: {self.wandb_run_name}")

        self.counter += 1

        if self.method == "mesh_sdf":  # MeshSDF loss needs special handling

            def wrapper_solar_loss(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
                if self.validation_sim and self.simulator is not None:
                    run_sim = False
                    if (self.validation_interval > 0 and self.counter % self.validation_interval == 0) or (
                        self.number_epochs is not None and self.counter == self.number_epochs
                    ):
                        run_sim = True

                    if run_sim and not disable_rendering:
                        self.simulator.run(vertices, faces, output_dir=self.rendering_dir, epoch_name=epoch_name)

                        if (
                            self.number_epochs is not None
                            and self.counter == self.number_epochs
                            and self.rendering_dir is not None
                        ):
                            from subprojects.wp4_aeroshell_optimizer import (  # noqa: PLC0415
                                create_and_upload_gif_from_folder,
                            )

                            create_and_upload_gif_from_folder(self.rendering_dir / "sim_renders", step=self.counter, fps=10)

                loss = solar_mesh_loss_v3(
                    vertices,
                    faces,
                    self.sun_dir,
                    self.area_budget_mm2,
                    solar_irradiance_W_m2=self.solar_irradiance_W_m2,
                    efficiency=self.efficiency,
                    power_output_scale=self.power_output_scale,
                    sigmoid_sharpness=self.sigmoid_sharpness,
                    soft_mask_selection_threshold=self.soft_mask_selection_threshold,
                    soft_mask_full_selection_threshold=self.soft_mask_full_selection_threshold,
                    epsilon=self.eps,
                    epoch_name=epoch_name,
                    log_group_name=self.log_group_name,
                    enable_wandb_logging=self.enable_wandb_logging and not disable_logging,
                    rendering=self.rendering and not disable_rendering,
                    counter=self.counter,
                    output_dir=self.rendering_dir,
                    num_epochs=self.number_epochs,
                    render_solar=self.renderSolar,
                    selection_threshold_render=self.selection_threshold_render,
                )
                return loss

            assert isinstance(self.extractor, MeshSDFExtractor)  # Get mypy to shut up, confirm we are using MeshSDF
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

                if run_sim and not disable_rendering:
                    self.simulator.run(vertices, faces, output_dir=self.rendering_dir, epoch_name=epoch_name)

                    if self.number_epochs is not None and self.counter == self.number_epochs and self.rendering_dir is not None:
                        from subprojects.wp4_aeroshell_optimizer import (  # noqa: PLC0415
                            create_and_upload_gif_from_folder,
                        )

                        create_and_upload_gif_from_folder(self.rendering_dir / "sim_renders", step=self.counter, fps=10)

            loss = solar_mesh_loss_v3(
                vertices,
                faces,
                self.sun_dir,
                self.area_budget_mm2,
                solar_irradiance_W_m2=self.solar_irradiance_W_m2,
                efficiency=self.efficiency,
                power_output_scale=self.power_output_scale,
                sigmoid_sharpness=self.sigmoid_sharpness,
                soft_mask_selection_threshold=self.soft_mask_selection_threshold,
                soft_mask_full_selection_threshold=self.soft_mask_full_selection_threshold,
                epsilon=self.eps,
                epoch_name=epoch_name,
                log_group_name=self.log_group_name,
                enable_wandb_logging=self.enable_wandb_logging and not disable_logging,
                rendering=self.rendering and not disable_rendering,
                counter=self.counter,
                output_dir=self.rendering_dir,
                num_epochs=self.number_epochs,
                render_solar=self.renderSolar,
                selection_threshold_render=self.selection_threshold_render,
            )

        if self.method != "mesh_sdf":
            return loss

        # Straight-through merge:
        # - forward value: meaningful power in watts (loss)
        # - backward gradient: surrogate gradient (surrogate_loss)
        return surrogate_loss + (loss - surrogate_loss).detach()
