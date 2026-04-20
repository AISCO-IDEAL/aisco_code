"""
Solar loss (v3): fully differentiable soft-selection loss without custom autograd.

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
from subprojects.wp4_aeroshell_optimizer import (
    SolarV3Selection,
    Visualiser,
    dict_hashing,
    load_config,
    multi_log_in_one,
    reformat_wandb_name,
)


def _cumsum(x: torch.Tensor) -> torch.Tensor:
    """
    Manual cumulative sum implementation that is deterministic

    Args:
        x: (N,) input tensor
    Returns:
        (N,) cumulative sum tensor where cumsum[i] = x[0] + x[1] + ... + x[i]
    """
    cumsum = torch.zeros_like(x)
    running_sum = torch.zeros((), device=x.device, dtype=x.dtype)
    for i in range(x.shape[0]):
        running_sum += x[i]
        cumsum[i] = running_sum
    return cumsum


def solar_mesh_loss_v3(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    sun_direction: torch.Tensor,
    area_budget_mm2: float,
    sigmoid_sharpness: float = 10.0,
    soft_mask_selection_threshold: float = 0.5,
    soft_mask_full_selection_threshold: float = 0.95,
    budget_penalty_weight: float = 0.1,
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
    Compute differentiable solar mesh loss using soft selection.

    Args:
        vertices: (N, 3) mesh vertices in mm
        faces: (M, 3) face indices
        sun_direction: (3,) sun direction vector
        area_budget_mm2: maximum selectable area in mm²
        sigmoid_sharpness: steepness of selection blur (higher = sharper boundary)
        soft_mask_selection_threshold: threshold for soft selection mask (0.0 to 1.0)
        soft_mask_full_selection_threshold: threshold for full selection mask (0.0 to 1.0)
        budget_penalty_weight: weight for unused budget penalty
        epsilon: numerical stability
        enable_wandb_logging: whether to log to wandb
        rendering: whether to render intermediate results
        counter: counter for rendering
        epoch_name: name of the current epoch for logging
        log_group_name: name of the log group in wandb
    Returns:
        scalar loss in m²
    """

    print(f"Number of epochs: {epoch_name}, Counter: {counter}")

    if faces.numel() == 0 or vertices.numel() == 0:
        print("Warning: Empty vertices or faces in solar_mesh_loss_v3, returning zero loss")
        return torch.zeros((), device=vertices.device, dtype=vertices.dtype)

    # Compute face areas and normals
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    edge_a, edge_b = v1 - v0, v2 - v0
    face_cross = torch.cross(edge_a, edge_b, dim=-1)
    face_cross_norm = torch.linalg.norm(face_cross, dim=-1)
    face_area_mm2 = 0.5 * face_cross_norm

    # Compute illumination quality (cosine of angle with sun direction, clamped to [0, 1])
    face_normals_unit = face_cross / (face_cross_norm.clamp_min(epsilon)).unsqueeze(-1)
    sun_direction_unit = sun_direction / (torch.linalg.norm(sun_direction).clamp_min(epsilon))
    illumination_quality = torch.clamp((face_normals_unit * sun_direction_unit).sum(-1), 0.0, 1.0)

    # Sort faces by descending quality
    # NOTE: Currently ignores order priority for equal-quality faces (TODO: could optimize face ordering)
    quality_order_desc = torch.argsort(illumination_quality, descending=True)
    sorted_areas = face_area_mm2[quality_order_desc]
    sorted_quality = illumination_quality[quality_order_desc]

    # Deterministic-safe cumulative area
    cumulative_area = _cumsum(sorted_areas)

    # Create soft selection mask using sigmoid
    # mask(i) = sigmoid(sharpness * (area_budget - cumsum_area[i]))
    # This smoothly transitions from 1.0 (well within budget) to 0.0 (exceeds budget)
    distance_to_budget = (area_budget_mm2 - cumulative_area) / max(area_budget_mm2, epsilon)
    soft_mask = torch.sigmoid(sigmoid_sharpness * distance_to_budget)

    # Weighted quality-area product
    weighted_quality_area = sorted_areas * sorted_quality * soft_mask
    quality_loss = weighted_quality_area.sum()

    # Penalty for unused budget (encourage filling the budget)
    total_selected_area = (sorted_areas * soft_mask).sum()
    # Normalize budget usage to dimensionless ratio
    unused_ratio = torch.clamp((area_budget_mm2 - total_selected_area) / max(area_budget_mm2, epsilon), min=0.0)
    budget_penalty = budget_penalty_weight * (unused_ratio**2)

    # Combined loss: maximize quality (minimize negative) + penalize unused budget
    loss_mm2 = quality_loss - budget_penalty

    # Sort softmax back to original face order for reasonable logging/visualization
    original_order = torch.argsort(quality_order_desc)
    sorted_soft_mask = soft_mask[original_order]

    # Store intermediate metrics for logging and rendering
    metrics_log = {
        "face_area_mm2": face_area_mm2,
        "illumination_quality": illumination_quality,
        "sorted_areas": sorted_areas,
        "sorted_quality": sorted_quality,
        "soft_mask": soft_mask,
        "sorted_soft_mask": sorted_soft_mask,
        "quality_loss": quality_loss,
        "budget_penalty": budget_penalty,
        "soft_mask_selection_threshold": soft_mask_selection_threshold,
        "soft_mask_full_selection_threshold": soft_mask_full_selection_threshold,
    }

    metrics_render = {
        "vertices": vertices,
        "faces": faces,
        "selection_weights": sorted_soft_mask,
        "threshold": selection_threshold_render,
    }

    loss = loss_mm2 * 1e-6  # Convert to m² for final loss

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
        loss: Scalar loss value (in m²)
        vertices: Mesh vertices (N, 3)
        faces: Face indices (M, 3)
        metrics: Dict of intermediate metrics from solar_mesh_loss_v3
        epoch_name: Current epoch number
        log_group_name: Name of the log group in wandb
    """
    try:
        if faces.numel() > 0 and vertices.numel() > 0 and metrics:
            face_area_mm2 = metrics["face_area_mm2"]  # (M,) area of each face in mm²
            illumination_quality = metrics["illumination_quality"]  # (M,) cosine similarity with sun direction
            sorted_areas = metrics["sorted_areas"]  # (M,) face areas sorted by descending quality
            soft_mask = metrics["soft_mask"]  # (M,) soft selection mask values for each face
            quality_loss = metrics["quality_loss"]  # Scalar tensor representing the weighted quality-area loss component
            budget_penalty = metrics["budget_penalty"]  # Scalar tensor representing the penalty for unused area budget
            soft_mask_selection_threshold = metrics["soft_mask_selection_threshold"]
            soft_mask_full_selection_threshold = metrics["soft_mask_full_selection_threshold"]

            total_area_mm2 = face_area_mm2.sum().item()  # Total area of all faces in mm²
            total_num_faces = len(face_area_mm2)  # Total number of faces in the mesh
            selected_area_mm2 = (
                (sorted_areas * soft_mask).sum().item()
            )  # Total area of selected faces (weighted by soft mask) in mm²

            # Compute illumination quality statistics
            quality_mean = illumination_quality.mean().item()  # Mean illumination quality across all faces
            quality_max = illumination_quality.max().item()  # Maximum illumination quality across all faces
            quality_min = illumination_quality.min().item()  # Minimum illumination quality across all faces

            # Compute soft mask statistics
            soft_mask_mean = soft_mask.mean().item()  # Mean soft mask value across all faces, Average "softness" across all faces. High mean → most faces are selected; low mean → most faces rejected
            soft_mask_max = soft_mask.max().item()  # Maximum soft mask value across all faces, Indicates the highest selection confidence among all faces. High max → at least one face is fully selected; low max → no face is fully selected
            soft_mask_min = soft_mask.min().item()  # Minimum soft mask value across all faces, Indicates the lowest selection confidence among all faces. High min → even the least selected face has a high selection confidence; low min → at least one face is strongly rejected

            # Count selected faces (using soft_mask thresholds from config)
            selected_faces = (
                (soft_mask > soft_mask_selection_threshold).sum().item()
            )  # Number of faces with soft mask above selection threshold, Indicates how many faces are considered "selected" for the loss computation based on the configured threshold. Higher means more faces contribute to the loss; lower means fewer faces contribute.
            fully_selected_faces = (
                (soft_mask > soft_mask_full_selection_threshold).sum().item()
            )  # Number of faces with soft mask above full selection threshold, Indicates how many faces are considered "fully selected" (i.e., have a very high selection confidence) based on the configured full selection threshold. Higher means more faces are strongly contributing to the loss; lower means fewer faces are strongly contributing.

            # Log metrics
            log_dict = {
                # Main loss and areas
                "solar_loss": float(loss.detach().cpu().item()),
                "solar_total_area_m2": total_area_mm2 * 1e-6,
                "solar_selected_area_m2": selected_area_mm2 * 1e-6,
                "solar_selection_ratio": selected_area_mm2 / max(total_area_mm2, epsilon),
                # Loss components
                "solar_quality_loss": float(quality_loss.detach().cpu().item()) * 1e-6,
                "solar_budget_penalty": float(budget_penalty.detach().cpu().item()),
                # Illumination quality statistics
                "solar_quality_mean": quality_mean,
                "solar_quality_max": quality_max,
                "solar_quality_min": quality_min,
                # Soft mask statistics
                "solar_soft_mask_mean": soft_mask_mean,
                "solar_soft_mask_max": soft_mask_max,
                "solar_soft_mask_min": soft_mask_min,
                # Face statistics
                "solar_total_faces": total_num_faces,
                "solar_selected_faces": selected_faces,
                "solar_fully_selected_faces": fully_selected_faces,
            }
            multi_log_in_one(log_group_name, log_dict, step=int(epoch_name))
    except Exception as e:
        print(f"Warning: Failed to log solar loss metrics to wandb: {e}")


class SolarLossEstimator:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
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
            "wandb_logging": False,
            "soft_mask_selection_threshold": 0.5,
            "soft_mask_full_selection_threshold": 0.95,
            "rendering": False,
            "selection_threshold_render": 0.5,
            "log_group_name": "solar_loss_metrics",
        }

        general_config = load_config(v="v1", cfg_name="general_config.yaml")
        cfg: dict[str, Any] = defaults if config is None else config
        print(cfg)
        self.config = cfg

        self.method = cfg["method"]
        self.bbox = cfg["bbox"]
        self.resolution = int(cfg["resolution"])
        self.area_budget_mm2 = float(cfg["area_budget_mm2"])
        self.device = cfg["device"]
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            print(f"Configured solar estimator device '{self.device}' unavailable. Falling back to CPU.")
            self.device = "cpu"
        self.batch_size = int(cfg["batch_size"])
        self.eps = float(cfg["eps"])
        self.log_group_name = str(cfg["log_group_name"])
        self.sigmoid_sharpness = float(cfg["sigmoid_sharpness"])
        self.budget_penalty_weight = float(cfg["budget_penalty_weight"])
        self.enable_wandb_logging = bool(cfg["wandb_logging"])
        self.soft_mask_selection_threshold = float(cfg["soft_mask_selection_threshold"])
        self.soft_mask_full_selection_threshold = float(cfg["soft_mask_full_selection_threshold"])
        self.selection_threshold_render = float(cfg["selection_threshold_render"])
        self.counter = 0  # For logging purposes

        self.rendering = bool(cfg["rendering"])
        self.visualiser: Visualiser | None
        self.renderSolar: SolarV3Selection | None
        self.number_epochs: int | None
        self.model_hash: str | None
        self.wandb_run_name: str | None
        self.rendering_dir: Path | None
        if self.rendering and self.enable_wandb_logging:
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
            train_cfg = load_config(v="v1", cfg_name="training_v0_config.yaml")["training_params"]
            self.number_epochs = int(train_cfg["num_epochs"])
            self.model_hash = None  # Will be set when solar_loss is first called with a model
            self.wandb_run_name = wandb.run.name if wandb.run else None
            self.rendering_dir = None  # Will be set when solar_loss is first called with a model
            self.storage_paths = general_config["storage_paths"]
        else:
            self.visualiser = None
            self.renderSolar = None
            self.number_epochs = None
            self.model_hash = None
            self.wandb_run_name = None
            self.rendering_dir = None

        sun_dir = cfg["sun_dir"]
        if not isinstance(sun_dir, torch.Tensor):
            self.sun_dir = torch.tensor(sun_dir, device=self.device, dtype=torch.float32)
        else:
            self.sun_dir = sun_dir.to(device=self.device, dtype=torch.float32)

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

        Args:
            model: SDF model (nn.Module)
            epoch_name: Current epoch (for logging)

        Returns:
            Scalar loss tensor
        """

        # Initialize model_hash and rendering_dir on first call
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

        vertices, faces = self.extract_mesh(model)
        if self.method == "mesh_sdf":  # MeshSDF loss needs special handling
            print("Using MeshSDF for Solar loss, gradients may be weird")

            def wrapper_solar_loss(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
                loss = solar_mesh_loss_v3(
                    vertices,
                    faces,
                    self.sun_dir,
                    self.area_budget_mm2,
                    sigmoid_sharpness=self.sigmoid_sharpness,
                    soft_mask_selection_threshold=self.soft_mask_selection_threshold,
                    soft_mask_full_selection_threshold=self.soft_mask_full_selection_threshold,
                    budget_penalty_weight=self.budget_penalty_weight,
                    epsilon=self.eps,
                    epoch_name=epoch_name,
                    log_group_name=self.log_group_name,
                    enable_wandb_logging=self.enable_wandb_logging,
                    rendering=self.rendering,
                    counter=self.counter,
                    output_dir=self.rendering_dir,
                    num_epochs=self.number_epochs,
                    render_solar=self.renderSolar,
                    selection_threshold_render=self.selection_threshold_render,
                )
                return loss

            assert isinstance(self.extractor, MeshSDFExtractor)
            _verts, faces, loss, surrogate_loss, _vertice_grads = self.extractor.extract_and_compute_loss(
                sdf_model=model,
                loss_fn=wrapper_solar_loss,
                enable_backward=True,
            )

        else:
            print(f"Using non-MeshSDF method for Solar loss ({self.method}), should be fully differentiable")
            loss = solar_mesh_loss_v3(
                vertices,
                faces,
                self.sun_dir,
                self.area_budget_mm2,
                sigmoid_sharpness=self.sigmoid_sharpness,
                soft_mask_selection_threshold=self.soft_mask_selection_threshold,
                soft_mask_full_selection_threshold=self.soft_mask_full_selection_threshold,
                budget_penalty_weight=self.budget_penalty_weight,
                epsilon=self.eps,
                epoch_name=epoch_name,
                log_group_name=self.log_group_name,
                enable_wandb_logging=self.enable_wandb_logging,
                rendering=self.rendering,
                counter=self.counter,
                output_dir=self.rendering_dir,
                num_epochs=self.number_epochs,
                render_solar=self.renderSolar,
                selection_threshold_render=self.selection_threshold_render,
            )

        return loss if self.method != "mesh_sdf" else surrogate_loss


def sweep_solar_methods(
    model: Any,
    methods: list[str],
    config_factory: Callable[[str], dict],
    compute_grad_norm: bool = False,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for method in methods:
        cfg = config_factory(method)
        loss_fn = SolarLossEstimator(cfg)
        loss_val = loss_fn.solar_loss(model)
        entry: dict[str, float] = {"loss": float(loss_val.detach().cpu().item())}

        if compute_grad_norm:
            if hasattr(model, "zero_grad"):
                model.zero_grad(set_to_none=True)
            grads = torch.autograd.grad(loss_val, [p for p in model.parameters() if p.requires_grad], allow_unused=True)
            grad_norm = 0.0
            for g in grads:
                if g is not None:
                    grad_norm += float(g.detach().norm().cpu().item())
            entry["grad_norm"] = grad_norm

        results[method] = entry

    return results
