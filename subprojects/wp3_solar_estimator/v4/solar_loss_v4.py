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

from subprojects.wp3_solar_estimator.australia.route_v1 import RouteV1 as Route
from subprojects.wp3_solar_estimator.mesh_extractors.diso.diso_wrapper import DISOExtractor
from subprojects.wp3_solar_estimator.mesh_extractors.kaolin_flexicubes.flexicubes_wrapper import FCExtractor
from subprojects.wp3_solar_estimator.mesh_extractors.mesh_sdf.mesh_sdf_extractor import MeshSDFExtractor
from subprojects.wp3_solar_estimator.mesh_extractors.sparse_dmc.sparse_dmc_extractor import SparseDMCExtractor
from subprojects.wp3_solar_estimator.utils.solar_utils import averaging_solar_quality
from subprojects.wp4_aeroshell_optimizer.v1.logging_lib.basic_plots import multi_log_in_one
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.hashing import dict_hashing
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.storage_utils import reformat_wandb_name
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.detached_modes.solar_v3_selection import SolarV3Selection
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser import Visualiser


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


def solar_mesh_loss_v4(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    solar_vectors: torch.Tensor,
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
        solar_vectors: (K, 3) solar vectors along the route
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
    face_normals_unit = face_cross / (face_cross_norm.clamp_min(epsilon)).unsqueeze(-1)  # (M, 3) unit normals of faces
    sun_directions_unit = solar_vectors / (
        torch.linalg.norm(solar_vectors, dim=-1, keepdim=True).clamp_min(epsilon)
    )  # (K,3) unit solar vectors
    # now calculate the (average) illumination quality (solar_vectors dot normal vector) of each face for all solar vectors, use average so that we get a single quality value per face that considers all solar vectors along the route

    aggregate_illumination_quality = averaging_solar_quality(
        face_normals_unit, sun_directions_unit
    )  # (M,) average quality of each face across all solar vectors
    # Sort faces by descending quality
    # NOTE: Currently ignores order priority for equal-quality faces (TODO: could optimize face ordering)
    quality_order_desc = torch.argsort(aggregate_illumination_quality, descending=True)
    sorted_areas = face_area_mm2[quality_order_desc]
    sorted_quality = aggregate_illumination_quality[quality_order_desc]

    # Deterministic-safe cumulative area
    cumulative_area = _efficient_cumsum(sorted_areas, dim=0)

    # Create soft selection mask using sigmoid
    # mask(i) = sigmoid(sharpness * (area_budget - cumsum_area[i]))
    # This smoothly transitions from 1.0 (well within budget) to 0.0 (exceeds budget)
    distance_to_budget = (area_budget_mm2 - cumulative_area) / max(
        area_budget_mm2, epsilon
    )  # Consider dividing by just 1e6 instead
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

    # Analyze sigmoid transition zone (faces significantly affected by smooth boundary)
    transition_zone_faces = ((soft_mask > 0.1) & (soft_mask < 0.9)).sum().item()
    faces_near_boundary = (soft_mask > 0.1) & (soft_mask < 0.9)
    transition_zone_area_mm2 = (sorted_areas[faces_near_boundary]).sum().item() if faces_near_boundary.any() else 0.0

    # Find the approximate budget crossover point in the sorted list
    budget_crossover_idx = (
        torch.argmax((cumulative_area >= area_budget_mm2).float()).item()
        if (cumulative_area >= area_budget_mm2).any()
        else len(cumulative_area)
    )

    # Store intermediate metrics for logging and rendering
    metrics_log = {
        "face_area_mm2": face_area_mm2,
        "illumination_quality": aggregate_illumination_quality,
        "sorted_areas": sorted_areas,
        "sorted_quality": sorted_quality,
        "soft_mask": soft_mask,
        "sorted_soft_mask": sorted_soft_mask,
        "quality_loss": quality_loss,
        "budget_penalty": budget_penalty,
        "soft_mask_selection_threshold": soft_mask_selection_threshold,
        "soft_mask_full_selection_threshold": soft_mask_full_selection_threshold,
        # Sigmoid transition analysis
        "transition_zone_faces": transition_zone_faces,
        "transition_zone_area_mm2": transition_zone_area_mm2,
        "budget_crossover_idx": budget_crossover_idx,
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
            # Sigmoid transition zone metrics
            transition_zone_faces = metrics["transition_zone_faces"]
            transition_zone_area_mm2 = metrics["transition_zone_area_mm2"]
            budget_crossover_idx = metrics["budget_crossover_idx"]

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
                "total_area_m2": total_area_mm2 * 1e-6,
                "selected_area_m2": selected_area_mm2 * 1e-6,
                "selection_ratio": selected_area_mm2 / max(total_area_mm2, epsilon),
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
                "total_faces": total_num_faces,
                "selected_faces": selected_faces,
                "fully_selected_faces": fully_selected_faces,
                # Sigmoid transition zone analysis
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

        self.config = cfg

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
            self.run_locally = general_config["general_info"].get("run_locally", "False") == "True"
        else:
            self.visualiser = None
            self.renderSolar = None
            self.number_epochs = None
            self.model_hash = None
            self.wandb_run_name = None
            self.rendering_dir = None
            self.run_locally = False

        self.route = Route(
            cfg_path=config["australia_cfg"]
        )  # For solar vector calculation -- used to aggregate different solar vecs from australia cfg into mixed quality (no shading yet)
        self.solar_vectors = (
            self.route.get_route_solar_vecs()
        )  # (N,T, 3) solar vectors along the route, pre-moved to device for efficiency
        # ensure format is (K,3) where K is the number of solar vectors (e.g. number of points along the route), by flattening the first two dimensions if necessary
        self.solar_vectors = self.solar_vectors.view(-1, 3).to(self.device)
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
        elif method == "sparse_sdf":
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
        if self.method == "sparse_sdf":
            assert isinstance(self.extractor, SparseDMCExtractor)
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
                storage_path = self.storage_paths["local_storage_dir"] if self.run_locally else self.storage_paths["storage_dir"]
                self.rendering_dir = (
                    Path(storage_path) / self.model_hash / reformat_wandb_name(self.wandb_run_name) / "solar_loss_rendering"
                )
                print(f"Rendering enabled: saving to {self.rendering_dir}")
                os.makedirs(self.rendering_dir, exist_ok=True)
            else:
                print("Warning: Wandb logging is enabled but no active run found, solar_loss_rendering will not be saved")
                self.rendering_dir = None
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

                    if run_sim:
                        self.simulator.run(vertices, faces, output_dir=self.rendering_dir, epoch_name=epoch_name)

                        if (
                            self.number_epochs is not None
                            and self.counter == self.number_epochs
                            and self.rendering_dir is not None
                        ):
                            from subprojects.wp4_aeroshell_optimizer.v1.logging_lib.image_uploads import (  # noqa: PLC0415 So we can save us from importing
                                create_and_upload_gif_from_folder,
                            )

                            create_and_upload_gif_from_folder(self.rendering_dir / "sim_renders", step=self.counter, fps=10)

                loss = solar_mesh_loss_v4(
                    vertices,
                    faces,
                    self.solar_vectors,
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

            assert isinstance(self.extractor, MeshSDFExtractor)  # Get mypy to shut up, confirm we are using MeshSDF
            _vertices, _faces, loss, surrogate_loss = self.extractor.extract_and_compute_loss(
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
                        from subprojects.wp4_aeroshell_optimizer.v1.logging_lib.image_uploads import (  # noqa: PLC0415 So we can save us from importing
                            create_and_upload_gif_from_folder,
                        )

                        create_and_upload_gif_from_folder(self.rendering_dir / "sim_renders", step=self.counter, fps=10)

            loss = solar_mesh_loss_v4(
                vertices,
                faces,
                self.solar_vectors,
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
