from pathlib import Path

import torch
import trimesh

import wandb
from subprojects.wp3_solar_estimator.mesh_extractors.non_diff_algos.marching_cubes import MeshExtractor
from subprojects.wp3_solar_estimator.validation.statistics import MeshStatistics
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.cruising_speed import cruising_speed
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.weights import weights_alm, weights_manual
from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.dynamic_loading import load_class_from_path
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.visualizer import CADVisualizer
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visual_wrapper import VisualisationManager


class TotalLoss:
    def __init__(self, dev: str = "cuda", renderer_manager: VisualisationManager | None = None) -> None:
        # set up configs
        self.__general_config = load_config()  # default arg is the general config
        self._use_wandb = self.__general_config.get("wandb_general", {}).get("use_wandb", True)
        self._use_renderer = self.__general_config.get("wandb_general", {}).get("use_renderer", True)
        self._use_neural_renderer = self.__general_config.get("wandb_general", {}).get(
            "use_neural_renderer", False
        )  # Does anything?
        self.losses_config = load_config(
            v=self.__general_config["general_info"]["version"],
            cfg_name=self.__general_config["active_configs"]["losses"],
        )
        # NOTE: sweep overrides are applied automatically by the config
        # registry (see config_utils.py). No manual wandb.config patching needed.

        self.weights = self.losses_config.get("losses", {})  # get nested losses dict
        self.__active_losses: dict = {}
        self._renderer_initialized = False
        self._renderer_manager = renderer_manager
        # Populated each call to get_total_loss(); maps loss component name → weighted live tensor.
        # The training loop uses this to call .backward() per component instead of on the summed total.
        self._last_weighted_losses: dict[str, torch.Tensor] = {}

        project_root = Path.cwd()  # Correctly defines project root
        self.losses = self.losses_config.get("losses", {})  # get nested losses dict
        self.__rendering_frequency = self.__general_config.get("wandb_general", {}).get("rendering_frequency")
        self._mesh_statistics = MeshStatistics(device=dev, use_wandb=self._use_wandb)  # For logging mesh statistics to wandb

        # Load logging feature flags (once, not per iteration)
        _logging_config = load_config(v="v1", cfg_name="logging.yaml")
        self._memory_diagnostics: bool = _logging_config.get("memory_diagnostics", False)
        self._mesh_statistics_enabled: bool = _logging_config.get("mesh_statistics", False)
        self._gradient_memory_diagnostics: bool = _logging_config.get("gradient_memory_diagnostics", False)

        # Initialize mesh extractor with config from neural_renderer_v0.yaml
        renderer_config = load_config(v="v1", cfg_name="neural_renderer_v0.yaml")
        mesh_config = renderer_config.get("mesh_extraction", {})
        self._mesh_extractor = MeshExtractor(
            algorithm=mesh_config.get("algorithm", "mc"),
            device=mesh_config.get("device", dev),
            verbose=mesh_config.get("verbose", False),  # Keep quiet during training by default
        )

        # Pre-configure grid for repeated extractions (major speedup)
        self._mesh_extractor.set_grid(
            bbox=mesh_config.get("bbox", [[-5000, 5000], [-5000, 5000], [-5000, 5000]]),
            resolution=mesh_config.get("resolution", 128),  # Default 128 for compatibility with old get_diff_mesh
        )
        self._mesh_extraction_config = mesh_config  # Store for later use

        # ALM parameters init
        # Values come from losses_config which is already sweep-patched by the
        # config registry.  To sweep these, use parameter names like:
        #   losses_v0_config.general_info.initial_lambda
        #   losses_v0_config.general_info.alpha
        #   losses_v0_config.general_info.gamma
        if self.losses_config["general_info"].get("ALM", False):
            alm_info = self.losses_config["general_info"]
            self.lamda: float = alm_info.get("initial_lambda", 0.01)
            self.mu: float = alm_info.get("initial_mu", 1.0)
            self.nu: float = alm_info.get("initial_nu", 1.0)
            self.lamdas = torch.tensor([[self.lamda], [self.lamda], [self.lamda]], device=dev)
            self.mus = torch.tensor(
                [[self.mu], [self.mu], [self.mu]], device=dev
            )  # η=1 balances constraint enforcement and objective minimization
            # Ref: "Conditionally adaptive augmented Lagrangian method" — Hua, Basira, Senocak
            #      https://arxiv.org/html/2508.15695v1
            self.nus = torch.tensor([[self.nu], [self.nu], [self.nu]], device=dev)
            self.alpha: float = alm_info.get("alpha", 0.99)
            self.alphas = torch.tensor(
                [[self.alpha], [self.alpha], [self.alpha]], device=dev
            )  # increase mu by this factor if constraint not met
            self.gamma: float = alm_info.get("gamma", 0.01)
            self.gamma_aggressive: float = alm_info.get(
                "gamma_aggressive", 1.1
            )  # multiplicative increase factor for mu in aggressive CAPU mode
            # Ref: "An Adaptive Augmented Lagrangian Method for Training Physics and Equality
            #       Constrained ANNs" — Basir & Senocak, https://arxiv.org/pdf/2306.04904

        # Dynamically load and initialise active losses
        # category: aero, ... | lossnames: frontal_area_v0, ...
        for category, lossnames in self.losses.items():
            if category == "total":
                continue

            # loss_name: frontal_area_v0, ... | compute: bool | weight: float | path: str
            for loss_name, (compute, _weight, path) in lossnames.items():
                if compute:
                    # Dynamically load the loss class
                    # 1. Correctly construct the absolute path from the relative path in config
                    absolute_path = project_root / Path(path)

                    loss_family = Path(path).stem.title().replace("_", "")

                    # 2. Call load_class_from_path with the absolute path string
                    loss_class = load_class_from_path(str(absolute_path), loss_family)
                    loss_instance = loss_class(
                        device=dev,
                        loss_name=loss_name,
                        losses_config=self.losses_config,
                    )
                    self.__active_losses[loss_name] = loss_instance

    def _log_memory_diagnostics(self, epoch_name: int, device: torch.device) -> None:
        """Print and optionally log GPU memory stats. Toggled by memory_diagnostics in logging.yaml."""
        if not self._memory_diagnostics:
            return
        if device.type != "cuda":
            return
        allocated_mb = torch.cuda.memory_allocated(device) / 1024**2
        reserved_mb = torch.cuda.memory_reserved(device) / 1024**2
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        free_cached_mb = reserved_mb - allocated_mb
        print(
            f"[Epoch {epoch_name}] GPU Memory | "
            f"Allocated: {allocated_mb:.1f} MB | "
            f"Reserved: {reserved_mb:.1f} MB | "
            f"Peak: {peak_mb:.1f} MB | "
            f"Free (cached): {free_cached_mb:.1f} MB"
        )
        if self._use_wandb:
            wandb.log(
                {
                    "memory/allocated_mb": allocated_mb,
                    "memory/reserved_mb": reserved_mb,
                    "memory/peak_allocated_mb": peak_mb,
                    "memory/free_cached_mb": free_cached_mb,
                },
                step=epoch_name,
            )

    def get_total_loss(  # noqa: PLR0912, PLR0915
        self, model: BaseModel, epoch_name: int = -1
    ) -> torch.Tensor:  # epoch_name for rendering
        # Calculate the individual Losses
        loss_names = {}  # For logging purposes
        device = next(iter(model.parameters())).device
        weighted_total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_tensors = []  # To store individual loss tensors for further processing

        # TODO: add config for individual losses (e.g. flexicubes resolution)
        # wandb sweeping config:

        # 1. Calculate and sum up the active losses dynamically
        for loss_name, loss_instance in self.__active_losses.items():
            current_loss: torch.Tensor = loss_instance.get_loss(model, epoch_name)
            loss_tensors.append(current_loss)

            # Store for logging
            loss_names[loss_name] = current_loss.item()  # convert to float, gradient is lost

        if self.losses_config["general_info"]["ALM"] == "True":  # ALM weighting
            use_alm_aggressive = self.losses_config["general_info"].get("ALM_agressive") == "True"
            # DOCUMENTATION: `Losses and their weights` in Obsidian

            # o: maximise cruising speed (CSE)
            o: torch.Tensor
            if self.losses_config["general_info"]["CSE"] == "True":  # TODO: go through each separate loss and check
                o = -cruising_speed(
                    solar_power=loss_tensors[4],
                    drag_force_at_25ms=loss_tensors[0],
                    mass=210,
                    c_losses=20,
                    rolling_friction_coefficient=0.006,
                    max_iter=12,
                    tol=1e-4,
                )
            else:  # o: maximise solar power - drag losses
                o = torch.as_tensor(-1.0 * (loss_tensors[4] - loss_tensors[0]), device=device)

            # c: constraint loss (bb, eikonal, smoothness)
            # NOTE: we must NOT use torch.stack here — it creates a shared StackBackward node
            # that causes "backward through the graph a second time" in the per-component loop. Has to to with our memory issues
            constraint_tensors = [loss_tensors[1], loss_tensors[2], loss_tensors[3]]

            alm_thresholds = self.losses_config["general_info"]["alm_thresholds"]
            thresholds_tensor = torch.tensor(alm_thresholds, device=device) if alm_thresholds else None
            # weights_alm needs a stacked tensor, but detach first to avoid shared graph nodes.
            c_for_alm = torch.stack([ct.detach() for ct in constraint_tensors])
            self.lamdas, self.mus, self.nus = weights_alm(
                lamdas=self.lamdas,
                mus=self.mus,
                nus=self.nus,
                alphas=self.alphas,
                gamma=self.gamma,
                gamma_aggressive=self.gamma_aggressive,
                constrained_losses=c_for_alm,
                thresholds=thresholds_tensor,
                use_alm_aggressive=use_alm_aggressive,
            )

            # Build per-component weighted tensors for individual backward() calls.
            constraint_names = list(self.__active_losses.keys())[1:4]  # [bb, eikonal, smoothness]
            self._last_weighted_losses = {"alm_objective": o}
            for i, cname in enumerate(constraint_names):
                ci = constraint_tensors[i]  # direct reference, no StackBackward
                self._last_weighted_losses[f"alm_{cname}"] = self.lamdas[i].detach() * ci + 0.5 * self.mus[i].detach() * ci**2
            weighted_total_loss = sum(self._last_weighted_losses.values())  # type: ignore[assignment]

        else:  # manual weighting
            weights = weights_manual(
                loss_names=loss_names,
                losses_config=self.losses_config,
                loss_families=list(self.__active_losses.keys()),
                device=device,
            )
            weights = torch.as_tensor(weights, device=device, dtype=torch.float32)
            # Build per-component weighted tensors for individual backward() calls.
            # NOTE: we do NOT use torch.dot(torch.stack(...), weights) here because
            # torch.stack creates a shared StackBackward node across all loss graphs.
            self._last_weighted_losses = {
                name: loss_tensors[i] * weights[i] for i, name in enumerate(self.__active_losses.keys())
            }
            weighted_total_loss = sum(self._last_weighted_losses.values())  # type: ignore[assignment]

        # Add the total loss to the logging dictionary
        loss_names["total_loss"] = weighted_total_loss.item()

        # # 2. Logging them on WandB (not working as of now)
        if self._use_wandb is True:
            wandb.log(loss_names, step=epoch_name)
            if self.losses_config["general_info"]["ALM"] == "True":
                # log also the ALM parameters
                alm_params = {
                    "alm/lambda_bb": self.lamdas[0].item(),
                    "alm/lambda_eikonal": self.lamdas[1].item(),
                    "alm/lambda_smoothness": self.lamdas[2].item(),
                    "alm/mu_bb": self.mus[0].item(),
                    "alm/mu_eikonal": self.mus[1].item(),
                    "alm/mu_smoothness": self.mus[2].item(),
                    "alm/nu_bb": self.nus[0].item(),
                    "alm/nu_eikonal": self.nus[1].item(),
                    "alm/nu_smoothness": self.nus[2].item(),
                }
                wandb.log(alm_params, step=epoch_name)

            if self._use_renderer and (self._renderer_manager is not None):
                # Use optimized non-differentiable marching cubes extractor
                verts, faces = self._mesh_extractor.extract_mesh(
                    model=model,
                    isovalue=self._mesh_extraction_config.get("isovalue", 0.0),
                    batch_size=self._mesh_extraction_config.get("batch_size", 8192),
                )
                if self._mesh_statistics_enabled:
                    self._mesh_statistics.log_basic_mesh_stats(verts, faces, epoch_name)  # Log basic mesh stats to wandb
                    self._mesh_statistics.log_solar_area_stats(verts, faces, epoch_name)  # Log solar area related stats to wandb

                if self.__rendering_frequency is not None and epoch_name % self.__rendering_frequency == 0:
                    if self._use_neural_renderer and self._renderer_manager is not None:
                        # Use shared Manager to update all active rendering modes
                        try:
                            self._renderer_manager.update(
                                {
                                    "model": model,
                                    "iteration": epoch_name,
                                    "loss": weighted_total_loss.item(),
                                    "vertices": verts,
                                    "faces": faces,
                                }
                            )
                        except Exception as e:
                            print(f"Renderer wrapper update failed: {e}")

                    else:
                        # Use Clemens' CADVisualizer for rendering and logging
                        # Convert to numpy if tensors
                        if torch.is_tensor(verts):
                            verts_np = verts.detach().cpu().numpy()
                        if torch.is_tensor(faces):
                            faces_np = faces.detach().cpu().numpy()

                        # Create trimesh object and export to STL
                        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
                        stl_path = Path(f"subprojects/wp4_aeroshell_optimizer/v1/logs/mesh_epoch_{epoch_name}.stl")
                        stl_path.parent.mkdir(parents=True, exist_ok=True)
                        mesh.export(str(stl_path))

                        # Visualize the mesh using CADVisualizer
                        image_path = Path(f"subprojects/wp4_aeroshell_optimizer/v1/logs/render_epoch_{epoch_name}.png")
                        visualizer = CADVisualizer(stl_path=str(stl_path), out_image_path=str(image_path))
                        visualizer.render_until_blue_detected()

                        # Delete local files after logging to WandB
                        if stl_path.exists():
                            wandb.log({"mesh_3d": wandb.Object3D(str(stl_path))}, step=epoch_name)
                            stl_path.unlink()
                        if image_path.exists():
                            wandb.log({"render_image": wandb.Image(str(image_path))}, step=epoch_name)
                            image_path.unlink()
            else:
                print(f"Skipping rendering at epoch {epoch_name} due to rendering frequency setting.")
        # logg the gradients of each loss component
        for loss_instance in self.__active_losses.values():
            loss_instance.log_gradients(epoch_name)

        # Log GPU memory diagnostics (toggled by memory_diagnostics in logging.yaml)
        self._log_memory_diagnostics(epoch_name, device)

        return weighted_total_loss
