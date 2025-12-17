from pathlib import Path

import torch
import trimesh
import wandb

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.weights import weights_alm, weights_manual
from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.dynamic_loading import load_class_from_path
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.kaolin_conversions import get_diff_mesh

# # Visualization imports (commented out for now)
# import trimesh # noqa: ERA001
# from subprojects.wp4_aeroshell_optimizer.v1.src.utils.kaolin_conversions import get_diff_mesh # noqa: ERA001
# from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser_utils import render_image  # noqa: ERA001
# from subprojects.wp4_aeroshell_optimizer.v1.src.utils.visualizer import CADVisualizer # noqa: ERA001
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.losses_utils import deep_update
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.visualizer import CADVisualizer
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visual_wrapper import Wrapper


class TotalLoss:
    def __init__(self, dev: str = "cuda", renderer_wrapper: Wrapper | None = None) -> None:
        # set up configs
        self.__general_config = load_config()  # default arg is the general config
        self._use_wandb = self.__general_config.get("wandb_general", {}).get("use_wandb", False)
        self._use_neural_renderer = self.__general_config.get("wandb_general", {}).get("use_neural_renderer", False)
        self.losses_config = load_config(
            v=self.__general_config["general_info"]["version"],
            cfg_name=self.__general_config["active_configs"]["losses"],
        )
        self.weights = self.losses_config.get("losses", {})  # get nested losses dict
        self.__active_losses: dict = {}
        self._renderer_initialized = False
        self._renderer_wrapper = renderer_wrapper

        project_root = Path.cwd()  # Correctly defines project root
        self.losses = self.losses_config.get("losses", {})  # get nested losses dict
        self.__rendering_frequency = self.__general_config.get("wandb_general", {}).get("rendering_frequency")

        # wandb init
        if self.__general_config["wandb_general"].get("use_wandb", False):
            run = wandb.init(
                # Set the wandb log directory
                dir="wp4_aeroshell_optimizer/data/wandb_logs",
                # Set the wandb entity where your project will be logged (generally your team name).
                entity=self.__general_config["wandb_general"].get("entity_name", "aisco"),
                # Set the wandb project where this run will be logged.
                project=self.__general_config["wandb_general"].get("project_name", "Aeroshell Optimiser with ALM and Sweeping"),
                # Track hyperparameters and run metadata.
                config=self.__general_config,
            )
            config = deep_update(self.__general_config, dict(wandb.config))
            run.config.update(config, allow_val_change=True)

        # ALM parameters init
        if self.losses_config["general_info"].get("ALM", False):
            # store lambda, mu, nu as 1D tensors
            self.lamda: float = (
                wandb.config.get("alm.lambda_init", 0.01)
                if wandb.run
                else self.losses_config["general_info"].get("initial_lamda", 0.1)
            )
            self.lamdas = torch.tensor(
                [[self.lamda], [self.lamda], [self.lamda]], device=dev
            )  # SWEEP: could be log-uniform in [10−5,10−1][10−5,10−1] per constraint
            self.mus = torch.tensor(
                [[1.0], [1.0], [1.0]], device=dev
            )  # ... a larger value such as η=1 is advisable to balance constraint enforcement and objective minimization effectively.:
            # NOTE: as seen in "Conditionally adaptive augmented Lagrangian method for physics-informed learning of forward and inverse problems using artificial neural networks"
            #       Authors: Qifeng Hua, Shamsulhaq Basira, Inanc Senocak
            #       https://arxiv.org/html/2508.15695v1
            self.nus = torch.tensor([[1.0], [1.0], [1.0]], device=dev)  # Initialize to 1.0 to prevent mu explosion at start
            self.alpha: float = (
                wandb.config.get("alm.alpha", 0.99) if wandb.run else self.losses_config["general_info"].get("alpha", 0.99)
            )
            self.alphas = torch.tensor(
                [[self.alpha], [self.alpha], [self.alpha]], device=dev
            )  # increase mu by this factor if constraint not me, default = 0.99 acc. to "Conditionally adaptive augmented Lagrangian method"
            # SWEEP: α ∈ {0.9,0.95,0.98,0.99}
            self.gamma: float = (
                wandb.config.get("alm.gamma", 0.01) if wandb.run else self.losses_config["general_info"].get("gamma", 0.01)
            )
            # NOTE: as seen "An Adaptive Augmented Lagrangian Method for Training Physics and Equality Constrained Artificial Neural Networks", Algortihm 3, ALM with adaptive penalty update (APU)
            # AUTHORS: Shamsulhaq Basir and Inanc Senocak
            #       https://arxiv.org/pdf/2306.04904
            #       SWEEP: γ ∈ log-uniform in [10−3,10e0]

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

    def get_total_loss(self, model: BaseModel, epoch_name: int = -1) -> torch.Tensor:  # epoch_name for rendering  # noqa: PLR0912
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
            loss_names[loss_name] = current_loss  # convert to float, gradient is lost

        if self.losses_config["general_info"]["ALM"] == "True":  # ALM weighting
            # documentation: `Losses and their weights` in Obsidian
            # o: primary loss (solar and drag) (objectives), for now just use basic weighting, later could be BOTorch or with a RaceTimeEstimator
            # c: constraint loss (bb, eikonal, smoothness)
            o = (loss_tensors[0] * 0.0) + (loss_tensors[4] * (-1.0))
            # NOTE: hardcoded weights for now, TODO: implement bayesian optimization here

            # 1D tenosrs for constraint losses
            c = torch.stack([loss_tensors[1], loss_tensors[2], loss_tensors[3]])

            alm_thresholds = self.losses_config["general_info"]["alm_thresholds"]
            thresholds_tensor = torch.tensor(alm_thresholds, device=c.device) if alm_thresholds else None
            # get lambda, mu each for constraint losses
            self.lamdas, self.mus, self.nus = weights_alm(
                lamdas=self.lamdas,
                mus=self.mus,
                nus=self.nus,
                alphas=self.alphas,
                gamma=self.gamma,
                constrained_losses=c,
                thresholds=thresholds_tensor,
            )

            weighted_total_loss = (
                o + torch.dot(self.lamdas.squeeze(), c) + 1 / 2 * torch.dot(self.mus.squeeze(), c**2)
            )  # dot products of weights and losses tensors
            # Sum all loss tensors in a differentiable way

        else:  # manual weighting
            weights = weights_manual(
                loss_names=loss_names,
                losses_config=self.losses_config,
                loss_families=list(self.__active_losses.keys()),
                device=device,
            )
            weights = torch.as_tensor(weights, device=device, requires_grad=True, dtype=torch.float32)
            # Stack losses into a single tensor and multiply by weights
            # We convert weights to a tensor on the correct device to allow element-wise multiplication
            weighted_total_loss = torch.dot(torch.stack(loss_tensors), weights)

        # # Add the total loss to the logging dictionary
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
            # Initialization of per-model renderer is deprecated; wrapper is preferred.
            # Ensure a wrapper exists if neural rendering is requested.
            if self._use_neural_renderer and self._renderer_wrapper is None:
                # Lazily construct a wrapper with defaults from config
                try:
                    self._renderer_wrapper = Wrapper(verbose=False)
                except Exception as e:
                    print(f"Failed to initialize Visual Wrapper: {e}")

            if self.__rendering_frequency is not None and epoch_name % self.__rendering_frequency == 0:
                verts, faces = get_diff_mesh(model, grid_res=64, device=device)

                if self._use_neural_renderer and self._renderer_wrapper is not None:
                    # Use shared Wrapper to update all active rendering modes
                    try:
                        self._renderer_wrapper.update(
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
                        verts = verts.detach().cpu().numpy()
                    if torch.is_tensor(faces):
                        faces = faces.detach().cpu().numpy()

                    # Create trimesh object and export to STL
                    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
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

        return weighted_total_loss
