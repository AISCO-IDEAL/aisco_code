from pathlib import Path

import torch
import wandb

from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.dynamic_loading import load_class_from_path
from subprojects.wp4_aeroshell_optimizer.v1.src.weights import weights_alm, weights_manual

# # Visualization imports (commented out for now)
# import trimesh # noqa: ERA001
# from subprojects.wp4_aeroshell_optimizer.v1.src.utils.kaolin_conversions import get_diff_mesh # noqa: ERA001
# from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser_utils import render_image  # noqa: ERA001
# from subprojects.wp4_aeroshell_optimizer.v1.src.utils.visualizer import CADVisualizer # noqa: ERA001


class TotalLoss:
    def __init__(self, dev: str = "cuda") -> None:
        # set up configs
        self.__general_config = load_config()
        self._use_wandb = self.__general_config.get("wandb_general", {}).get("use_wandb", False)
        self._use_neural_renderer = self.__general_config.get("wandb_general", {}).get("use_neural_renderer", False)
        self.losses_config = load_config(
            v=self.__general_config["general_info"]["version"],
            cfg_name=self.__general_config["active_configs"]["losses"],
        )
        self.weights = self.losses_config.get("losses", {})  # get nested losses dict
        self.__active_losses: dict = {}
        self._renderer_initialized = False

        project_root = Path.cwd()  # Correctly defines project root
        self.losses = self.losses_config.get("losses", {})  # get nested losses dict
        self.__rendering_frequency = self.__general_config.get("wandb_general", {}).get("rendering_frequency")

        # ALM parameters init
        if self.losses_config["general_info"].get("ALM", False):
            # store lambda, mu, nu as 1D tensors
            self.lamdas = torch.tensor([[0.001], [0.001], [0.001]], device=dev)  # tricky, needs to be sweeped over
            self.mus = torch.tensor([[1.0], [1.0], [1.0]], device=dev)  # standard literature values:
            # TODO: # add link for a paper here.
            self.nus = torch.tensor([[0.0], [0.0], [0.0]], device=dev)  # standard literature values
            self.alphas = torch.tensor([[0.99], [0.99], [0.99]], device=dev)  # increase mu by this factor if constraint not me
            self.gamma: float = 0.1  # global learning rate for mu updates

        # wandb init
        if self.__general_config["wandb_general"].get("use_wandb", False):
            wandb.init(
                # Set the wandb log directory
                dir="wp4_aeroshell_optimizer/data/wandb_logs",
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="aisco",
                # Set the wandb project where this run will be logged.
                project="Aeroshell Optimiser",
                # Track hyperparameters and run metadata.
                config=self.__general_config,
            )

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

    def get_total_loss(self, model: BaseModel, epoch_name: int = -1) -> torch.Tensor:  # noqa: ARG002 # epoch_name for rendering
        # Calculate the individual Losses
        loss_names = {}  # For logging purposes
        device = next(iter(model.parameters())).device
        weighted_total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_tensors = []  # To store individual loss tensors for further processing

        # TODO: add config for individual losses (e.g. flexicubes resolution)
        # wandb sweeping config:

        # 1. Calculate and sum up the active losses dynamically
        for loss_name, loss_instance in self.__active_losses.items():
            current_loss: torch.Tensor = loss_instance.get_loss(model)
            loss_tensors.append(current_loss)

            # Store for logging
            loss_names[loss_name] = current_loss.item()  # convert to float, gradient is lost

        if self.losses_config["general_info"]["ALM"] == "True":  # ALM weighting
            # documentation: `Losses and their weights` in Obsidian
            # o: primary loss (solar and drag) (objectives), for now just use basic weighting, later could be BOTorch or with a RaceTimeEstimator
            # c: constraint loss (bb, eikonal, smoothness)
            o = torch.tensor((loss_tensors[0] * 1.0 + loss_tensors[4] * (-1.0)), device=device, requires_grad=True)
            # NOTE: hardcoded weights for now, TODO: implement bayesian optimization here

            # 1D tenosrs for constraint losses
            c = torch.tensor([loss_tensors[1], loss_tensors[2], loss_tensors[3]], device=device, requires_grad=True)

            # get lambda, mu each for constraint losses
            self.lamdas, self.mus, self.nus = weights_alm(
                lamdas=self.lamdas, mus=self.mus, nus=self.nus, alphas=self.alphas, gamma=self.gamma, constrained_losses=c
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
        loss_names["total_loss"] = weighted_total_loss.squeeze()

        return weighted_total_loss
