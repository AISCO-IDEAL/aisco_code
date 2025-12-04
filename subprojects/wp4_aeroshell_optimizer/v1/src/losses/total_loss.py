# just the loss class which calls other losses here


import torch
import wandb
from pathlib import Path

# NOTE: You can remove all hardcoded loss imports (like BbLoss1, SolarLoss, etc.)
# because you are now dynamically loading them based on the config path.

from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import (
    load_general_config,
    load_specific_config,
)
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.dynamic_loading import load_class_from_path


class TotalLoss:
    def __init__(self, dev: str = "cuda") -> None:
        # set up configs
        self.__general_config = load_general_config()
        self.losses_config = load_specific_config(
            v=self.__general_config["general_info"]["version"],
            cfg_name=self.__general_config["active_configs"]["losses"],
        )
        self.weights = self.losses_config.get("losses", {})  # get nested losses dict
        self.__active_losses = {}

        project_root = Path.cwd()  # Correctly defines project root

        # category: aero, ... | losses: frontal_area_v0, ...
        for category, losses in self.weights.items():
            if category == "total":
                continue

            # loss_name: frontal_area_v0, ... | compute: bool | weight: float | path: str
            for loss_name, (compute, weight, path) in losses.items():
                if compute:
                    # Dynamically load the loss class
                    # 1. Correctly construct the absolute path from the relative path in config
                    absolute_path = project_root / Path(path)

                    class_name = Path(path).stem.title().replace("_", "")

                    # 2. Call load_class_from_path with the absolute path string
                    loss_class = load_class_from_path(path=str(absolute_path), class_name=class_name)
                    loss_instance = loss_class(
                        device=dev,
                        loss_name=loss_name,
                        losses_config=self.losses_config,
                    )
                    self.__active_losses[loss_name] = loss_instance

    def get_total_loss(self, model, epoch_name: int = -1) -> torch.Tensor:
        # Calculate the individual Losses
        individual_losses = {}
        # Initialize total_loss as a sum of all loss tensors (not in-place)
        loss_tensors = []
        # TODO: add config for individual losses (e.g. flexicubes resolution)

        # 1. Calculate and sum up the active losses dynamically
        for loss_name, loss_instance in self.__active_losses.items():
            # Get the loss value (which should be a weighted value if done inside the loss)
            # NOTE: This assumes the loss instances have a .get_loss(model) method.
            current_loss: torch.Tensor = loss_instance.get_loss(model)
            loss_tensors.append(current_loss)

            # Store for logging
            individual_losses[loss_name] = current_loss.item()

        # Sum all loss tensors in a differentiable way
        total_loss = torch.sum(torch.stack(loss_tensors))

        # Add the total loss to the logging dictionary
        individual_losses["total_loss"] = total_loss.item()

        # 2. Logging them on WandB (not working as of now)
        if self.__general_config.get("wandb_general", {}).get("use_wandb") is True:
            # Only reinit if necessary, often it's better to manage the run externally
            # but sticking to your original logic for reinit=True.
            wandb.init(
                project=self.__general_config["wandb_general"]["project_name"],
                entity=self.__general_config["wandb_general"]["entity_name"],
            )

            # Log the collected individual losses
            wandb.log(individual_losses, step=epoch_name)

        return total_loss
