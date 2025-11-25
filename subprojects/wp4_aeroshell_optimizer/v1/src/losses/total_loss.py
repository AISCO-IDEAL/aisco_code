# just the loss class which calls other losses here


import torch
import wandb

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.bb_loss.bb_v1 import BbLoss1
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.bb_loss.bb_v2 import BbLoss2
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import (
    load_general_config,
    load_specific_config,
)
general_config = load_general_config()
        training_config = load_specific_config(
            v=general_config["general_info"]["version"],
            cfg_name=general_config["general_info"]["training_config_name"],
        )
        # load pointgrid params
        pointgrid_params = training_config["pointgrid_params"]
        dx, dy, dz, rand = (
            pointgrid_params.get("dx"),

class TotalLoss:
    def __init__(self, dev: str = "cuda") -> None:
        self.device = dev
        general_config = load_general_config()
        losses_config = load_specific_config(
            v=general_config["general_info"]["version"],
            cfg_name=general_config["active_configs"]["losses"],
        )
        self.weights = losses_config.get("losses")

        # setup all the individual Losses
        self.__bb_loss_1 = BbLoss1(self.device, self.weights["bb_loss_1"])
        self.__bb_loss_2 = BbLoss1(self.device, self.weights["bb_loss_2"])

    def get_total_loss(self, model, epoch_name: int = -1) -> torch.Tensor:
        # Calculate the individual Losses
        bb1 = self.__bb_loss_1.get_loss(model)
        bb2 = self.__bb_loss_2.get_loss(model)

        # Add them up
        total_loss = bb1 + bb2
        
        # Logging them on WandB
        
        wandb_config = load_specific_config(
            v=general_config["general_info"]["version"],
            cfg_name=general_config["active_configs"]["wandb"],
        )
        if wandb_config.get("general")["use_wandb"] is True:
            wandb.log(
                {"Bounding Box Loss 1:": bb1.item(), "Bounding Box Loss 2:": bb2.item()},
                step=epoch_name,
            )
        return total_loss
