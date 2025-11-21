# just the loss class which calls other losses here


import torch
import wandb

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.bb_loss.bb_v1 import BbLoss1

class TotalLoss:
    def __init__(self, dev: str = "cuda")->None:
        self.device = dev
        self.weights = load_config().get("losses")

        # setup all the individual Losses
        self.__bb_loss_1 = BbLoss1(self.device, self.weights["bb_loss_1"])

    def get_total_loss(self, model, epoch_name:int=-1)->torch.Tensor:
        # Calculate the individual Losses
        bb1 = self.__bb_loss_1.get_loss(model)

        # Add them up
        total_loss = bb1

        # Logg them on WandB
        wandb.log(
            {
                "Bounding Box Loss 1:", bb1.item(),
            },
            step=epoch_name,
        )

        return total_loss