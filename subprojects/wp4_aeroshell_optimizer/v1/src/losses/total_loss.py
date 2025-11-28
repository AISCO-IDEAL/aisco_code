# just the loss class which calls other losses here


import torch
import wandb

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.bb_loss.bb_v1 import BbLoss1
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.bb_loss.bb_v2 import BbLoss2
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.solar_loss.solar_v0 import (
    SolarLoss,
)
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.aerodynamic_loss.crude_estimator_v0 import (
    CrudeEstimator,
)
from sub

class TotalLoss:
    def __init__(self, dev: str = "cuda") -> None:
        # setup all the individual Losses
        self.__bb_loss_1 = BbLoss1()
        self.__bb_loss_2 = BbLoss2()
        self.__solar_loss = SolarLoss()
        self.__crude_estimator_loss = CrudeEstimator()
        #        self.__frontal_area_loss = 

    def get_total_loss(self, model, epoch_name: int = -1) -> torch.Tensor:
        # Calculate the individual Losses
        # TODO: add config for individual losses (e.g. flexicubes resolution)
        bb1 = self.__bb_loss_1.get_loss(model)
        bb2 = self.__bb_loss_2.get_loss(model)
        solar = self.__solar_loss.get_loss(model)
        aero = self.__

        # Add them up
        total_loss = bb1 + bb2 + solar

        # Logging them on WandB

        if self.general_config["wandb_general"]["use_wandb"] is True:
            wandb.log(
                {
                    "Bounding Box Loss 1:": bb1.item(),
                    "Bounding Box Loss 2:": bb2.item(),
                    "Solar Loss:": solar.item(),
                    "Total Loss:": total_loss.item(),
                },
                step=epoch_name,
            )
        return total_loss
