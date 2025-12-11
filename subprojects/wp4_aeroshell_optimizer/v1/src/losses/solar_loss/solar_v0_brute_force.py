

import torch

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.brute_force_differentiator import BruteForceDifferentiator
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.solar_loss.solar_v0 import SolarV0


class SolarV0BruteForce(ParentLoss):
    def _setup(self) -> None:
        self.loss = SolarV0(device=self.device,
        losses_config:=self.losses_config,
        weight:=self.__weight,
        loss_name=self.loss_name)

        self.__differentiator = BruteForceDifferentiator(
            loss_function=self.loss._loss, device=self.device
        )
    def _loss(self, model) -> torch.Tensor:
        return self.__differentiator.loss(model)