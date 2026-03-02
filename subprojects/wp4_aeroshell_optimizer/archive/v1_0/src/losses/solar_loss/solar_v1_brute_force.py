import torch

from subprojects.wp3_solar_estimator.v1 import n_diff_solar_loss
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.brute_force_differentiator import BruteForceDifferentiator
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss


class SolarV1BruteForce(ParentLoss):
    def _setup(self) -> None:
        self.__differentiator = BruteForceDifferentiator(loss_function=n_diff_solar_loss, device=self.device)

    def _loss(self, model) -> torch.Tensor:
        return self.__differentiator.loss(model)
