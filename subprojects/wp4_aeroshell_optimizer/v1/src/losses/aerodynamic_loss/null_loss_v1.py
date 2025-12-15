# this is the dummy null loss that returns zero always
import torch

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss


class NullLossV1(ParentLoss):
    def _setup(self) -> None:
        return

    def _loss(self, model) -> torch.Tensor:
        return 0
