# this is the dummy null loss that returns zero always
import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss


class NullLossAeroV1(ParentLoss):
    def _setup(self) -> None:
        return

    def _loss(self, _model) -> torch.Tensor:
        return torch.tensor(50.0, device=self.device) # defaults to 50 Newtons
