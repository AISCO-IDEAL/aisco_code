# this is the dummy null loss that returns zero always
import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss


class NullLossSolarV1(ParentLoss):
    def _setup(self) -> None:
        return

    def _loss(self, model) -> torch.Tensor:  # noqa: ARG002
        return torch.tensor(700.0, device=self.device) # defaults to 700 Watts
