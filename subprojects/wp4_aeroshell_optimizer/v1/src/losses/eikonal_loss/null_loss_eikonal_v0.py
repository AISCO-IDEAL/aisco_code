# eikonal v0
import torch

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss


class NullLossEikonalV0(ParentLoss):
    def _setup(self) -> None:
        pass

    def _loss(self, _model) -> torch.Tensor:
        return torch.tensor(0.0, device=self.device)
