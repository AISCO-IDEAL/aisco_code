# crude estimator loss v0
import torch

from subprojects.wp1_cfd_ai.crude_estimator.crude_estimator_diff import (
    calculateCdProxyDifferentiable,
)
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.kaolin_conversions import (
    get_diff_mesh,
)


class CrudeEstimatorV0(ParentLoss):
    def _setup(self) -> None:
        return

    def _loss(self, model) -> torch.Tensor:
        vertices = get_diff_mesh(model, grid_res=64)
        crude_estimator_loss = calculateCdProxyDifferentiable(vertices)
        if crude_estimator_loss is None:
            # To satisfy mypy: Return a zero tensor that requires grad
            return torch.tensor(0.0, device=vertices.device, requires_grad=True)
        return crude_estimator_loss
