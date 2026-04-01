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

        # Store diagnostics for DebugManager
        self._last_diagnostics = {
            "n_vertices": int(vertices.shape[0]) if vertices is not None else 0,
            "mesh_empty": vertices is None or vertices.numel() == 0,
            "estimator_returned_none": crude_estimator_loss is None,
        }

        if crude_estimator_loss is None:
            self._last_diagnostics["cd_proxy"] = 0.0
            # To satisfy mypy: Return a zero tensor that requires grad
            return torch.tensor(0.0, device=vertices.device, requires_grad=True)

        self._last_diagnostics["cd_proxy"] = crude_estimator_loss.item()
        self._last_diagnostics["cd_proxy_is_nan"] = bool(torch.isnan(crude_estimator_loss).any())
        self._last_diagnostics["cd_proxy_is_inf"] = bool(torch.isinf(crude_estimator_loss).any())
        return crude_estimator_loss
