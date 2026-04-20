from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from subprojects.wp4_aeroshell_optimizer.v2.src.GINN.diversity_loss import field_diversity_loss
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat, SupportsModelContract


class FieldDiversityV0:
    """Field-based diversity over a latent population.

    Computes SDF outputs for each latent on shared random sample points and applies
    nearest-neighbor aggregation in field space.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        loss_name: str = "field_diversity_v0",
        losses_config: dict[str, Any] | None = None,
    ) -> None:
        self._device: torch.device = torch.device(device) if isinstance(device, str) else device
        self.loss_name = loss_name
        self.losses_config = losses_config or {}

    def get_loss(
        self,
        model: SupportsModelContract,
        Z: Tensor | None = None,
        num_points: int = 2048,
        bbox_size: float = 10.0,
        norm_order: int = 2,
        neighbor_agg_fn: str = "min",
    ) -> Tensor:
        if not isinstance(model, SupportsLatentConcat):
            return torch.zeros((), device=self._device, dtype=torch.float32)
        if Z is None:
            return torch.zeros((), device=self._device, dtype=torch.float32)
        if Z.shape[0] <= 1:
            return torch.zeros((), device=Z.device, dtype=torch.float32)

        points = (torch.rand(num_points, 3, device=Z.device) - 0.5) * bbox_size
        field_values: list[Tensor] = []
        for i in range(Z.shape[0]):
            model.set_latent_vector(Z[i])
            y_i = model(points)
            if isinstance(y_i, tuple):
                y_i = y_i[0]
            field_values.append(y_i.squeeze(-1))

        Y = torch.stack(field_values, dim=0)
        return field_diversity_loss(Y, norm_order=norm_order, neighbor_agg_fn=neighbor_agg_fn)
