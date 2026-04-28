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
        bbox_lo: list[float] | None = None,
        bbox_hi: list[float] | None = None,
        norm_order: int = 2,
        neighbor_agg_fn: str = "min",
    ) -> Tensor:
        """Compute field-diversity loss over the given latent population.

        Args:
            model: SDF network with latent-concat support.
            Z: Latent population tensor of shape (N, z_dim).
            num_points: Number of random sample points.
            bbox_lo: Per-axis lower bounds [x_lo, y_lo, z_lo] in mm.
                Defaults to outer_prism x/y/z lower bounds.
            bbox_hi: Per-axis upper bounds [x_hi, y_hi, z_hi] in mm.
                Defaults to outer_prism x/y/z upper bounds.
            norm_order: Lp norm order for field distance.
            neighbor_agg_fn: Aggregation over neighbours ("min" or "mean").

        Returns:
            Scalar diversity loss tensor.
        """
        if not isinstance(model, SupportsLatentConcat):
            return torch.zeros((), device=self._device, dtype=torch.float32)
        if Z is None:
            return torch.zeros((), device=self._device, dtype=torch.float32)
        if Z.shape[0] <= 1:
            return torch.zeros((), device=Z.device, dtype=torch.float32)

        # Per-axis non-cubic sampling matching the outer prism (default = outer_prism bounds in mm).
        # Previous implementation used a scalar `bbox_size` creating a cube, which wasted ~75%
        # of samples outside the actual car volume (Y and Z axes are much narrower than X).
        _lo = bbox_lo if bbox_lo is not None else [-2900.0, -1150.0, -100.0]
        _hi = bbox_hi if bbox_hi is not None else [2900.0, 1150.0, 1650.0]
        lo = torch.tensor(_lo, device=Z.device, dtype=torch.float32)
        hi = torch.tensor(_hi, device=Z.device, dtype=torch.float32)
        points = lo + torch.rand(num_points, 3, device=Z.device) * (hi - lo)
        field_values: list[Tensor] = []
        for i in range(Z.shape[0]):
            model.set_latent_vector(Z[i])
            y_i = model(points)
            if isinstance(y_i, tuple):
                y_i = y_i[0]
            field_values.append(y_i.squeeze(-1))

        Y = torch.stack(field_values, dim=0)
        return field_diversity_loss(Y, norm_order=norm_order, neighbor_agg_fn=neighbor_agg_fn)
