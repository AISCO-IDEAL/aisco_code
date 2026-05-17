"""Hybrid diversity: masked volumetric + masked surface (A + D).

Computes both the masked volumetric diversity (`FieldDiversityMaskedV0`) and the
masked surface diversity (`FieldDiversitySurfaceV0`), then returns a weighted
sum. Surface diversity is the dominant signal; the small volumetric contribution
keeps the interior SDF from being completely free, which helps the marching-cubes
extraction stay stable between refreshes.

Both sub-losses are passed `max_div=None` internally; the optional clamp is
applied once on the combined loss instead, so the threshold is interpreted
against the total magnitude the caller actually sees.

Config block (in losses_v0_config.yaml):

    field_diversity_hybrid_v0_config:
      w_vol: 0.1
      w_surf: 1.0
      max_div: null
      # The sub-loss configs are read from their own blocks:
      #   field_diversity_masked_v0_config:  ...
      #   field_diversity_surface_v0_config: ...
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.diversity_loss.field_diversity_masked_v0 import (
    FieldDiversityMaskedV0,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.diversity_loss.field_diversity_surface_v0 import (
    FieldDiversitySurfaceV0,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat, SupportsModelContract


class FieldDiversityHybridV0:
    def __init__(
        self,
        device: str | torch.device = "cuda",
        loss_name: str = "field_diversity_hybrid_v0",
        losses_config: dict[str, Any] | None = None,
    ) -> None:
        self._device: torch.device = torch.device(device) if isinstance(device, str) else device
        self.loss_name = loss_name
        self.losses_config = losses_config or {}

        cfg = self.losses_config.get("field_diversity_hybrid_v0_config", {}) or {}
        self._w_vol: float = float(cfg.get("w_vol", 0.1))
        self._w_surf: float = float(cfg.get("w_surf", 1.0))
        self._max_div: float | None = cfg.get("max_div", None)
        self._max_div = None if self._max_div is None else float(self._max_div)

        # Build sub-losses with their max_div suppressed — clamp is applied here on the sum.
        vol_cfg = dict(self.losses_config)
        vol_cfg["field_diversity_masked_v0_config"] = {
            **(self.losses_config.get("field_diversity_masked_v0_config", {}) or {}),
            "max_div": None,
        }
        surf_cfg = dict(self.losses_config)
        surf_cfg["field_diversity_surface_v0_config"] = {
            **(self.losses_config.get("field_diversity_surface_v0_config", {}) or {}),
            "max_div": None,
        }
        self._vol = FieldDiversityMaskedV0(device=self._device, loss_name=f"{loss_name}__vol", losses_config=vol_cfg)
        self._surf = FieldDiversitySurfaceV0(device=self._device, loss_name=f"{loss_name}__surf", losses_config=surf_cfg)

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
        if not isinstance(model, SupportsLatentConcat) or Z is None or Z.shape[0] <= 1:
            return torch.zeros((), device=self._device, dtype=torch.float32)

        vol = (
            self._vol.get_loss(model, Z, num_points, bbox_lo, bbox_hi, norm_order, neighbor_agg_fn)
            if self._w_vol != 0.0
            else torch.zeros((), device=Z.device, dtype=torch.float32)
        )
        surf = (
            self._surf.get_loss(model, Z, num_points, bbox_lo, bbox_hi, norm_order, neighbor_agg_fn)
            if self._w_surf != 0.0
            else torch.zeros((), device=Z.device, dtype=torch.float32)
        )
        loss = self._w_vol * vol + self._w_surf * surf
        if self._max_div is not None:
            loss = torch.clamp(loss - self._max_div, min=0.0)
        return loss

    @property
    def vol(self) -> FieldDiversityMaskedV0:
        return self._vol

    @property
    def surf(self) -> FieldDiversitySurfaceV0:
        return self._surf
