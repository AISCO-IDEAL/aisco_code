"""Volumetric diversity with hard-constraint masking (Option A).

Identical sampling-and-aggregation strategy as `FieldDiversityV0`, but the
uniform prism samples are filtered through `interface_mask.mask_points_by_distance`
before evaluation: any point within the configured cutoff of a wheel-contact
patch or inner-prism (seat) surface point is dropped. Dropped points are
replenished by resampling so the diversity gradient sees a constant point
budget regardless of how the constraint regions move.

Config block (in losses_v0_config.yaml):

    field_diversity_masked_v0_config:
      wheel_cutoff_mm: 40.0
      bb_inner_cutoff_mm: 15.0
      wheel_sampling_density: 10
      bb_inner_target_point_count: 20000
      max_div: null            # null disables the GINN clamp; e.g. -0.1 enables
      max_resample_iters: 8    # safety cap on the top-up loop

The wheel/bb_inner cutoffs are expressed in SDF coordinates (millimetres).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from subprojects.wp2_assembler.version4.utils.reg_imp import get_outer_prism_bounds_mm
from subprojects.wp4_aeroshell_optimizer.v2.src.GINN.diversity_loss import field_diversity_loss
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.diversity_loss.interface_mask import (
    InterfaceSource,
    build_default_sources,
    mask_points_by_distance,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat, SupportsModelContract


class FieldDiversityMaskedV0:
    """Field-based diversity with assembly-sourced interface masking."""

    def __init__(
        self,
        device: str | torch.device = "cuda",
        loss_name: str = "field_diversity_masked_v0",
        losses_config: dict[str, Any] | None = None,
    ) -> None:
        self._device: torch.device = torch.device(device) if isinstance(device, str) else device
        self.loss_name = loss_name
        self.losses_config = losses_config or {}

        cfg = self.losses_config.get("field_diversity_masked_v0_config", {}) or {}
        self._wheel_cutoff_mm: float = float(cfg.get("wheel_cutoff_mm", 40.0))
        self._bb_inner_cutoff_mm: float = float(cfg.get("bb_inner_cutoff_mm", 15.0))
        self._wheel_sampling_density: int = int(cfg.get("wheel_sampling_density", 10))
        self._bb_inner_target_point_count: int = int(cfg.get("bb_inner_target_point_count", 20000))
        self._max_div: float | None = cfg.get("max_div", None)
        self._max_div = None if self._max_div is None else float(self._max_div)
        self._max_resample_iters: int = int(cfg.get("max_resample_iters", 8))

        assembly_id = int(self.losses_config.get("bb_config", {}).get("assembly_id", 1))
        wheel_cfg = self.losses_config.get("wheel_loss_config", {}) or {}
        wheel_manual_z_shift = float(wheel_cfg.get("manual_z_shift", 0.0))

        self._sources: list[InterfaceSource] = build_default_sources(
            assembly_id=assembly_id,
            wheel_cutoff_mm=self._wheel_cutoff_mm,
            bb_inner_cutoff_mm=self._bb_inner_cutoff_mm,
            wheel_sampling_density=self._wheel_sampling_density,
            wheel_manual_z_shift=wheel_manual_z_shift,
            bb_inner_target_point_count=self._bb_inner_target_point_count,
            device=self._device,
        )

    def _sample_masked_points(
        self, num_points: int, lo: Tensor, hi: Tensor, device: torch.device
    ) -> tuple[Tensor, dict[str, int]]:
        """Top up to `num_points` by repeated rejection sampling against the masks."""
        kept_parts: list[Tensor] = []
        kept_total = 0
        drop_counts: dict[str, int] = {src.name: 0 for src in self._sources}
        for _ in range(self._max_resample_iters):
            needed = num_points - kept_total
            if needed <= 0:
                break
            batch = lo + torch.rand(needed, 3, device=device) * (hi - lo)
            keep_mask, tag = mask_points_by_distance(batch, self._sources)
            kept_parts.append(batch[keep_mask])
            for src in self._sources:
                drop_counts[src.name] += int((tag == src.tag).sum().item())
            kept_total += int(keep_mask.sum().item())
        points = torch.cat(kept_parts, dim=0)[:num_points] if kept_parts else torch.zeros((0, 3), device=device)
        return points, drop_counts

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
        if not isinstance(model, SupportsLatentConcat):
            return torch.zeros((), device=self._device, dtype=torch.float32)
        if Z is None:
            return torch.zeros((), device=self._device, dtype=torch.float32)
        if Z.shape[0] <= 1:
            return torch.zeros((), device=Z.device, dtype=torch.float32)

        if bbox_lo is None or bbox_hi is None:
            _prism = get_outer_prism_bounds_mm()
            _lo = bbox_lo if bbox_lo is not None else [_prism["x"][0], _prism["y"][0], _prism["z"][0]]
            _hi = bbox_hi if bbox_hi is not None else [_prism["x"][1], _prism["y"][1], _prism["z"][1]]
        else:
            _lo, _hi = bbox_lo, bbox_hi
        lo = torch.tensor(_lo, device=Z.device, dtype=torch.float32)
        hi = torch.tensor(_hi, device=Z.device, dtype=torch.float32)

        points, _drops = self._sample_masked_points(num_points, lo, hi, device=Z.device)
        if points.shape[0] == 0:
            return torch.zeros((), device=Z.device, dtype=torch.float32)

        field_values: list[Tensor] = []
        for i in range(Z.shape[0]):
            model.set_latent_vector(Z[i])
            y_i = model(points)
            if isinstance(y_i, tuple):
                y_i = y_i[0]
            field_values.append(y_i.squeeze(-1))

        Y = torch.stack(field_values, dim=0)
        return field_diversity_loss(Y, norm_order=norm_order, neighbor_agg_fn=neighbor_agg_fn, max_div=self._max_div)

    def debug_sample(self, num_points: int, device: torch.device | None = None) -> dict[str, Tensor]:
        """Return raw + masked sample sets for the visualizer. Does not touch the model."""
        target_device = torch.device(device) if device is not None else self._device
        _prism = get_outer_prism_bounds_mm()
        lo = torch.tensor([_prism["x"][0], _prism["y"][0], _prism["z"][0]], device=target_device, dtype=torch.float32)
        hi = torch.tensor([_prism["x"][1], _prism["y"][1], _prism["z"][1]], device=target_device, dtype=torch.float32)
        raw = lo + torch.rand(num_points, 3, device=target_device) * (hi - lo)
        keep_mask, tag = mask_points_by_distance(raw, self._sources)
        kept, _ = self._sample_masked_points(num_points, lo, hi, device=target_device)
        return {
            "raw": raw,
            "keep_mask": keep_mask,
            "tag": tag,
            "kept_topped_up": kept,
        }

    @property
    def sources(self) -> list[InterfaceSource]:
        return self._sources
