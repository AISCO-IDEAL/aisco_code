"""Surface-only diversity with hard-constraint masking (Option D, GINN-style).

For each latent in the population, extracts the zero level-set of the SDF via
marching cubes (reusing `MeshSDFExtractor`), subsamples its vertices, drops any
vertex within the interface cutoffs, and computes diversity on the SDF
re-evaluated at those surface points. Marching-cubes vertex positions are
themselves non-differentiable w.r.t. the network, but the diversity signal comes
from `model(p_surface)` evaluated through autograd at those (frozen) positions —
same trick as GINN's domain-on-surface variant.

Surface extraction is expensive, so the per-latent point sets are cached and
refreshed every `surface_resample_cadence` calls.

Config block (in losses_v0_config.yaml):

    field_diversity_surface_v0_config:
      wheel_cutoff_mm: 40.0
      bb_inner_cutoff_mm: 15.0
      wheel_sampling_density: 10
      bb_inner_target_point_count: 20000
      surface_resample_cadence: 50   # extract every N get_loss calls
      surface_subsample_n: 4096      # subsample MC vertices to this many per shape
      mc_resolution: 64              # marching-cubes grid resolution
      mc_bbox_scale: 1.0             # multiplier on the outer prism bbox
      max_div: null
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from subprojects.wp2_assembler.version4.utils.reg_imp import get_outer_prism_bounds_mm, scale_outer_prism
from subprojects.wp4_aeroshell_optimizer.v2.src.GINN.diversity_loss import field_diversity_loss
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.diversity_loss.interface_mask import (
    InterfaceSource,
    build_default_sources,
    mask_points_by_distance,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat, SupportsModelContract
from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor
from subprojects.wp5.unified_mesh_extraction.utils import MeshExtractionConfig, load_mesh_extraction_config


class FieldDiversitySurfaceV0:
    """Field-based diversity computed on per-shape surface points (masked)."""

    def __init__(
        self,
        device: str | torch.device = "cuda",
        loss_name: str = "field_diversity_surface_v0",
        losses_config: dict[str, Any] | None = None,
    ) -> None:
        self._device: torch.device = torch.device(device) if isinstance(device, str) else device
        self.loss_name = loss_name
        self.losses_config = losses_config or {}

        cfg = self.losses_config.get("field_diversity_surface_v0_config", {}) or {}
        self._wheel_cutoff_mm: float = float(cfg.get("wheel_cutoff_mm", 40.0))
        self._bb_inner_cutoff_mm: float = float(cfg.get("bb_inner_cutoff_mm", 15.0))
        self._wheel_sampling_density: int = int(cfg.get("wheel_sampling_density", 10))
        self._bb_inner_target_point_count: int = int(cfg.get("bb_inner_target_point_count", 20000))
        self._surface_resample_cadence: int = max(1, int(cfg.get("surface_resample_cadence", 50)))
        self._surface_subsample_n: int = int(cfg.get("surface_subsample_n", 4096))
        self._mc_resolution: int = int(cfg.get("mc_resolution", 64))
        self._mc_bbox_scale: float = float(cfg.get("mc_bbox_scale", 1.0))
        self._max_div: float | None = cfg.get("max_div", None)
        self._max_div = None if self._max_div is None else float(self._max_div)
        self._warmup_epochs: int = int(cfg.get("warmup_epochs", 0))

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
        self._extractor: MeshSDFExtractor = self._build_extractor()

        # Cached masked surface points per population slot; refreshed on cadence.
        self._cached_points: list[Tensor] | None = None
        self._call_counter: int = 0

    def _build_extractor(self) -> MeshSDFExtractor:
        prism = scale_outer_prism(get_outer_prism_bounds_mm(), self._mc_bbox_scale)
        bbox = [
            [prism["x"][0], prism["x"][1]],
            [prism["y"][0], prism["y"][1]],
            [prism["z"][0], prism["z"][1]],
        ]
        base_cfg = load_mesh_extraction_config()
        cfg = MeshExtractionConfig(
            device=str(self._device),
            isovalue=base_cfg.isovalue,
            base_resolution=self._mc_resolution,
            subdivision_factor=base_cfg.subdivision_factor,
            max_refine_levels=base_cfg.max_refine_levels,
            batch_size=base_cfg.batch_size,
            bbox=bbox,
            target_num_vertices=None,
            vertex_count_tolerance=base_cfg.vertex_count_tolerance,
        )
        return MeshSDFExtractor(config=cfg)

    def _refresh_cache(self, model: SupportsLatentConcat, Z: Tensor) -> None:
        """Re-extract masked surface points for every latent in Z."""
        points_per_shape: list[Tensor] = []
        for i in range(Z.shape[0]):
            model.set_latent_vector(Z[i])
            with torch.no_grad():
                verts, _faces = self._extractor.extract_mesh(model)
            if verts.numel() == 0:
                points_per_shape.append(torch.zeros((0, 3), device=Z.device))
                continue
            verts = verts.to(Z.device, dtype=torch.float32)
            if verts.shape[0] > self._surface_subsample_n:
                idx = torch.randperm(verts.shape[0], device=verts.device)[: self._surface_subsample_n]
                verts = verts[idx]
            keep_mask, _tag = mask_points_by_distance(verts, self._sources)
            points_per_shape.append(verts[keep_mask].detach())
        self._cached_points = points_per_shape

    def get_loss(
        self,
        model: SupportsModelContract,
        Z: Tensor | None = None,
        num_points: int = 2048,  # noqa: ARG002 — kept for manager signature compatibility
        bbox_lo: list[float] | None = None,  # noqa: ARG002
        bbox_hi: list[float] | None = None,  # noqa: ARG002
        norm_order: int = 2,
        neighbor_agg_fn: str = "min",
        epoch: int = -1,
        leinster_temp: float | None = None,
        leinster_q: float | None = None,
    ) -> Tensor | None:
        # Warmup: while inside the warmup window, return None so the manager skips the
        # diversity backward entirely (constraints shape the population first).
        if 0 <= epoch < self._warmup_epochs:
            return None
        if not isinstance(model, SupportsLatentConcat):
            return torch.zeros((), device=self._device, dtype=torch.float32)
        if Z is None:
            return torch.zeros((), device=self._device, dtype=torch.float32)
        if Z.shape[0] <= 1:
            return torch.zeros((), device=Z.device, dtype=torch.float32)

        needs_refresh = (
            self._cached_points is None
            or len(self._cached_points) != Z.shape[0]
            or (self._call_counter % self._surface_resample_cadence == 0)
        )
        if needs_refresh:
            self._refresh_cache(model, Z)
        self._call_counter += 1

        # Diversity needs a common point set across shapes — concatenate per-shape
        # surface points and evaluate every shape there. Each shape's own surface
        # is included; SDFs of other shapes at those points carry the divergence
        # signal.
        cached = self._cached_points or []
        common_pts = torch.cat([p for p in cached if p.shape[0] > 0], dim=0) if cached else torch.zeros((0, 3), device=Z.device)
        if common_pts.shape[0] > self._surface_subsample_n:
            idx = torch.randperm(common_pts.shape[0], device=common_pts.device)[: self._surface_subsample_n]
            common_pts = common_pts[idx]
        if common_pts.shape[0] == 0:
            return torch.zeros((), device=Z.device, dtype=torch.float32)

        field_values: list[Tensor] = []
        for i in range(Z.shape[0]):
            model.set_latent_vector(Z[i])
            y_i = model(common_pts)
            if isinstance(y_i, tuple):
                y_i = y_i[0]
            field_values.append(y_i.squeeze(-1))

        Y = torch.stack(field_values, dim=0)
        return field_diversity_loss(
            Y,
            norm_order=norm_order,
            neighbor_agg_fn=neighbor_agg_fn,
            max_div=self._max_div,
            leinster_temp=leinster_temp,
            leinster_q=leinster_q,
        )

    def debug_surface_points(self, model: SupportsLatentConcat, Z: Tensor) -> list[Tensor]:
        """Force-refresh and return the per-shape masked surface point clouds (for visualizer)."""
        self._refresh_cache(model, Z)
        return self._cached_points or []

    @property
    def sources(self) -> list[InterfaceSource]:
        return self._sources
