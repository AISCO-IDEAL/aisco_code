"""Strain-energy curvature loss on extracted SDF surface points.

Port of GINN's `loss_curv` (`train/losses_ginn.py:217-250`) using the strain expression
`(2H)² - 2K` from GINN's `compute_strain_curvatures` (`train/losses.py:381-390`), with
mean / Gauss curvature formulas from `get_mean_curvature_normalized` and
`get_gauss_curvature` (same file, lines 393-439).

For each call the loss extracts the zero level-set of the current model's SDF via
marching cubes (`MeshSDFExtractor`, also used by `field_diversity_surface_v0`), reuses
those vertex positions as autograd-tracked sample points, then computes ∇f and the full
3×3 Hessian via two passes of `torch.autograd.grad(create_graph=True)`. Marching-cubes
vertex positions are themselves non-differentiable — gradients flow only through the
re-evaluation of `model(p_surface)`. Same trick as `field_diversity_surface_v0` and
GINN's surface curvature path.

Surface is re-extracted on every `_loss` call (matches `field_diversity_surface_v0`'s
default cadence=1). No cache — the surface depends on the current model + latent state,
and the loss is called once per latent by the trainer, so caching across calls without
keying on latent would silently mis-attribute surfaces. Same single-shape / multi-shape
compatibility story as `smoothness_v0` and the eikonal losses (no model introspection).

Config block (in losses_v0_config.yaml):

    curvature_v1_config:
      surface_subsample_n: 4096
      mc_resolution: 64
      mc_bbox_scale: 1.0
      strain_curvature_clip_max: 500.0  # per-vertex clamp on (2H)^2 - 2K
      max_curv: 0.0                     # subtract baseline; loss = max(0, strain_mean - max_curv)
"""

from __future__ import annotations

import torch
from torch import Tensor

from subprojects.wp2_assembler.version4.utils.reg_imp import get_outer_prism_bounds_mm, scale_outer_prism
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss
from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor
from subprojects.wp5.unified_mesh_extraction.utils import MeshExtractionConfig, load_mesh_extraction_config


class CurvatureV1(ParentLoss):
    def _setup(self) -> None:
        cfg = self.losses_config.get("curvature_v1_config", {}) or {}
        self._surface_subsample_n: int = int(cfg.get("surface_subsample_n", 4096))
        self._mc_resolution: int = int(cfg.get("mc_resolution", 64))
        self._mc_bbox_scale: float = float(cfg.get("mc_bbox_scale", 1.0))
        self._strain_clip_max: float = float(cfg.get("strain_curvature_clip_max", 500.0))
        self._max_curv: float = float(cfg.get("max_curv", 0.0))

        self._extractor: MeshSDFExtractor = self._build_extractor()

    def _build_extractor(self) -> MeshSDFExtractor:
        prism = scale_outer_prism(get_outer_prism_bounds_mm(), self._mc_bbox_scale)
        bbox = [
            [prism["x"][0], prism["x"][1]],
            [prism["y"][0], prism["y"][1]],
            [prism["z"][0], prism["z"][1]],
        ]
        base_cfg = load_mesh_extraction_config()
        cfg = MeshExtractionConfig(
            device=str(self.device),
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

    def _loss(self, model) -> Tensor:
        with torch.no_grad():
            verts, _faces = self._extractor.extract_mesh(model)
        if verts.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        verts = verts.to(self.device, dtype=torch.float32)
        if verts.shape[0] > self._surface_subsample_n:
            idx = torch.randperm(verts.shape[0], device=verts.device)[: self._surface_subsample_n]
            verts = verts[idx]

        points = verts.detach().requires_grad_(True)

        with torch.enable_grad():
            sdf = model(points)
            if isinstance(sdf, tuple):
                sdf = sdf[0]
            sdf = sdf.squeeze(-1)

            grad_sdf = torch.autograd.grad(
                outputs=sdf,
                inputs=points,
                grad_outputs=torch.ones_like(sdf),
                create_graph=True,
                retain_graph=True,
            )[0]  # [N, 3]

            hessian_rows = []
            for dim in range(3):
                row = torch.autograd.grad(
                    outputs=grad_sdf[:, dim],
                    inputs=points,
                    grad_outputs=torch.ones_like(grad_sdf[:, dim]),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                hessian_rows.append(row)
            hessian = torch.stack(hessian_rows, dim=1)  # [N, 3, 3]

            # Mean curvature (normalised), GINN losses.py:393-416:
            #   H = -(F·Hess·Fᵀ - |F|²·tr(Hess)) / (2|F|³)
            grad_norm = grad_sdf.square().sum(dim=1).sqrt().clamp(min=1e-5)
            fhf = torch.einsum("bi,bij,bj->b", grad_sdf, hessian, grad_sdf)
            tr_h = torch.einsum("bii->b", hessian)
            mean_curv = -(fhf - grad_norm.pow(2) * tr_h) / (2.0 * grad_norm.pow(3))

            # Gauss curvature, losses.py:418-439:
            #   K = -1/|F|⁴ · det([H, F; Fᵀ, 0])
            n = grad_sdf.shape[0]
            bordered = torch.cat(
                [
                    torch.cat([hessian, grad_sdf.unsqueeze(2)], dim=2),  # [N, 3, 4]
                    torch.cat(
                        [grad_sdf.unsqueeze(1), torch.zeros(n, 1, 1, device=self.device)],
                        dim=2,
                    ),  # [N, 1, 4]
                ],
                dim=1,
            )  # [N, 4, 4]
            grad_norm4 = grad_sdf.square().sum(dim=1).square().clamp(min=1e-15)
            gauss_curv = -torch.det(bordered) / grad_norm4

            strain = (2.0 * mean_curv).square() - 2.0 * gauss_curv
            strain = strain.clamp(max=self._strain_clip_max)

            loss = strain.mean() - self._max_curv
            return loss.clamp(min=0.0)
