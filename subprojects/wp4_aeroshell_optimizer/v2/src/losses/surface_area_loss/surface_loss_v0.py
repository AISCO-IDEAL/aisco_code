"""Aims to reduce surface area, with a softplus activation that weakens near TARGET_AREA_M2.
# SDF coordinates are in mm; surface area is converted to m² before applying the activation.
"""

from dataclasses import replace as _dc_replace

import torch
import torch.nn.functional as F  # noqa: N812 is standard alias for torch.nn.functional

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss
from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor, load_mesh_extraction_config


class SurfaceLossV0(ParentLoss):
    """Surface area loss using MeshSDF differentiable mesh extraction.

    Extracts a triangle mesh from the SDF via Marching Cubes, computes the
    total surface area in m² (SDF space is in mm), and applies a softplus
    activation centred at TARGET_AREA_M2:

        loss = softplus( SOFTPLUS_K * (area_m2 - TARGET_AREA_M2) )

    This gives a smooth, near-zero penalty when area <= target, and a
    linearly growing penalty when area >> target. Gradients are propagated
    back to the SDF network via the full MeshSDF surrogate gradient flow.
    """

    TARGET_AREA_M2: float = 15.0  # desired upper-bound surface area [m²]
    MM2_TO_M2: float = 1e-6  # unit conversion: mm² → m²
    SOFTPLUS_K: float = 1.0  # sharpness; 1.0 gives ~1 m² transition width

    def _setup(self) -> None:
        mesh_config = load_mesh_extraction_config()
        mesh_config = _dc_replace(mesh_config, device=self.device)
        self.mesh_extractor = MeshSDFExtractor(config=mesh_config)
        # Bbox + resolution come from `mesh_extraction_config.yaml` so every
        # loss that uses MeshSDFExtractor (developability, surface_area, …)
        # produces the same `extraction_key` and reuses a single Marching
        # Cubes mesh via `_inject_shared_meshes`. The previous prism-bbox +
        # base_resolution combination created a redundant extraction whose
        # interior level sets briefly opened a torus-like hole in the
        # surface-gradient panel.
        if mesh_config.bbox is None:
            raise ValueError(
                "surface_loss_v0 requires `bbox` to be set in "
                "subprojects/wp5/unified_mesh_extraction/mesh_extraction_config.yaml."
            )
        self.mesh_extractor.set_grid(bbox=mesh_config.bbox, resolution=mesh_config.base_resolution)

    def _area_loss_fn(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Compute softplus-activated surface area loss from mesh geometry.

        Args:
            vertices: (V, 3) vertex positions in mm
            faces:    (F, 3) triangle index tensor

        Returns:
            Scalar softplus loss value.
        """
        v0 = vertices[faces[:, 0]]  # (F, 3)
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        cross = torch.cross(v1 - v0, v2 - v0, dim=1)  # (F, 3)
        area_mm2 = 0.5 * torch.norm(cross, dim=1).sum()  # scalar, mm²
        area_m2 = area_mm2 * self.MM2_TO_M2  # scalar, m²

        return F.softplus(self.SOFTPLUS_K * (area_m2 - self.TARGET_AREA_M2))

    def _loss(self, model) -> torch.Tensor:
        """Extract mesh and return MeshSDF surrogate loss with gradient flow.

        Args:
            model: SDF network; callable as model(points) -> SDF values

        Returns:
            Scalar surrogate loss (gradients flow back to SDF parameters).
        """
        _verts, _faces, _area_loss, surrogate_loss, _ = self.mesh_extractor.extract_and_compute_loss(
            sdf_model=model,
            loss_fn=self._area_loss_fn,
            enable_backward=True,
        )
        return surrogate_loss
