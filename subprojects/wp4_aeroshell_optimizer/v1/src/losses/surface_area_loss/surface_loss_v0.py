# Aims to reduce surface area, with a softplus activation that weakens near TARGET_AREA_M2.
# SDF coordinates are in mm; surface area is converted to m² before applying the activation.

from typing import ClassVar

import torch
import torch.nn.functional as F  # noqa: N812 is standard alias for torch.nn.functional

from subprojects.wp3_solar_estimator.mesh_extractors.mesh_sdf.mesh_sdf_extractor import MeshSDFExtractor
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss


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

    # MeshSDF extractor settings (hard-coded for speed/quality balance)
    _RESOLUTION: int = 64
    # in mm, covers typical aeroshell
    _BBOX: ClassVar[list[float]] = [
        -5000,
        5000,
        -5000,
        5000,
        -5000,
        5000,
    ]  # not sure what the ClassVar does, ruff complains otherwise

    def _setup(self) -> None:
        self.mesh_extractor = MeshSDFExtractor(
            resolution=self._RESOLUTION,
            bbox=self._BBOX,
            isovalue=0.0,
            device=self.device,
        )

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
        _vertices, _faces, _area_loss, surrogate_loss, _ = self.mesh_extractor.extract_and_compute_loss(
            sdf_model=model,
            loss_fn=self._area_loss_fn,
            enable_backward=True,
        )
        return surrogate_loss
