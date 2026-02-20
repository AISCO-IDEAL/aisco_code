"""
Solar loss (v3): fully differentiable soft-selection loss without custom autograd.

Supported methods:
  - mesh_sdf: MeshSDF (non-diff extraction + gradient projection surrogate)
  - flexicubes: Kaolin FlexiCubes (diff)
  - diso_mc: DISO DiffMC (diff)
  - diso_dmc: DISO DiffDMC (diff)
  - mc: Marching Cubes baseline (non-diff, forward only)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from subprojects.wp3_solar_estimator.mesh_extractors.diso.diso_wrapper import DISOExtractor
from subprojects.wp3_solar_estimator.mesh_extractors.kaolin_flexicubes.flexicubes_wrapper import FCExtractor
from subprojects.wp3_solar_estimator.mesh_extractors.mesh_sdf.mesh_sdf_extractor import MeshSDFExtractor


def _cumsum(x: torch.Tensor) -> torch.Tensor:
    """
    Manual cumulative sum implementation that is deterministic

    Args:
        x: (N,) input tensor
    Returns:
        (N,) cumulative sum tensor where cumsum[i] = x[0] + x[1] + ... + x[i]
    """
    cumsum = torch.zeros_like(x)
    running_sum = torch.zeros((), device=x.device, dtype=x.dtype)
    for i in range(x.shape[0]):
        running_sum += x[i]
        cumsum[i] = running_sum
    return cumsum


def solar_mesh_loss_v3(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    sun_direction: torch.Tensor,
    area_budget_mm2: float,
    sigmoid_sharpness: float = 10.0,
    budget_penalty_weight: float = 0.1,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """
    Compute differentiable solar mesh loss using soft selection.

    Args:
        vertices: (N, 3) mesh vertices in mm
        faces: (M, 3) face indices
        sun_direction: (3,) sun direction vector
        area_budget_mm2: maximum selectable area in mm²
        sigmoid_sharpness: steepness of selection blur (higher = sharper boundary)
        budget_penalty_weight: weight for unused budget penalty
        epsilon: numerical stability

    Returns:
        Scalar loss (in m²)
    """
    if faces.numel() == 0 or vertices.numel() == 0:
        print("Warning: Empty vertices or faces in solar_mesh_loss_v3, returning zero loss")
        return torch.zeros((), device=vertices.device, dtype=vertices.dtype)

    # Compute face areas and normals
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    edge_a, edge_b = v1 - v0, v2 - v0
    face_cross = torch.cross(edge_a, edge_b, dim=-1)
    face_cross_norm = torch.linalg.norm(face_cross, dim=-1)
    face_area_mm2 = 0.5 * face_cross_norm

    # Compute illumination quality (cosine of angle with sun direction, clamped to [0, 1])
    face_normals_unit = face_cross / (face_cross_norm.clamp_min(epsilon)).unsqueeze(-1)
    sun_direction_unit = sun_direction / (torch.linalg.norm(sun_direction).clamp_min(epsilon))
    illumination_quality = torch.clamp((face_normals_unit * sun_direction_unit).sum(-1), 0.0, 1.0)

    # Sort faces by descending quality
    # NOTE: Currently ignores order priority for equal-quality faces (TODO: could optimize face ordering)
    quality_order_desc = torch.argsort(illumination_quality, descending=True)
    sorted_areas = face_area_mm2[quality_order_desc]
    sorted_quality = illumination_quality[quality_order_desc]

    # Deterministic-safe cumulative area
    cumulative_area = _cumsum(sorted_areas)

    # Create soft selection mask using sigmoid
    # mask(i) = sigmoid(sharpness * (area_budget - cumsum_area[i]))
    # This smoothly transitions from 1.0 (well within budget) to 0.0 (exceeds budget)
    distance_to_budget = (area_budget_mm2 - cumulative_area) / max(area_budget_mm2, epsilon)
    soft_mask = torch.sigmoid(sigmoid_sharpness * distance_to_budget)

    # Weighted quality-area product
    weighted_quality_area = sorted_areas * sorted_quality * soft_mask
    quality_loss = weighted_quality_area.sum()

    # Penalty for unused budget (encourage filling the budget)
    total_selected_area = (sorted_areas * soft_mask).sum()
    # Normalize budget usage to dimensionless ratio
    unused_ratio = torch.clamp((area_budget_mm2 - total_selected_area) / max(area_budget_mm2, epsilon), min=0.0)
    budget_penalty = budget_penalty_weight * (unused_ratio**2)

    # Combined loss: maximize quality (minimize negative) + penalize unused budget
    loss_mm2 = quality_loss - budget_penalty
    return -loss_mm2 * 1e-6


class SolarLossEstimator:
    def __init__(self, config: dict | None = None) -> None:
        defaults = {
            "method": "flexicubes",
            "bbox": [[-5000, 5000], [-5000, 5000], [-5000, 5000]],
            "resolution": 64,
            "area_budget_mm2": 1e6,
            "sun_dir": [0.0, 1.0, 1.0],
            "device": "cuda",
            "batch_size": 4096,
            "eps": 1e-12,
            "sigmoid_sharpness": 10.0,
            "budget_penalty_weight": 0.1,
        }
        cfg = {**defaults, **(config or {})}
        self.config = cfg

        self.method = cfg["method"]
        self.bbox = cfg["bbox"]
        self.resolution = int(cfg["resolution"])
        self.area_budget_mm2 = float(cfg["area_budget_mm2"])
        self.device = cfg["device"]
        self.batch_size = int(cfg["batch_size"])
        self.eps = float(cfg["eps"])
        self.sigmoid_sharpness = float(cfg["sigmoid_sharpness"])
        self.budget_penalty_weight = float(cfg["budget_penalty_weight"])

        sun_dir = cfg["sun_dir"]
        if not isinstance(sun_dir, torch.Tensor):
            self.sun_dir = torch.tensor(sun_dir, device=self.device, dtype=torch.float32)
        else:
            self.sun_dir = sun_dir.to(device=self.device, dtype=torch.float32)

        self.extractor: FCExtractor | DISOExtractor | MeshSDFExtractor
        self._init_extractor()

    def _init_extractor(self) -> None:
        method = self.method
        if method == "flexicubes":
            self.extractor = FCExtractor(device=self.device, verbose=False, fixed_box=True)
            self.extractor.set_grid(self.bbox, self.resolution)
        elif method in {"diso_mc", "diso_dmc"}:
            extractor_type = "DiffMC" if method == "diso_mc" else "DiffDMC"
            self.extractor = DISOExtractor(
                extractor_type=extractor_type,
                dtype="float32",
                device=self.device,
                verbose=False,
                deform=False,
                fixed_box=True,
            )
            self.extractor.set_grid(self.bbox, self.resolution)
        elif method in {"mesh_sdf", "mc"}:
            bbox = self.bbox
            self.extractor = MeshSDFExtractor(
                resolution=self.resolution,
                bbox=[
                    bbox[0][0],
                    bbox[0][1],
                    bbox[1][0],
                    bbox[1][1],
                    bbox[2][0],
                    bbox[2][1],
                ],
                isovalue=0.0,
                device=self.device,
            )
            self.extractor.set_grid(self.bbox, self.resolution)
        else:
            raise ValueError(f"Unknown method: {method}")

    def extract_mesh(self, model: Callable[[torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.method == "flexicubes":
            assert isinstance(self.extractor, FCExtractor)
            return self.extractor.extract_mesh(model, batch_size=self.batch_size)
        if self.method in {"diso_mc", "diso_dmc"}:
            assert isinstance(self.extractor, DISOExtractor)
            return self.extractor.extract_mesh(model, isovalue=0.0, batch_size=self.batch_size)
        if self.method in {"mesh_sdf", "mc"}:
            assert isinstance(self.extractor, MeshSDFExtractor)
            return self.extractor.extract_mesh(model=model)

        raise ValueError(f"Unknown method: {self.method}")

    def solar_loss(self, model: Any, epoch_name: int | None = None) -> torch.Tensor:
        """
        Compute the solar loss for the given model.

        Args:
            model: SDF model (nn.Module)
            epoch_name: Current epoch (for logging)

        Returns:
            Scalar loss tensor
        """
        vertices, faces = self.extract_mesh(model)
        loss = solar_mesh_loss_v3(
            vertices,
            faces,
            self.sun_dir,
            self.area_budget_mm2,
            sigmoid_sharpness=self.sigmoid_sharpness,
            budget_penalty_weight=self.budget_penalty_weight,
            epsilon=self.eps,
        )
        return loss


def sweep_solar_methods(
    model: Any,
    methods: list[str],
    config_factory: Callable[[str], dict],
    compute_grad_norm: bool = False,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for method in methods:
        cfg = config_factory(method)
        loss_fn = SolarLossEstimator(cfg)
        loss_val = loss_fn.solar_loss(model)
        entry: dict[str, float] = {"loss": float(loss_val.detach().cpu().item())}

        if compute_grad_norm:
            if hasattr(model, "zero_grad"):
                model.zero_grad(set_to_none=True)
            grads = torch.autograd.grad(loss_val, [p for p in model.parameters() if p.requires_grad], allow_unused=True)
            grad_norm = 0.0
            for g in grads:
                if g is not None:
                    grad_norm += float(g.detach().norm().cpu().item())
            entry["grad_norm"] = grad_norm

        results[method] = entry

    return results
