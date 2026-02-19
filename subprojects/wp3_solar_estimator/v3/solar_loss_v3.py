"""
Solar loss (v3): unified wrapper to compare SDF->mesh extraction methods.

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


class SolarMeshLossFnV3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertices, faces, sun_direction, area_budget_mm2_tensor, epsilon: float):
        if faces.numel() == 0 or vertices.numel() == 0:
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

        # Illumination quality is the cosine of the angle between face normal and sun direction, clamped to [0, 1]
        illumination_quality = torch.clamp((face_normals_unit * sun_direction_unit).sum(-1), 0.0, 1.0)

        # Greedy selection of faces based on quality/area ratio until area budget is reached
        area_budget_mm2 = float(area_budget_mm2_tensor.item())
        # Sort by descending quality
        quality_order_desc = torch.argsort(illumination_quality, descending=True)
        selected_face_indices_list: list[int] = []
        accumulated_area_mm2 = 0.0
        for face_idx in quality_order_desc.tolist():
            area_i = float(face_area_mm2[face_idx].detach())
            # if face has zero area, skip it
            if area_i <= 0.0:
                continue
            # if illumination quality is zero (orhtogonal) or negative (back-facing), break the loop as we won't select any more faces
            if float(illumination_quality[face_idx].detach()) <= 0.0:
                break
            # if adding this face would exceed the area budget, break the loop
            if accumulated_area_mm2 + area_i <= area_budget_mm2 + 1e-12:
                selected_face_indices_list.append(face_idx)
                accumulated_area_mm2 += area_i

        selected_face_indices = torch.tensor(selected_face_indices_list, device=vertices.device, dtype=torch.long)

        ctx.save_for_backward(
            vertices,
            faces,
            selected_face_indices,
            edge_a[selected_face_indices] if selected_face_indices.numel() else edge_a.new_empty((0, 3)),
            edge_b[selected_face_indices] if selected_face_indices.numel() else edge_b.new_empty((0, 3)),
            sun_direction,
        )
        ctx.epsilon = float(epsilon)

        if selected_face_indices.numel() == 0:
            return torch.zeros((), device=vertices.device, dtype=vertices.dtype)

        selected_areas_mm2 = face_area_mm2[selected_face_indices]
        selected_quality = illumination_quality[selected_face_indices]
        loss_mm2 = (selected_areas_mm2 * selected_quality).sum()  # in mm^2
        return loss_mm2 * 1e-6  # convert to m^2

    @staticmethod
    def backward(ctx, grad_output):
        vertices, faces, selected_face_indices, selected_edge_a, selected_edge_b, sun_direction = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None

        grad_vertices = torch.zeros_like(vertices)
        if selected_face_indices.numel():
            faces_selected = faces[selected_face_indices]
            idx_v0, idx_v1, idx_v2 = faces_selected[:, 0], faces_selected[:, 1], faces_selected[:, 2]

            sun_direction_unit = sun_direction / (torch.linalg.norm(sun_direction).clamp_min(1e-12))
            dloss_dcross = 0.5 * sun_direction_unit * 1e-6  # convert to m^2

            grad_edge_a = torch.cross(dloss_dcross.expand_as(selected_edge_b), selected_edge_b, dim=-1)
            grad_edge_b = torch.cross(selected_edge_a, dloss_dcross.expand_as(selected_edge_a), dim=-1)

            grad_v1 = grad_edge_a
            grad_v2 = grad_edge_b
            grad_v0 = -(grad_v1 + grad_v2)
            grad_vertices.index_add_(0, idx_v1, grad_v1)
            grad_vertices.index_add_(0, idx_v2, grad_v2)
            grad_vertices.index_add_(0, idx_v0, grad_v0)

        grad_vertices_out = grad_vertices * grad_output * 1e-6
        return grad_vertices_out, None, None, None, None, None


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

    def _solar_mesh_loss(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        area_budget_tensor = torch.tensor(float(self.area_budget_mm2), device=vertices.device, dtype=vertices.dtype)
        return SolarMeshLossFnV3.apply(vertices, faces, self.sun_dir, area_budget_tensor, float(self.eps))

    def extract_mesh(self, model: Callable[[torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.method == "flexicubes":
            assert isinstance(self.extractor, FCExtractor)
            return self.extractor.extract_mesh(model, batch_size=self.batch_size)
        if self.method in {"diso_mc", "diso_dmc"}:
            assert isinstance(self.extractor, DISOExtractor)
            return self.extractor.extract_mesh(model, isovalue=0.0, batch_size=self.batch_size)
        if self.method in {"mesh_sdf", "mc"}:
            assert isinstance(self.extractor, MeshSDFExtractor)
            return self.extractor.extract_mesh(model)

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
        loss = 0
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
