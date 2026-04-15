"""Unified SDF to mesh extraction with config-driven adaptive refinement."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, cast

import numpy as np
import torch
import trimesh
import yaml
from skimage import measure

SDFModel = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True, slots=True)
class MeshExtractionConfig:
    """Explicit configuration for adaptive SDF mesh extraction."""

    method: str
    device: str
    isovalue: float
    bbox: list[list[float]]
    base_resolution: int
    subdivision_factor: int
    max_refine_levels: int
    leaf_cell_resolution: int
    sign_epsilon: float
    batch_size: int
    surface_area_tolerance: float
    merge_vertices: bool


@dataclass(frozen=True, slots=True)
class _Cell:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    depth: int


def _default_config_path() -> Path:
    return Path(__file__).with_name("mesh_extraction_config.yaml")


def _normalise_bbox(bbox: list[list[float]] | list[float]) -> list[list[float]]:
    if len(bbox) == 6 and all(isinstance(value, (int, float)) for value in bbox):
        numeric_bbox = cast(list[float], bbox)
        flat_bbox = [float(value) for value in numeric_bbox]
        return [[flat_bbox[0], flat_bbox[1]], [flat_bbox[2], flat_bbox[3]], [flat_bbox[4], flat_bbox[5]]]
    if len(bbox) != 3:
        raise ValueError("bbox must contain either 6 flat values or 3 [min, max] pairs.")

    nested_bbox: list[list[float]] = []
    for axis_bounds in bbox:
        if not isinstance(axis_bounds, list) or len(axis_bounds) != 2:
            raise ValueError("Each bbox axis must contain exactly two values: [min, max].")
        nested_bbox.append([float(axis_bounds[0]), float(axis_bounds[1])])
    return nested_bbox


def _validate_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required_keys = {
        "method",
        "device",
        "isovalue",
        "bbox",
        "base_resolution",
        "subdivision_factor",
        "max_refine_levels",
        "leaf_cell_resolution",
        "sign_epsilon",
        "batch_size",
        "surface_area_tolerance",
        "merge_vertices",
    }
    missing_keys = sorted(required_keys.difference(payload))
    if missing_keys:
        raise KeyError(f"Mesh extraction config is missing required keys: {missing_keys}")
    return payload


def load_mesh_extraction_config(config_path: Path | None = None) -> MeshExtractionConfig:
    """Load the mesh extraction config from the WP5 config file."""

    resolved_path = config_path if config_path is not None else _default_config_path()
    with resolved_path.open("r", encoding="utf-8") as file_handle:
        payload = yaml.safe_load(file_handle)

    if not isinstance(payload, dict):
        raise TypeError(f"Mesh extraction config must be a mapping, got {type(payload).__name__}.")

    payload = _validate_config_payload(payload)
    return MeshExtractionConfig(
        method=str(payload["method"]),
        device=str(payload["device"]),
        isovalue=float(payload["isovalue"]),
        bbox=_normalise_bbox(payload["bbox"]),
        base_resolution=int(payload["base_resolution"]),
        subdivision_factor=int(payload["subdivision_factor"]),
        max_refine_levels=int(payload["max_refine_levels"]),
        leaf_cell_resolution=int(payload["leaf_cell_resolution"]),
        sign_epsilon=float(payload["sign_epsilon"]),
        batch_size=int(payload["batch_size"]),
        surface_area_tolerance=float(payload["surface_area_tolerance"]),
        merge_vertices=bool(payload["merge_vertices"]),
    )


def compute_surface_area(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute triangle mesh surface area."""

    if vertices.numel() == 0 or faces.numel() == 0:
        device = vertices.device if vertices.numel() > 0 else faces.device
        return torch.tensor(0.0, device=device)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = torch.cross(v1 - v0, v2 - v0, dim=1)
    return 0.5 * torch.norm(cross, dim=1).sum()


class MeshSDFExtractor:
    """Adaptive active-cell SDF mesh extractor."""

    def __init__(self, config: MeshExtractionConfig | None = None) -> None:
        self.config = config if config is not None else load_mesh_extraction_config()
        self.device = self.config.device
        self.method = self.config.method
        self.bbox = self.config.bbox
        self.isovalue = self.config.isovalue
        self.base_resolution = self.config.base_resolution
        self.subdivision_factor = self.config.subdivision_factor
        self.max_refine_levels = self.config.max_refine_levels
        self.leaf_cell_resolution = self.config.leaf_cell_resolution
        self.sign_epsilon = self.config.sign_epsilon
        self.batch_size = self.config.batch_size
        self.surface_area_tolerance = self.config.surface_area_tolerance
        self.merge_vertices = self.config.merge_vertices
        self._point_cache: dict[tuple[float, float, float], float] = {}
        self._output_mesh: trimesh.Trimesh | None = None

        self._validate_runtime_config()

    def _validate_runtime_config(self) -> None:
        if self.subdivision_factor < 2:
            raise ValueError("subdivision_factor must be >= 2.")
        if self.base_resolution < 2:
            raise ValueError("base_resolution must be >= 2.")
        if self.max_refine_levels < 0:
            raise ValueError("max_refine_levels must be >= 0.")
        if self.leaf_cell_resolution < 2:
            raise ValueError("leaf_cell_resolution must be >= 2.")

    def set_grid(self, bbox: list[list[float]] | list[float], resolution: int) -> None:
        """Update the active bounding box and base resolution."""

        self.bbox = _normalise_bbox(bbox)
        self.base_resolution = int(resolution)
        self._validate_runtime_config()

    def _round_point(self, point: torch.Tensor) -> tuple[float, float, float]:
        point_values = point.tolist()
        return (
            round(float(point_values[0]), 10),
            round(float(point_values[1]), 10),
            round(float(point_values[2]), 10),
        )

    def _evaluate_points(self, sdf_model: SDFModel, points: torch.Tensor) -> torch.Tensor:
        values: list[float] = [0.0] * points.shape[0]
        uncached_points: list[torch.Tensor] = []
        uncached_indices: list[int] = []
        uncached_keys: list[tuple[float, float, float]] = []

        for index, point in enumerate(points):
            key = self._round_point(point)
            cached_value = self._point_cache.get(key)
            if cached_value is None:
                uncached_points.append(point)
                uncached_indices.append(index)
                uncached_keys.append(key)
            else:
                values[index] = cached_value

        if uncached_points:
            stacked_points = torch.stack(uncached_points).to(self.device)
            batches: list[torch.Tensor] = []
            with torch.no_grad():
                for start in range(0, stacked_points.shape[0], self.batch_size):
                    batch = stacked_points[start : start + self.batch_size]
                    batch_values = sdf_model(batch)
                    if batch_values.dim() > 1:
                        batch_values = batch_values.squeeze(-1)
                    batches.append(batch_values.detach().reshape(-1).to("cpu"))

            uncached_values = torch.cat(batches, dim=0).tolist()
            for index, key, value in zip(uncached_indices, uncached_keys, uncached_values, strict=True):
                scalar_value = float(value)
                self._point_cache[key] = scalar_value
                values[index] = scalar_value

        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _make_grid_points(
        self,
        minimum: tuple[float, float, float],
        maximum: tuple[float, float, float],
        resolution: int,
    ) -> torch.Tensor:
        axes = [torch.linspace(minimum[index], maximum[index], resolution, device=self.device) for index in range(3)]
        grid_x, grid_y, grid_z = torch.meshgrid(*axes, indexing="ij")
        return torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    def _cell_corner_points(self, cell: _Cell) -> torch.Tensor:
        xs = torch.tensor([cell.minimum[0], cell.maximum[0]], dtype=torch.float32, device=self.device)
        ys = torch.tensor([cell.minimum[1], cell.maximum[1]], dtype=torch.float32, device=self.device)
        zs = torch.tensor([cell.minimum[2], cell.maximum[2]], dtype=torch.float32, device=self.device)
        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
        return torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    def _is_active(self, values: torch.Tensor) -> bool:
        shifted = values - self.isovalue
        lower = torch.any(shifted < -self.sign_epsilon)
        upper = torch.any(shifted > self.sign_epsilon)
        near_zero = torch.any(torch.abs(shifted) <= self.sign_epsilon)
        return bool((lower and upper) or (near_zero and (lower or upper)))

    def _subdivide_cell(self, cell: _Cell) -> list[_Cell]:
        minimum = np.asarray(cell.minimum, dtype=np.float32)
        maximum = np.asarray(cell.maximum, dtype=np.float32)
        step = (maximum - minimum) / float(self.subdivision_factor)
        children: list[_Cell] = []
        for ix in range(self.subdivision_factor):
            for iy in range(self.subdivision_factor):
                for iz in range(self.subdivision_factor):
                    child_min = minimum + np.array([ix, iy, iz], dtype=np.float32) * step
                    child_max = child_min + step
                    children.append(
                        _Cell(
                            minimum=(float(child_min[0]), float(child_min[1]), float(child_min[2])),
                            maximum=(float(child_max[0]), float(child_max[1]), float(child_max[2])),
                            depth=cell.depth + 1,
                        )
                    )
        return children

    def _refine_active_cells(self, sdf_model: SDFModel) -> list[tuple[_Cell, torch.Tensor]]:  # noqa: PLR0912
        bbox = self.bbox
        base_axes = [
            torch.linspace(bbox[axis][0], bbox[axis][1], self.base_resolution, device=self.device)
            for axis in range(3)
        ]
        grid_x, grid_y, grid_z = torch.meshgrid(*base_axes, indexing="ij")
        base_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        base_values = self._evaluate_points(sdf_model, base_points).reshape(
            self.base_resolution,
            self.base_resolution,
            self.base_resolution,
        )

        active_cells: list[tuple[_Cell, torch.Tensor]] = []
        for ix in range(self.base_resolution - 1):
            for iy in range(self.base_resolution - 1):
                for iz in range(self.base_resolution - 1):
                    cell = _Cell(
                        minimum=(float(base_axes[0][ix]), float(base_axes[1][iy]), float(base_axes[2][iz])),
                        maximum=(float(base_axes[0][ix + 1]), float(base_axes[1][iy + 1]), float(base_axes[2][iz + 1])),
                        depth=0,
                    )
                    corner_values = base_values[ix : ix + 2, iy : iy + 2, iz : iz + 2].reshape(-1)
                    if self._is_active(corner_values):
                        active_cells.append((cell, corner_values))

        leaf_cells: list[tuple[_Cell, torch.Tensor]] = []
        current_level = active_cells
        for _depth in range(self.max_refine_levels):
            next_level: list[tuple[_Cell, torch.Tensor]] = []
            for cell, corner_values in current_level:
                if cell.depth >= self.max_refine_levels:
                    leaf_cells.append((cell, corner_values))
                    continue

                active_children: list[tuple[_Cell, torch.Tensor]] = []
                for child_cell in self._subdivide_cell(cell):
                    child_corner_points = self._cell_corner_points(child_cell)
                    child_corner_values = self._evaluate_points(sdf_model, child_corner_points)
                    if self._is_active(child_corner_values):
                        active_children.append((child_cell, child_corner_values))

                if active_children:
                    next_level.extend(active_children)
                else:
                    leaf_cells.append((cell, corner_values))

            if not next_level:
                break
            current_level = next_level
        else:
            leaf_cells.extend(current_level)

        return leaf_cells

    def _extract_leaf_mesh(self, sdf_model: SDFModel, cell: _Cell) -> trimesh.Trimesh:
        sample_points = self._make_grid_points(cell.minimum, cell.maximum, self.leaf_cell_resolution)
        sdf_values = self._evaluate_points(sdf_model, sample_points)
        sdf_grid = sdf_values.reshape(self.leaf_cell_resolution, self.leaf_cell_resolution, self.leaf_cell_resolution)

        sdf_grid_np = sdf_grid.detach().cpu().numpy()
        sdf_min = float(np.min(sdf_grid_np))
        sdf_max = float(np.max(sdf_grid_np))
        if not (sdf_min <= self.isovalue <= sdf_max):
            raise RuntimeError(
                "Leaf cell does not bracket the isovalue. Increase base_resolution or refine_levels in the mesh config."
            )

        spacing = tuple(
            (cell.maximum[index] - cell.minimum[index]) / float(self.leaf_cell_resolution - 1)
            for index in range(3)
        )
        vertices, faces, _, _ = measure.marching_cubes(sdf_grid_np, level=self.isovalue, spacing=spacing)
        vertices[:, 0] += cell.minimum[0]
        vertices[:, 1] += cell.minimum[1]
        vertices[:, 2] += cell.minimum[2]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if self.merge_vertices:
            mesh.merge_vertices()
        return mesh

    def _extract_mesh_uniform(self, sdf_model: SDFModel) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniform grid marching cubes — fast path when max_refine_levels=0."""

        res = self.base_resolution
        bbox = self.bbox

        axes = [torch.linspace(bbox[a][0], bbox[a][1], res, device=self.device) for a in range(3)]
        gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
        points = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

        sdf_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, points.shape[0], self.batch_size):
                batch = points[start : start + self.batch_size]
                vals = sdf_model(batch)
                if vals.dim() > 1:
                    vals = vals.squeeze(-1)
                sdf_chunks.append(vals.cpu())

        sdf_volume = torch.cat(sdf_chunks).numpy().reshape(res, res, res)

        sdf_min, sdf_max = float(np.min(sdf_volume)), float(np.max(sdf_volume))
        if not (sdf_min <= self.isovalue <= sdf_max):
            print(
                f"[sdf2mesh WARNING] SDF does not bracket isovalue {self.isovalue} "
                f"(range [{sdf_min:.4f}, {sdf_max:.4f}]). Returning empty mesh."
            )
            return (
                torch.empty((0, 3), dtype=torch.float32, device=self.device),
                torch.empty((0, 3), dtype=torch.int64, device=self.device),
            )

        spacing = tuple((bbox[a][1] - bbox[a][0]) / (res - 1) for a in range(3))
        vertices_np, faces_np, _, _ = measure.marching_cubes(sdf_volume, level=self.isovalue, spacing=spacing)
        vertices_np[:, 0] += bbox[0][0]
        vertices_np[:, 1] += bbox[1][0]
        vertices_np[:, 2] += bbox[2][0]

        vertices = torch.from_numpy(vertices_np.astype(np.float32)).to(self.device)
        faces = torch.from_numpy(faces_np.astype(np.int64)).to(self.device)
        return vertices, faces

    def extract_mesh(self, sdf_model: SDFModel) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract a mesh by refining only sign-changing cells."""

        if isinstance(sdf_model, torch.nn.Module):
            sdf_model = sdf_model.to(self.device)
        self._point_cache.clear()

        if self.max_refine_levels == 0:
            return self._extract_mesh_uniform(sdf_model)

        leaf_cells = self._refine_active_cells(sdf_model)
        if not leaf_cells:
            raise RuntimeError(
                "Adaptive mesh extraction found no sign-changing cells. Increase base_resolution or expand the bbox."
            )

        meshes: list[trimesh.Trimesh] = []
        for cell, _corner_values in leaf_cells:
            meshes.append(self._extract_leaf_mesh(sdf_model, cell))

        if not meshes:
            raise RuntimeError("Adaptive mesh extraction produced no geometry.")

        combined_mesh = trimesh.util.concatenate(meshes)
        if self.merge_vertices:
            combined_mesh.merge_vertices()

        if combined_mesh.vertices.size == 0 or combined_mesh.faces.size == 0:
            raise RuntimeError("Adaptive mesh extraction returned an empty mesh.")

        vertices = torch.from_numpy(np.asarray(combined_mesh.vertices, dtype=np.float32)).to(self.device)
        faces = torch.from_numpy(np.asarray(combined_mesh.faces, dtype=np.int64)).to(self.device)
        self._output_mesh = combined_mesh
        return vertices, faces

    def compute_normals_at_vertices(self, vertices: torch.Tensor, sdf_model: SDFModel) -> torch.Tensor:
        """Compute surface normals at mesh vertices via SDF gradients."""

        vertices_grad = vertices.detach().clone().requires_grad_(True)
        sdf_values = sdf_model(vertices_grad)
        if sdf_values.dim() > 1:
            sdf_values = sdf_values.squeeze(-1)
        normals = torch.autograd.grad(
            outputs=sdf_values,
            inputs=vertices_grad,
            grad_outputs=torch.ones_like(sdf_values),
            create_graph=False,
            retain_graph=False,
        )[0]
        return normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)

    def compute_gradient_flow(self, vertices: torch.Tensor, dL_dvertices: torch.Tensor, sdf_model: SDFModel) -> torch.Tensor:
        """Project vertex gradients back to the SDF parameters."""

        vertices_grad = vertices.detach().clone().requires_grad_(True)
        sdf_values = sdf_model(vertices_grad)
        if sdf_values.dim() > 1:
            sdf_values = sdf_values.squeeze(-1)

        sdf_grad = torch.autograd.grad(
            outputs=sdf_values,
            inputs=vertices_grad,
            grad_outputs=torch.ones_like(sdf_values),
            create_graph=True,
            retain_graph=True,
        )[0]
        normals = sdf_grad / (torch.norm(sdf_grad, dim=1, keepdim=True) + 1e-8)
        dL_ds = -torch.sum(dL_dvertices * normals, dim=1)

        vertices_for_backward = vertices.requires_grad_(True)
        sdf_values_backward = sdf_model(vertices_for_backward)
        if sdf_values_backward.dim() > 1:
            sdf_values_backward = sdf_values_backward.squeeze(-1)
        return torch.sum(dL_ds.detach() * sdf_values_backward)

    def extract_and_compute_loss(
        self,
        sdf_model: SDFModel,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        enable_backward: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract a mesh, compute a loss on it, and return a surrogate loss term."""

        torch.cuda.synchronize()
        _t0 = time.perf_counter()
        vertices, faces = self.extract_mesh(sdf_model)
        torch.cuda.synchronize()
        _t1 = time.perf_counter()
        if vertices.numel() == 0 or faces.numel() == 0:
            print("[sdf2mesh WARNING] Empty mesh returned — using penalty loss.")
            empty_verts = torch.zeros((1, 3), device=self.device)
            empty_faces = torch.zeros((1, 3), dtype=torch.int64, device=self.device)
            penalty = torch.tensor(1e4, device=self.device, requires_grad=False)
            zero_surr = torch.tensor(0.0, device=self.device, requires_grad=True)
            empty_grad = torch.zeros((1, 3), device=self.device)
            print(f"[TIMING sdf2mesh] extract={_t1-_t0:.2f}s  (empty mesh, penalty)")
            return empty_verts, empty_faces, penalty, zero_surr, empty_grad

        vertices_for_loss = vertices.detach().requires_grad_(True)
        loss = loss_fn(vertices_for_loss, faces)
        torch.cuda.synchronize()
        _t2 = time.perf_counter()

        if not enable_backward:
            print(f"[TIMING sdf2mesh] extract={_t1-_t0:.2f}s  loss_fn={_t2-_t1:.2f}s  (no backward)")
            empty_grad = torch.zeros_like(vertices)
            return vertices, faces, loss, torch.tensor(0.0, device=vertices.device), empty_grad

        dL_dvertices = torch.autograd.grad(
            outputs=loss,
            inputs=vertices_for_loss,
            grad_outputs=torch.ones_like(loss),
            create_graph=False,
            retain_graph=False,
        )[0]
        torch.cuda.synchronize()
        _t3 = time.perf_counter()
        surrogate_loss = self.compute_gradient_flow(vertices=vertices, dL_dvertices=dL_dvertices, sdf_model=sdf_model)
        torch.cuda.synchronize()
        _t4 = time.perf_counter()
        print(f"[TIMING sdf2mesh] extract={_t1-_t0:.2f}s  loss_fn={_t2-_t1:.2f}s  autograd={_t3-_t2:.2f}s  grad_flow={_t4-_t3:.2f}s  total={_t4-_t0:.2f}s")
        return vertices, faces, loss, surrogate_loss, dL_dvertices


def _write_ply(vertices: torch.Tensor, faces: torch.Tensor, ply_path: str | Path) -> None:
    mesh = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), process=False)
    mesh.export(ply_path)


def extract_sdf_to_temp_ply(
    sdf_model: SDFModel,
    config: MeshExtractionConfig | None = None,
    output_name: str = "mesh_sdf_output.ply",
) -> str:
    """Extract a mesh from an SDF model and write it to a temporary PLY file."""

    extractor = MeshSDFExtractor(config=config)
    vertices, faces = extractor.extract_mesh(sdf_model)
    tmp_dir = mkdtemp(prefix="mesh_sdf_")
    ply_path = Path(tmp_dir) / output_name
    _write_ply(vertices, faces, ply_path)
    return str(ply_path)
