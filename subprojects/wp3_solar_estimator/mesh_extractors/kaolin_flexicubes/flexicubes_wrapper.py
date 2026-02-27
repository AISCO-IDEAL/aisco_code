import os
import typing

import torch
from kaolin.ops.conversions import FlexiCubes


class FCExtractor:
    """
    FlexiCubes mesh extractor wrapper.

    Inspired by the DISO wrapper structure: generates a fixed sampling grid
    (bbox + resolution) once via `set_grid`, then extracts a mesh by evaluating
    the provided SDF `model` on that grid and running Kaolin's FlexiCubes.

    Usage:
            fx = FCExtractor(device="cuda", verbose=True)
            fx.set_grid(bbox=[[xmin,xmax],[ymin,ymax],[zmin,zmax]], resolution=128)
            verts, faces = fx.extract_mesh(model)
    """

    def __init__(
        self,
        device: str = "cuda",
        verbose: bool = True,
        fixed_box: bool = True,
        deterministic: bool = True,
    ):
        self.device = device
        self.verbose = verbose
        self._fixed_box = fixed_box
        self._grid_points: torch.Tensor | None = None
        self._resolution: int | None = None

        if deterministic:
            # is optional: improve reproducibility for benchmarking/backprop tests
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(42)
            # Some environments require this for determinism with cuBLAS
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def _to_nested_bbox(self, bbox: list) -> list[list[float]]:
        """
        Convert flat bbox [xmin, xmax, ymin, ymax, zmin, zmax] to
        nested [[xmin,xmax],[ymin,ymax],[zmin,zmax]] if needed.
        """
        if isinstance(bbox, list | tuple) and len(bbox) == 6 and not isinstance(bbox[0], list | tuple):  # noqa: PLR2004
            return [[bbox[0], bbox[1]], [bbox[2], bbox[3]], [bbox[4], bbox[5]]]
        return bbox  # assume already nested

    def set_grid(self, bbox: list, resolution: int):
        """
        Prepare a fixed sampling grid for SDF evaluation.

        Args:
                bbox: [[xmin,xmax],[ymin,ymax],[zmin,zmax]] or flat [xmin,xmax,ymin,ymax,zmin,zmax]
                resolution: integer grid resolution per axis
        """
        if not self._fixed_box:
            raise NotImplementedError("Currently only fixed_box=True is supported")

        bbox_nested = self._to_nested_bbox(bbox)

        # Create meshgrid in requested bbox on device
        x = torch.linspace(bbox_nested[0][0], bbox_nested[0][1], resolution, device=self.device)
        y = torch.linspace(bbox_nested[1][0], bbox_nested[1][1], resolution, device=self.device)
        z = torch.linspace(bbox_nested[2][0], bbox_nested[2][1], resolution, device=self.device)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

        self._grid_points = grid
        self._resolution = resolution

        if self.verbose:
            print(
                f"FlexiCubes grid prepared: res={resolution}^3, "
                f"bbox=([{bbox_nested[0][0]}, {bbox_nested[0][1]}], "
                f"[{bbox_nested[1][0]}, {bbox_nested[1][1]}], "
                f"[{bbox_nested[2][0]}, {bbox_nested[2][1]}])"
            )

    def _eval_sdf_batched(self, model: typing.Callable, batch_size: int = 4096) -> torch.Tensor:
        """
        Evaluate SDF on the fixed grid in batches to avoid OOM.
        Returns a 1D tensor of SDF values of length (resolution^3).
        """
        if self._grid_points is None:
            raise RuntimeError("Grid points not initialized. Call set_grid() first.")

        points = self._grid_points
        values = []
        # Ensure gradients can flow through SDF evaluation if needed
        with torch.enable_grad():
            for i in range(0, points.shape[0], batch_size):
                batch = points[i : i + batch_size]
                sdf_batch = model(batch)
                if sdf_batch.dim() > 1:
                    sdf_batch = sdf_batch.squeeze(-1)
                values.append(sdf_batch)
        return torch.cat(values, dim=0)

    def extract_mesh(self, model: typing.Callable, batch_size: int = 4096):
        """
        Extract a mesh using Kaolin FlexiCubes given an SDF model.

        Args:
                model: Callable SDF function (N,3) -> (N,) or (N,1)
                batch_size: Batch size for SDF evaluation

        Returns:
                (verts, faces): torch.Tensors suitable for downstream backpropagation
        """
        if self._grid_points is None or self._resolution is None:
            raise RuntimeError("FlexiCubes grid not set. Call set_grid(bbox, resolution) before extracting.")

        # 1) Evaluate SDF on grid
        sdf_values = self._eval_sdf_batched(model, batch_size=batch_size)

        # 2) Prepare FlexiCubes indices for voxel grid (resolution-1 cubes)
        flexicubes = FlexiCubes(device=self.device)
        _, cube_idx = flexicubes.construct_voxel_grid(self._resolution - 1)

        # 3) Run FlexiCubes to extract mesh
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        verts, faces, _ = flexicubes(
            voxelgrid_vertices=self._grid_points,  # (N,3) points
            scalar_field=sdf_values,  # (N,) SDF values
            cube_idx=cube_idx,  # (num_cubes, 8) indices
            resolution=self._resolution - 1,  # grid resolution in cubes
        )

        if (verts.numel() == 0 or faces.numel() == 0) and self.verbose:
            print("Warning: FlexiCubes produced an empty mesh.")

        return verts, faces
