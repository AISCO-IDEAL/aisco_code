"""
MeshSDF: Differentiable mesh extraction using Marching Cubes with gradient flow.

Based on the MeshSDF paper (CVPR 2020): https://github.com/cvlab-epfl/MeshSDF
Core idea: Make standard Marching Cubes differentiable by deriving analytical
gradient flow from mesh vertices back to SDF parameters via the implicit surface constraint.
"""

import warnings

import numpy as np
import torch
from skimage import measure
from torch import nn


class MeshSDFExtractor:
    """
    Differentiable isosurface extraction using MeshSDF approach.

    Key innovation: Computes gradients through the zero-level set constraint
    without modifying the Marching Cubes algorithm itself.

    Gradient flow (Equation 4 from paper):
    1. Extract mesh vertices using standard Marching Cubes
    2. Compute normals at vertices via SDF gradient: n = ∇_x SDF(x) / ||∇_x SDF(x)||
    3. Project upstream gradients onto normals: dL/ds = -dL/dx · n
    4. Backward pass through SDF evaluation: chain rule to parameters
    """

    def __init__(self, resolution: int = 128, bbox: list | None = None, isovalue: float = 0.0, device: str = "cuda"):
        """
        Args:
            resolution: Grid resolution for SDF sampling (N x N x N)
            bbox: Bounding box [x_min, x_max, y_min, y_max, z_min, z_max]
            isovalue: Isosurface level to extract (typically 0.0 for zero-level set)
            device: Device for computation
        """
        if bbox is None:
            bbox = [-5000, 5000, -5000, 5000, -5000, 5000]
        self.resolution = resolution
        self.bbox = bbox
        self.isovalue = isovalue
        self.device = device

        # Cache for optimization
        self._cached_grid = None
        self._cached_sdf = None

        # Fixed grid support (to align with DISO/FC wrappers)
        self._grid_points: torch.Tensor | None = None
        self._resolution: int | None = None

    def _to_nested_bbox(self, bbox: list) -> list[list[float]]:
        """
        Convert flat bbox [xmin, xmax, ymin, ymax, zmin, zmax] to nested
        [[xmin,xmax],[ymin,ymax],[zmin,zmax]] if needed.
        """
        if isinstance(bbox, (list | tuple)) and len(bbox) == 6 and not isinstance(bbox[0], (list | tuple)):  # noqa: PLR2004
            return [[bbox[0], bbox[1]], [bbox[2], bbox[3]], [bbox[4], bbox[5]]]
        return bbox

    def set_grid(self, bbox: list, resolution: int):
        """
        Prepare a fixed sampling grid for SDF evaluation (similar to DISO/FC APIs).

        Args:
            bbox: [[xmin,xmax],[ymin,ymax],[zmin,zmax]] or flat [xmin,xmax,ymin,ymax,zmin,zmax]
            resolution: integer grid resolution per axis
        """
        bbox_nested = self._to_nested_bbox(bbox)
        # Store bbox in flat form for spacing and vertex offset computation

        self.bbox = [
            bbox_nested[0][0],
            bbox_nested[0][1],
            bbox_nested[1][0],
            bbox_nested[1][1],
            bbox_nested[2][0],
            bbox_nested[2][1],
        ]
        self._resolution = resolution

        # Create meshgrid in requested bbox on device
        x = torch.linspace(bbox_nested[0][0], bbox_nested[0][1], resolution, device=self.device)
        y = torch.linspace(bbox_nested[1][0], bbox_nested[1][1], resolution, device=self.device)
        z = torch.linspace(bbox_nested[2][0], bbox_nested[2][1], resolution, device=self.device)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

        self._grid_points = grid
        if self._grid_points.device.type != self.device:
            self._grid_points = self._grid_points.to(self.device)

    def extract_mesh(self,
                     sdf_model: nn.Module,
                     return_cache: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract isosurface mesh using Marching Cubes.

        Args:
            sdf_model: PyTorch module that evaluates SDF(x) -> scalar
            return_cache: If True, return additional data for fast re-extraction

        Returns:
            vertices: (V, 3) tensor of vertex positions
            faces: (F, 3) tensor of triangle indices
            (optional) cache: Dict with grid and SDF values for fast updates
        """
        # Use fixed grid if provided, else create on-the-fly
        if self._grid_points is not None and self._resolution is not None:
            points_tensor = self._grid_points
            res = self._resolution
        else:
            # Create sampling grid on-the-fly (fallback)
            x = torch.linspace(self.bbox[0], self.bbox[1], self.resolution, device=self.device)
            y = torch.linspace(self.bbox[2], self.bbox[3], self.resolution, device=self.device)
            z = torch.linspace(self.bbox[4], self.bbox[5], self.resolution, device=self.device)
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
            points_tensor = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
            res = self.resolution

        # Evaluate SDF in batches to avoid OOM
        batch_size = 2048
        sdf_values = []

        with torch.no_grad():
            for i in range(0, points_tensor.shape[0], batch_size):
                batch = points_tensor[i:i+batch_size]
                sdf_batch = sdf_model(batch)
                sdf_values.append(sdf_batch.cpu())

        sdf_values = torch.cat(sdf_values, dim=0).cpu().numpy()
        sdf_grid = sdf_values.reshape(res, res, res)

        # Cache for potential re-extraction
        if return_cache:
            self._cached_grid = points_tensor
            self._cached_sdf = sdf_grid

        # Run Marching Cubes
        try:
            vertices, faces, _, _ = measure.marching_cubes(  # 3rd return value would be normals
                sdf_grid,
                level=self.isovalue,
                spacing=(
                    (self.bbox[1] - self.bbox[0]) / (res - 1),
                    (self.bbox[3] - self.bbox[2]) / (res - 1),
                    (self.bbox[5] - self.bbox[4]) / (res - 1),
                ),
            )

            # Offset vertices to correct position in bbox
            vertices[:, 0] += self.bbox[0]
            vertices[:, 1] += self.bbox[2]
            vertices[:, 2] += self.bbox[4]

        except (ValueError, RuntimeError) as e:
            warnings.warn(f"Marching Cubes failed: {e}. Returning empty mesh.", stacklevel=2)
            vertices = np.zeros((0, 3))
            faces = np.zeros((0, 3), dtype=np.int32)

        # Ensure arrays are contiguous to avoid negative stride crashes
        vertices = np.ascontiguousarray(vertices)
        faces = np.ascontiguousarray(faces)

        # Convert to tensors
        vertices_tensor = torch.from_numpy(vertices).float().to(self.device)
        faces_tensor = torch.from_numpy(faces).long().to(self.device)

        if return_cache:
            cache = {
                "grid_points": points_tensor,
                "sdf_grid": sdf_grid,
                "spacing": (
                    (self.bbox[1] - self.bbox[0]) / (res - 1),
                    (self.bbox[3] - self.bbox[2]) / (res - 1),
                    (self.bbox[5] - self.bbox[4]) / (res - 1),
                ),
            }
            return vertices_tensor, faces_tensor, cache

        return vertices_tensor, faces_tensor

    def compute_normals_at_vertices(self,
                                     vertices: torch.Tensor,
                                     sdf_model: nn.Module) -> torch.Tensor:
        """
        Compute surface normals at mesh vertices via SDF gradient.

        Normals are computed as: n = ∇_x SDF(x) / ||∇_x SDF(x)||

        Args:
            vertices: (V, 3) tensor of vertex positions
            sdf_model: SDF evaluation function

        Returns:
            (V, 3) tensor of unit normals at each vertex
        """
        vertices_grad = vertices.detach().clone().requires_grad_(True)

        # Evaluate SDF at vertices
        sdf_values = sdf_model(vertices_grad)

        # Compute gradient of SDF w.r.t. positions
        grad_outputs = torch.ones_like(sdf_values)
        normals = torch.autograd.grad(
            outputs=sdf_values,
            inputs=vertices_grad,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False
        )[0]

        # Normalize to unit vectors
        normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)

        return normals

    def compute_gradient_flow(
        self,
        vertices: torch.Tensor,
        dL_dvertices: torch.Tensor,  # noqa: N803
        sdf_model: nn.Module,
    ) -> torch.Tensor:
        """
        Compute gradient flow through the implicit surface (Equation 4).

        This implements the key MeshSDF innovation:
        dL/ds = -dL/dx · n

        where:
        - dL/dx: Upstream gradients from loss w.r.t. vertex positions
        - n: Surface normal (gradient of SDF at vertices)
        - dL/ds: "Inflow derivative" - gradient w.r.t. SDF values at vertices

        Args:
            vertices: (V, 3) tensor of vertex positions
            dL_dvertices: (V, 3) tensor of upstream gradients w.r.t. vertices
            sdf_model: SDF evaluation function

        Returns:
            Scalar loss term that enables gradient flow to SDF parameters
        """
        # Compute normals via SDF gradient
        vertices_grad = vertices.detach().clone().requires_grad_(True)
        sdf_values = sdf_model(vertices_grad)

        # Compute ∇_x SDF(x)
        grad_outputs = torch.ones_like(sdf_values)
        sdf_grad = torch.autograd.grad(
            outputs=sdf_values,
            inputs=vertices_grad,
            grad_outputs=grad_outputs,
            create_graph=True,  # Need graph for second backward
            retain_graph=True
        )[0]

        # Normalize to get normals
        sdf_grad_norm = torch.norm(sdf_grad, dim=1, keepdim=True) + 1e-8
        normals = sdf_grad / sdf_grad_norm

        # Compute inflow derivative: dL/ds = -dL/dx · n
        # Negative sign because moving along normal increases SDF
        dL_ds = -torch.sum(dL_dvertices * normals, dim=1)  # (V,)  # noqa: N806

        # Re-evaluate SDF with gradient tracking for backward pass
        vertices_for_backward = vertices.requires_grad_(True)
        sdf_values_backward = sdf_model(vertices_for_backward)

        # Create surrogate loss that couples dL/ds to SDF parameters
        # This allows gradients to flow: dL/dθ = sum_i (dL/ds_i * ds_i/dθ)
        surrogate_loss = torch.sum(dL_ds.detach() * sdf_values_backward)

        return surrogate_loss

    def extract_mesh_differentiable(self,
                                    sdf_model: nn.Module,
                                    enable_grad: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract mesh with gradient computation enabled.

        This is the main entry point for differentiable extraction.
        Use this when you need gradients to flow through the extraction.

        Args:
            sdf_model: PyTorch module with SDF evaluation
            enable_grad: If True, enable gradient computation

        Returns:
            vertices: (V, 3) tensor with gradients enabled
            faces: (F, 3) tensor (no gradients, topology is discrete)
        """
        if enable_grad:
            # Extract mesh (this uses torch.no_grad internally)
            vertices, faces = self.extract_mesh(sdf_model, return_cache=False)

            # Re-evaluate vertices with gradient tracking
            # This creates a computation graph for gradient flow
            vertices_differentiable = vertices.detach().requires_grad_(True)

            return vertices_differentiable, faces
        else:
            return self.extract_mesh(sdf_model, return_cache=False)

    def extract_and_compute_loss(self,
                                 sdf_model: nn.Module,
                                 loss_fn: callable,
                                 enable_backward: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract mesh, compute loss, and return gradient flow.

        Complete MeshSDF pipeline:
        1. Extract mesh using Marching Cubes
        2. Compute loss on mesh (e.g., surface area, rendering loss)
        3. Compute gradients w.r.t. vertices
        4. Project gradients through implicit surface to SDF parameters

        Args:
            sdf_model: SDF evaluation module
            loss_fn: Function that takes (vertices, faces) and returns scalar loss
            enable_backward: If True, compute backward pass

        Returns:
            vertices: (V, 3) extracted vertices
            faces: (F, 3) triangle indices
            loss: Scalar loss value
            surrogate_loss: Gradient flow term (for backward pass)
        """
        # Extract mesh
        vertices, faces = self.extract_mesh(sdf_model)

        if vertices.shape[0] == 0:
            # Empty mesh - return zero loss
            return vertices, faces, torch.tensor(0.0), torch.tensor(0.0)

        # Compute loss with gradient tracking for vertices
        vertices_for_loss = vertices.detach().requires_grad_(True)
        loss = loss_fn(vertices_for_loss, faces)

        if not enable_backward:
            return vertices, faces, loss, torch.tensor(0.0)

        # Compute upstream gradients
        dL_dvertices = torch.autograd.grad(  # noqa: N806
            outputs=loss, inputs=vertices_for_loss, grad_outputs=torch.ones_like(loss), create_graph=False, retain_graph=False
        )[0]

        # Compute gradient flow through implicit surface
        surrogate_loss = self.compute_gradient_flow(
            vertices=vertices,
            dL_dvertices=dL_dvertices,
            sdf_model=sdf_model
        )

        return vertices, faces, loss, surrogate_loss
