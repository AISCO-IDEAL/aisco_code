"""
Differentiable pressure-field → drag integration.

Given per-vertex pressure predictions, surface normals, and a triangulated
mesh, this module computes drag force and drag coefficient in a fully
differentiable manner (compatible with torch.autograd).

Usage
-----
>>> drag  = pressure_to_drag(pred_pressure, normals, face, pos)
>>> cd    = drag_to_cd(drag, rho=1.225, u_inf=30.0, A_ref=2.16)
>>> grads = torch.autograd.grad(cd, pos)          # shape sensitivities

The integration follows:

    F_drag = Σ_i  p_i · (n_i · e_x) · A_i

where A_i is the Voronoi vertex area obtained by distributing each
triangle's area equally among its three vertices.
"""

import torch
from torch import Tensor


def compute_vertex_areas(pos: Tensor, face: Tensor) -> Tensor:
    """Compute per-vertex Voronoi areas from a triangle mesh.

    Each triangle's area is split equally among its three vertices
    (1/3 approximation of the true Voronoi region).

    Parameters
    ----------
    pos : Tensor [N, 3]
        Vertex coordinates.
    face : Tensor [3, F]
        Triangle face connectivity (vertex indices).

    Returns
    -------
    Tensor [N]
        Per-vertex area.
    """
    v0 = pos[face[0]]  # [F, 3]
    v1 = pos[face[1]]
    v2 = pos[face[2]]
    face_areas = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)  # [F]

    vertex_areas = torch.zeros(pos.shape[0], device=pos.device, dtype=pos.dtype)
    third = face_areas / 3.0
    vertex_areas.scatter_add_(0, face[0], third)
    vertex_areas.scatter_add_(0, face[1], third)
    vertex_areas.scatter_add_(0, face[2], third)
    return vertex_areas


def compute_face_normals(pos: Tensor, face: Tensor) -> Tensor:
    """Compute per-face unit normals for a triangle mesh.

    Parameters
    ----------
    pos : Tensor [N, 3]
    face : Tensor [3, F]

    Returns
    -------
    Tensor [F, 3]
        Unit normal per face.
    """
    v0 = pos[face[0]]
    v1 = pos[face[1]]
    v2 = pos[face[2]]
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)  # [F, 3]
    return cross / (cross.norm(dim=-1, keepdim=True) + 1e-12)


def pressure_to_drag(
    pred_pressure: Tensor,
    normals: Tensor,
    face: Tensor,
    pos: Tensor,
    drag_direction: int = 0,
) -> Tensor:
    """Differentiably integrate a per-vertex pressure field into drag force.

    Parameters
    ----------
    pred_pressure : Tensor [N, 1] or [N]
        Predicted (or ground-truth) pressure at each vertex.
    normals : Tensor [N, 3]
        Outward unit normals at each vertex (e.g. ``data.c``).
    face : Tensor [3, F]
        Triangle face connectivity.
    pos : Tensor [N, 3]
        Vertex coordinates.
    drag_direction : int
        Axis index for drag (0 = x, 1 = y, 2 = z). Default: 0 (x-axis).

    Returns
    -------
    Tensor (scalar)
        Drag force.  Differentiable w.r.t. ``pred_pressure`` and ``pos``.
    """
    p = pred_pressure.squeeze(-1) if pred_pressure.dim() > 1 else pred_pressure
    n_dir = normals[:, drag_direction]  # [N]
    vertex_areas = compute_vertex_areas(pos, face)  # [N]
    drag = (p * n_dir * vertex_areas).sum()
    return drag


def pressure_shear_to_drag(
    pred_pressure: Tensor,
    pred_shear: Tensor,
    normals: Tensor,
    face: Tensor,
    pos: Tensor,
    drag_direction: int = 0,
) -> Tensor:
    """Integrate both pressure and viscous shear stress into total drag.

    Total drag = pressure drag + friction drag.

    Parameters
    ----------
    pred_pressure : Tensor [N, 1] or [N]
        Per-vertex pressure.
    pred_shear : Tensor [N, 3]
        Per-vertex wall shear stress vector (tau_x, tau_y, tau_z).
    normals : Tensor [N, 3]
        Outward unit normals.
    face : Tensor [3, F]
        Triangle face connectivity.
    pos : Tensor [N, 3]
        Vertex coordinates.
    drag_direction : int
        Axis index for drag. Default: 0 (x-axis).

    Returns
    -------
    Tensor (scalar)
        Total drag force (pressure + friction).
    """
    p = pred_pressure.squeeze(-1) if pred_pressure.dim() > 1 else pred_pressure
    n_dir = normals[:, drag_direction]  # [N]
    tau_dir = pred_shear[:, drag_direction]  # [N]
    vertex_areas = compute_vertex_areas(pos, face)  # [N]

    pressure_drag = (p * n_dir * vertex_areas).sum()
    friction_drag = (tau_dir * vertex_areas).sum()
    return pressure_drag + friction_drag


def drag_to_cd(
    drag: Tensor,
    rho: float = 1.225,
    u_inf: float = 30.0,
    A_ref: float = 2.16,
) -> Tensor:
    """Convert drag force to drag coefficient.

    Parameters
    ----------
    drag : Tensor (scalar)
        Drag force.
    rho : float
        Freestream density [kg/m^3]. Default: 1.225 (air at sea level).
    u_inf : float
        Freestream velocity [m/s].
    A_ref : float
        Reference area [m^2].

    Returns
    -------
    Tensor (scalar)
        Drag coefficient Cd = F_drag / (0.5 * rho * u^2 * A_ref).
    """
    q = 0.5 * rho * u_inf**2 * A_ref
    return drag / q
