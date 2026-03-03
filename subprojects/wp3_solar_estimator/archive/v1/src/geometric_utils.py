"""Geometric utilities for 3D meshes:
- compute_face_normals: Calculate unit normal vectors for each face of a mesh.
- compute_triangle_areas: Calculate the area of each triangular face in a mesh.
Arguments:
    vertices (torch.Tensor): Tensor of shape (V, 3) representing vertex coordinates.
    faces (torch.Tensor): Tensor of shape (F, 3) representing indices of vertices forming each triangular face.
    eps (float): Small value to avoid division by zero in normal computation.
Returns:
    torch.Tensor: For compute_face_normals, a tensor of shape (F, 3) with unit normal vectors.
                  For compute_triangle_areas, a tensor of shape (F,) with areas of each triangle."""

import torch


def compute_face_normals(vertices: torch.Tensor, faces: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    v = vertices
    f = faces.to(dtype=torch.long, copy=False)

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0

    n = torch.cross(e1, e2, dim=1)
    norm = torch.linalg.norm(n, dim=1, keepdim=True).clamp_min(eps)
    n_unit = n / norm
    return n_unit


def compute_triangle_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    v = vertices
    f = faces.to(dtype=torch.long, copy=False)

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0

    cross_prod = torch.cross(e1, e2, dim=1)
    return 0.5 * torch.linalg.norm(cross_prod, dim=1)
