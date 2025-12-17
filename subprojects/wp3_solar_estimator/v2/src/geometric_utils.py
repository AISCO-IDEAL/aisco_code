import torch


def compute_face_normals(vertices: torch.Tensor, faces: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute unit normals for each triangular face using PyTorch (kept on device).

    Args:
        vertices: (V, 3) float tensor of vertex positions (device can be cuda/cpu).
        faces: (F, 3) long/int tensor of vertex indices per face.
        eps: small value to avoid division by zero.

    Returns:
        (F, 3) float tensor of normalized face normals.
    """
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
    """Compute areas for each triangular face using PyTorch (kept on device).

    Args:
        vertices: (V, 3) float tensor of vertex positions.
        faces: (F, 3) long/int tensor of vertex indices per face.

    Returns:
        (F,) float tensor of triangle areas.
    """
    v = vertices
    f = faces.to(dtype=torch.long, copy=False)

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0

    cross_prod = torch.cross(e1, e2, dim=1)
    return 0.5 * torch.linalg.norm(cross_prod, dim=1)
