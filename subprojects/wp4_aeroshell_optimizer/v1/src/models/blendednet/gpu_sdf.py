"""
GPU-accelerated signed distance computation from triangle meshes using PyTorch.

Core idea: brute-force point-to-triangle distance over all triangles,
chunked to fit GPU memory. Sign via generalized winding number.
"""

import torch
import torch.nn.functional as F


def _point_to_triangle_unsigned_sq(query, v0, v1, v2):
    """
    Compute squared unsigned distance from query points to triangles.

    Args:
        query: [Q, 3] query points
        v0, v1, v2: [T, 3] triangle vertices

    Returns:
        dist_sq: [Q, T] squared distances
        closest: [Q, T, 3] closest points on each triangle
    """
    Q = query.shape[0]
    T = v0.shape[0]

    # Expand for broadcasting: [Q, 1, 3] vs [1, T, 3]
    p = query[:, None, :]     # [Q, 1, 3]
    a = v0[None, :, :]        # [1, T, 3]
    b = v1[None, :, :]        # [1, T, 3]
    c = v2[None, :, :]        # [1, T, 3]

    # Edge vectors
    ab = b - a                # [1, T, 3]
    ac = c - a                # [1, T, 3]
    ap = p - a                # [Q, T, 3]

    # Parametric coords via dot products
    d1 = (ab * ap).sum(-1)    # [Q, T]
    d2 = (ac * ap).sum(-1)    # [Q, T]
    d3 = (ab * ab).sum(-1)    # [1, T]
    d4 = (ab * ac).sum(-1)    # [1, T]
    d5 = (ac * ac).sum(-1)    # [1, T]

    bp = p - b                # [Q, T, 3]
    cp = p - c                # [Q, T, 3]
    d6 = (ab * bp).sum(-1)    # [Q, T]
    d7 = (ac * bp).sum(-1)    # [Q, T]
    d8 = (ab * cp).sum(-1)    # [Q, T]
    d9 = (ac * cp).sum(-1)    # [Q, T]

    # Barycentric region classification
    # Region: vertex A
    det = d3 * d5 - d4 * d4   # [1, T]
    s = d1 * d5 - d2 * d4     # [Q, T]
    t = d2 * d3 - d1 * d4     # [Q, T]

    # Start with projection onto triangle plane
    inv_det = 1.0 / (det + 1e-30)
    s_param = s * inv_det      # [Q, T]
    t_param = t * inv_det      # [Q, T]

    # Clamp to triangle using Ericson's method (Real-Time Collision Detection)
    # Check each Voronoi region

    # Vertex A region
    mask_va = (d1 <= 0) & (d2 <= 0)

    # Vertex B region
    mask_vb = (d6 >= 0) & (d7 <= d6)

    # Vertex C region
    mask_vc = (d9 >= 0) & (d8 <= d9)

    # Edge AB region
    vc = d1 * d7 - d6 * d2
    mask_eab = (vc <= 0) & (d1 >= 0) & (d6 <= 0)
    v_ab = d1 / (d1 - d6 + 1e-30)

    # Edge AC region
    vb = d8 * d2 - d1 * d9
    mask_eac = (vb <= 0) & (d2 >= 0) & (d9 <= 0)
    w_ac = d2 / (d2 - d9 + 1e-30)

    # Edge BC region
    va = d6 * d9 - d8 * d7
    mask_ebc = (va <= 0) & ((d7 - d6) >= 0) & ((d8 - d9) >= 0)
    w_bc = (d7 - d6) / ((d7 - d6) + (d8 - d9) + 1e-30)

    # Interior region (default)
    # Compute closest point for each region
    # Start with interior projection
    closest = a + s_param[..., None] * ab + t_param[..., None] * ac  # [Q, T, 3]

    # Override with edge/vertex projections where needed
    closest = torch.where(mask_va[..., None], a.expand(Q, T, 3), closest)
    closest = torch.where(mask_vb[..., None], b.expand(Q, T, 3), closest)
    closest = torch.where(mask_vc[..., None], c.expand(Q, T, 3), closest)
    closest = torch.where(mask_eab[..., None], a + v_ab[..., None] * ab, closest)
    closest = torch.where(mask_eac[..., None], a + w_ac[..., None] * ac, closest)
    closest = torch.where(mask_ebc[..., None], b + w_bc[..., None] * (c - b), closest)

    diff = p - closest  # [Q, T, 3]
    dist_sq = (diff * diff).sum(-1)  # [Q, T]

    return dist_sq, closest


def _winding_number(query, v0, v1, v2):
    """
    Compute generalized winding number for query points w.r.t. a triangle mesh.
    Uses the solid angle formula (Van Oosterom & Strackee, 1983).

    Args:
        query: [Q, 3]
        v0, v1, v2: [T, 3]

    Returns:
        winding: [Q] winding number (≈1 inside, ≈0 outside for closed mesh)
    """
    Q = query.shape[0]
    T = v0.shape[0]

    # Vectors from query to triangle vertices: [Q, T, 3]
    a = v0[None, :, :] - query[:, None, :]
    b = v1[None, :, :] - query[:, None, :]
    c = v2[None, :, :] - query[:, None, :]

    # Lengths
    la = a.norm(dim=-1, keepdim=True).clamp(min=1e-30)  # [Q, T, 1]
    lb = b.norm(dim=-1, keepdim=True).clamp(min=1e-30)
    lc = c.norm(dim=-1, keepdim=True).clamp(min=1e-30)

    # Normalize
    a = a / la
    b = b / lb
    c = c / lc

    la = la.squeeze(-1)  # [Q, T]
    lb = lb.squeeze(-1)
    lc = lc.squeeze(-1)

    # Solid angle = 2 * atan2(a·(b×c), |a||b||c| + (a·b)|c| + (b·c)|a| + (c·a)|b|)
    # But we already normalized, so |a|=|b|=|c|=1
    cross_bc = torch.cross(b, c, dim=-1)           # [Q, T, 3]
    numerator = (a * cross_bc).sum(-1)              # [Q, T]
    ab = (a * b).sum(-1)                            # [Q, T]
    bc = (b * c).sum(-1)                            # [Q, T]
    ca = (c * a).sum(-1)                            # [Q, T]
    denominator = 1.0 + ab + bc + ca                # [Q, T]

    solid_angle = 2.0 * torch.atan2(numerator, denominator)  # [Q, T]

    winding = solid_angle.sum(dim=-1) / (4.0 * torch.pi)  # [Q]
    return winding


def mesh_sdf(query, vertices, faces, chunk_size=2048):
    """
    Compute signed distance from query points to a triangle mesh on GPU.

    Args:
        query:    [Q, 3] query points (float32, on GPU)
        vertices: [V, 3] mesh vertices (float32, on GPU)
        faces:    [T, 3] face indices (long, on GPU)
        chunk_size: number of query points per chunk

    Returns:
        sdf: [Q] signed distance (negative inside, positive outside)
    """
    v0 = vertices[faces[:, 0]]  # [T, 3]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    Q = query.shape[0]
    T = faces.shape[0]

    min_dist_sq = torch.full((Q,), float('inf'), device=query.device)
    sign = torch.zeros(Q, device=query.device)

    for start in range(0, Q, chunk_size):
        end = min(start + chunk_size, Q)
        q_chunk = query[start:end]  # [chunk, 3]

        # Unsigned distance
        dist_sq, _ = _point_to_triangle_unsigned_sq(q_chunk, v0, v1, v2)  # [chunk, T]
        chunk_min_dist_sq, _ = dist_sq.min(dim=1)  # [chunk]
        min_dist_sq[start:end] = chunk_min_dist_sq

        # Winding number for sign
        winding = _winding_number(q_chunk, v0, v1, v2)  # [chunk]
        sign[start:end] = winding

    dist = torch.sqrt(min_dist_sq)
    # winding ≈ 1 inside, ≈ 0 outside → sign = -1 inside, +1 outside
    sdf = torch.where(sign > 0.5, -dist, dist)

    return sdf


def load_vtk_to_torch(vtk_path, device='cpu'):
    """Load a VTK file and return vertices + faces as torch tensors."""
    import numpy as np
    import pyvista as pv

    mesh = pv.read(vtk_path)
    surf = mesh.extract_surface() if not isinstance(mesh, pv.PolyData) else mesh
    # Ensure triangulated
    surf = surf.triangulate()

    vertices = torch.tensor(np.array(surf.points), dtype=torch.float32, device=device)

    # PyVista faces array: [n0, i0, i1, i2, n1, i3, i4, i5, ...]
    f = np.array(surf.faces).reshape(-1, 4)[:, 1:]  # strip the "3" prefix
    faces = torch.tensor(f, dtype=torch.long, device=device)

    return vertices, faces, surf
