# ruff: noqa: RUF002, RUF003
"""
Differentiable mesh-based visibility via Möller-Trumbore ray-triangle intersection.

Computes per-face soft visibility v_i ∈ [0,1] by:
  1. Casting parallel rays from each face centre toward the sun
  2. Computing all ray-triangle intersections (vectorised Möller-Trumbore)
  3. Softmin over intersection distances to approximate the nearest occluder
  4. Sigmoid to smoothly decide whether that occluder is closer than the face itself

Fully differentiable w.r.t. vertex positions — gradients push geometry to reduce
self-shading and improve solar exposure.

Reference formulation (softmin + sigmoid):
    t_near_i  = -(1/β) log Σ_j exp(-β t_{i,j})
    occl_i    = σ(γ (t_face - t_near_i - m))
    v_i       = 1 - occl_i
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Vectorised Möller-Trumbore (batched rays × batched triangles)
# ---------------------------------------------------------------------------


def _moller_trumbore_batch(
    ray_origins: torch.Tensor,
    ray_dir: torch.Tensor,
    tri_v0: torch.Tensor,
    tri_v1: torch.Tensor,
    tri_v2: torch.Tensor,
    eps: float = 1e-8,
    miss_t: float = 1e9,
) -> torch.Tensor:
    """
    Batched Möller-Trumbore intersection for parallel rays against all triangles.

    All rays share the same direction (parallel sun rays).

    Args:
        ray_origins: (R, 3) origins of R rays
        ray_dir:     (3,)   shared ray direction (unit vector, toward surface, i.e. -sun_dir)
        tri_v0:      (T, 3) first vertex of T triangles
        tri_v1:      (T, 3) second vertex
        tri_v2:      (T, 3) third vertex
        eps:         tolerance for parallel-ray rejection
        miss_t:      large sentinel value for misses (must not influence softmin)

    Returns:
        t: (R, T) intersection distances.  miss_t where the ray misses.
    """
    R = ray_origins.shape[0]
    T = tri_v0.shape[0]

    # Edge vectors  (T, 3)
    edge1 = tri_v1 - tri_v0
    edge2 = tri_v2 - tri_v0

    # h = ray_dir × edge2   (T, 3)  — same direction for all rays
    h = torch.cross(ray_dir.unsqueeze(0).expand(T, -1), edge2, dim=-1)

    # a = edge1 · h   (T,)
    a = (edge1 * h).sum(dim=-1)

    # Rays parallel to the triangle → miss
    parallel_mask = a.abs() < eps  # (T,)

    inv_a = 1.0 / a.clamp(min=eps)  # safe division  (T,)

    # s is equal to origin - v0   (R, T, 3)
    s = ray_origins[:, None, :] - tri_v0[None, :, :]

    # u = (s · h) / a   (R, T)
    u = (s * h[None, :, :]).sum(dim=-1) * inv_a[None, :]

    # q = s × edge1   (R, T, 3)
    q = torch.cross(s, edge1[None, :, :].expand(R, -1, -1), dim=-1)

    # v = (ray_dir · q) / a   (R, T)
    v = (ray_dir[None, None, :] * q).sum(dim=-1) * inv_a[None, :]

    # t = (edge2 · q) / a   (R, T)
    t = (edge2[None, :, :] * q).sum(dim=-1) * inv_a[None, :]

    # Miss conditions: barycentric coords outside [0,1], t <= 0, parallel triangle
    miss = (u < 0.0) | (u > 1.0) | (v < 0.0) | ((u + v) > 1.0) | (t <= 0.0) | parallel_mask[None, :]

    # Replace misses with large sentinel so they don't affect softmin
    t = torch.where(miss, torch.tensor(miss_t, device=t.device, dtype=t.dtype), t)

    return t


# ---------------------------------------------------------------------------
# Normal computation helpers
# ---------------------------------------------------------------------------


def _normals_unit_check(normals: torch.Tensor | None) -> bool:
    """Check if provided normals are unit vectors.  If not, we will recompute them."""
    if normals is None:
        return False
    norms = normals.norm(dim=-1)
    return bool(torch.all((norms > 0) & (torch.abs(norms - 1.0) < 1e-3)).item())


def _normals_from_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Face normals from cross product of edges. (F, 3) unit normals."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    return cross / cross.norm(dim=-1, keepdim=True).clamp(min=eps)


def _normals_from_sdf(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    points: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Normals as the normalised SDF gradient at the given surface points.

    Uses torch.autograd so the SDF network must be differentiable.
    The returned normals are detached from the gradient-computation graph used
    internally but remain connected through `points` if `points` requires grad.
    """
    pts = points.detach().requires_grad_(True)
    sdf_vals = sdf_fn(pts)  # (Q,)
    grad_outputs = torch.ones_like(sdf_vals)
    (grads,) = torch.autograd.grad(sdf_vals, pts, grad_outputs=grad_outputs, create_graph=True)
    return grads / grads.norm(dim=-1, keepdim=True).clamp(min=eps)


def _compute_visibility_for_chunk(
    *,
    start: int,
    end: int,
    centroids: torch.Tensor,
    normals: torch.Tensor,
    sun: torch.Tensor,
    ray_dir: torch.Tensor,
    tri_v0: torch.Tensor,
    tri_v1: torch.Tensor,
    tri_v2: torch.Tensor,
    offset: torch.Tensor,
    L: torch.Tensor,
    beta: float,
    gamma: float,
    margin: torch.Tensor,
    miss_t: float,
    blocker_chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    chunk_centroids = centroids[start:end]  # (C, 3)
    chunk_normals = normals[start:end]  # (C, 3)

    # Ray origins: far along the sun direction
    ray_origins = chunk_centroids + offset * chunk_normals + L * sun  # (C, 3)

    # Distance from sun origin down to face centre = L  (for all rays)
    t_face = L

    # All-pairs intersection is computed as CxB blocks over occluder faces.
    # We then aggregate logsumexp blockwise to match the full CxF softmin.
    local_size = end - start
    global_indices = torch.arange(start, end, device=device)
    running_lse = torch.full((local_size,), float("-inf"), device=device, dtype=dtype)
    F_count = tri_v0.shape[0]

    for occluder_start in range(0, F_count, blocker_chunk_size):
        occluder_end = min(occluder_start + blocker_chunk_size, F_count)

        # Local CxB (or smaller edge block) intersection matrix.
        t_block = _moller_trumbore_batch(
            ray_origins,
            ray_dir,
            tri_v0[occluder_start:occluder_end],
            tri_v1[occluder_start:occluder_end],
            tri_v2[occluder_start:occluder_end],
            miss_t=miss_t,
        )

        # Mask out self-intersection only when the face index is in this block.
        self_in_block = (global_indices >= occluder_start) & (global_indices < occluder_end)
        if torch.any(self_in_block):
            row_idx = torch.nonzero(self_in_block, as_tuple=False).squeeze(-1)
            col_idx = global_indices[self_in_block] - occluder_start
            t_block[row_idx, col_idx] = miss_t

        block_lse = torch.logsumexp(-beta * t_block, dim=1)
        running_lse = torch.logaddexp(running_lse, block_lse)

    # Softmin over all triangles via accumulated logsumexp.
    t_near = -(1.0 / beta) * running_lse  # (C,)

    # Occlusion score: sigmoid(γ * (t_face - t_near - margin))
    occl = torch.sigmoid(gamma * (t_face - t_near - margin))  # (C,)
    return 1.0 - occl  # (C,)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mesh_visibility_v2(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    sun_dir: torch.Tensor,
    *,
    mesh_cfg: dict[str, Any],
    normals: torch.Tensor | None = None,
    sdf_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    normal_source: str = "mesh",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable per-face visibility using mesh ray-triangle intersections.

    For every mesh face, a ray is cast from a distant sun origin through the
    face centre.  Soft intersection logic (softmin + sigmoid) yields a smooth
    visibility value that is differentiable w.r.t. vertex positions.

    Args:
        vertices:           (V, 3) mesh vertex positions
        faces:              (F, 3) triangle face indices
        sun_dir:            (3,) unit vector **toward** the sun
        normals:            (F, 3) optional precomputed face normals (overrides normal_source)
        sdf_fn:             SDF callable (required when normal_source="sdf")
        normal_source:      "mesh" — normals from cross product of edges
                            "sdf"  — normals from ∇f evaluated at face centres
        mesh_cfg:           Mesh configuration dictionary with the following keys:
                             - "beta":               softmin sharpness (higher → closer to true min)
                             - "gamma":              sigmoid sharpness (higher → more binary shadow)
                             - "margin_factor":      self-occlusion margin as fraction of bbox diagonal
                             - "origin_offset_eps":  ray-origin offset along normal as fraction of bbox diagonal
                             - "sun_distance_factor": how far "behind the sun" rays originate (x bbox diagonal)
                             - "face_chunk_size":    number of faces processed per chunk (controls peak memory)
                             - "blocker_chunk_size": number of occluder faces processed per inner block
                                                    (defaults to face_chunk_size)
        miss_t:             sentinel distance for ray misses

    Returns:
        visibility:  (F,) soft visibility per face in [0, 1]
        normals:     (F, 3) unit face normals used
        centroids:   (F, 3) face centroids
    """
    print("Using 'mesh_visibility' version 2")
    if normal_source not in {"mesh", "sdf"}:
        raise ValueError(f"Invalid normal_source '{normal_source}'. Must be 'mesh' or 'sdf'.")
    beta = float(mesh_cfg["beta"])
    gamma = float(mesh_cfg["gamma"])
    margin_factor = float(mesh_cfg["margin_factor"])
    origin_offset_eps = float(mesh_cfg["origin_offset_eps"])
    sun_distance_factor = float(mesh_cfg["sun_distance_factor"])
    face_chunk_size = int(mesh_cfg["face_chunk_size"])
    blocker_chunk_size = int(mesh_cfg.get("blocker_chunk_size", face_chunk_size))
    miss_t = float(mesh_cfg["miss_t"])

    if blocker_chunk_size <= 0:
        raise ValueError("mesh_cfg['blocker_chunk_size'] must be > 0")

    device = vertices.device
    F_count = faces.shape[0]

    # -- Sun direction (unit) --------------------------------------------------
    sun = sun_dir.to(device)
    sun = sun / sun.norm().clamp(min=1e-12)
    ray_dir = -sun  # direction from sun toward surface

    # -- Face geometry ---------------------------------------------------------
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0  # (F, 3)

    # -- Normals ---------------------------------------------------------------
    if normal_source == "sdf":
        if sdf_fn is None:
            raise ValueError("sdf_fn is required when normal_source='sdf'")
        normals = _normals_from_sdf(sdf_fn, centroids)
    # check for mesh_sdf if normals are unit vectors
    elif normal_source == "mesh" and not _normals_unit_check(normals):
        normals = _normals_from_mesh(vertices, faces)

    if normals is None:
        raise ValueError("normals must be provided or computed before visibility chunking")

    # -- Bounding-box scale for adaptive parameters ----------------------------
    bbox_diag = (vertices.max(dim=0).values - vertices.min(dim=0).values).norm()
    margin = margin_factor * bbox_diag
    L = sun_distance_factor * bbox_diag
    offset = origin_offset_eps * bbox_diag

    # -- Triangle vertices (all triangles are potential occluders) --------------
    tri_v0, tri_v1, tri_v2 = v0, v1, v2  # (F, 3) each

    visibility_chunks: list[torch.Tensor] = []

    print(
        f"Computing visibility in chunks: ray faces={face_chunk_size}, blocker faces={blocker_chunk_size} (total {F_count} faces)"
    )

    for start in range(0, F_count, face_chunk_size):
        end = min(start + face_chunk_size, F_count)
        visibility_chunks.append(
            _compute_visibility_for_chunk(
                start=start,
                end=end,
                centroids=centroids,
                normals=normals,
                sun=sun,
                ray_dir=ray_dir,
                tri_v0=tri_v0,
                tri_v1=tri_v1,
                tri_v2=tri_v2,
                offset=offset,
                L=L,
                beta=beta,
                gamma=gamma,
                margin=margin,
                miss_t=miss_t,
                blocker_chunk_size=blocker_chunk_size,
                device=device,
                dtype=vertices.dtype,
            )
        )

    visibility = torch.cat(visibility_chunks, dim=0)  # (F,)

    return visibility, normals, centroids
