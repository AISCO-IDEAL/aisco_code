# ruff: noqa: RUF003
"""
Differentiable mesh visibility (v3) for solar shading.

This module keeps a legacy v2-compatible mode and adds a smoother mode designed
for optimization:

1. soft_barycentric_transmittance (default in v3)
     - Computes soft hit confidence from barycentric coordinates and depth sign.
     - Accumulates weighted transmittance over all potential blockers.
     - Supports multiple ray samples per face to smooth neighboring-triangle
         transitions.

2. v2_sigmoid (compatibility)
     - Preserves the v2-style softmin + sigmoid behavior.

Both modes are differentiable w.r.t. vertex positions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def _moller_trumbore_components(
    ray_origins: torch.Tensor,
    ray_dir: torch.Tensor,
    tri_v0: torch.Tensor,
    tri_v1: torch.Tensor,
    tri_v2: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized Moller-Trumbore components for parallel rays.

    Returns:
        t: (R, T) intersection depth parameters
        u: (R, T) first barycentric coordinate
        v: (R, T) second barycentric coordinate
        parallel_mask: (T,) triangles whose determinant is near zero
    """
    R = ray_origins.shape[0]
    T = tri_v0.shape[0]

    edge1 = tri_v1 - tri_v0  # (T, 3)
    edge2 = tri_v2 - tri_v0  # (T, 3)

    h = torch.cross(ray_dir.unsqueeze(0).expand(T, -1), edge2, dim=-1)  # (T, 3)
    a = (edge1 * h).sum(dim=-1)  # (T,)
    parallel_mask = a.abs() < eps

    # Preserve sign while avoiding singular division.
    denom = torch.where(parallel_mask, torch.sign(a) * eps, a)
    denom = torch.where(denom == 0.0, torch.full_like(denom, eps), denom)
    inv_a = 1.0 / denom

    s = ray_origins[:, None, :] - tri_v0[None, :, :]  # (R, T, 3)
    u = (s * h[None, :, :]).sum(dim=-1) * inv_a[None, :]

    q = torch.cross(s, edge1[None, :, :].expand(R, -1, -1), dim=-1)
    v = (ray_dir[None, None, :] * q).sum(dim=-1) * inv_a[None, :]
    t = (edge2[None, :, :] * q).sum(dim=-1) * inv_a[None, :]

    return t, u, v, parallel_mask


def _moller_trumbore_t_hardmask(
    ray_origins: torch.Tensor,
    ray_dir: torch.Tensor,
    tri_v0: torch.Tensor,
    tri_v1: torch.Tensor,
    tri_v2: torch.Tensor,
    eps: float,
    miss_t: float,
) -> torch.Tensor:
    """Legacy hard hit-mask t matrix for v2-compatible mode."""
    t, u, v, parallel_mask = _moller_trumbore_components(
        ray_origins=ray_origins,
        ray_dir=ray_dir,
        tri_v0=tri_v0,
        tri_v1=tri_v1,
        tri_v2=tri_v2,
        eps=eps,
    )
    miss = (u < 0.0) | (u > 1.0) | (v < 0.0) | ((u + v) > 1.0) | (t <= 0.0) | parallel_mask[None, :]
    return torch.where(miss, torch.tensor(miss_t, device=t.device, dtype=t.dtype), t)


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


def _sample_face_barycentric(
    face_samples: int,
    *,
    sample_pattern: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build deterministic barycentric samples for each face.

    The first sample is always the centroid to preserve behavior when reducing
    to a single sample.
    """
    if face_samples <= 1 or sample_pattern == "centroid":
        return torch.tensor([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], device=device, dtype=dtype)

    base = torch.tensor(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.3, 0.0],
            [0.7, 0.0, 0.3],
            [0.3, 0.7, 0.0],
            [0.0, 0.7, 0.3],
            [0.3, 0.0, 0.7],
            [0.0, 0.3, 0.7],
        ],
        device=device,
        dtype=dtype,
    )

    if face_samples <= base.shape[0]:
        return base[:face_samples]

    repeats = (face_samples + base.shape[0] - 1) // base.shape[0]
    tiled = base.repeat(repeats, 1)
    return tiled[:face_samples]


def _face_sample_points(
    v0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    bary: torch.Tensor,
) -> torch.Tensor:
    """Create sampled points per face from barycentric coordinates."""
    a = bary[:, 0][None, :, None]
    b = bary[:, 1][None, :, None]
    c = bary[:, 2][None, :, None]
    return a * v0[:, None, :] + b * v1[:, None, :] + c * v2[:, None, :]


def _compute_visibility_for_chunk_v2(
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
    """Legacy v2-compatible softmin + sigmoid visibility for one chunk."""
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
        t_block = _moller_trumbore_t_hardmask(
            ray_origins,
            ray_dir,
            tri_v0[occluder_start:occluder_end],
            tri_v1[occluder_start:occluder_end],
            tri_v2[occluder_start:occluder_end],
            eps=1e-8,
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


def _compute_visibility_for_chunk_soft(
    *,
    start: int,
    end: int,
    sampled_points: torch.Tensor,
    normals: torch.Tensor,
    sun: torch.Tensor,
    ray_dir: torch.Tensor,
    tri_v0: torch.Tensor,
    tri_v1: torch.Tensor,
    tri_v2: torch.Tensor,
    offset: torch.Tensor,
    L: torch.Tensor,
    depth_scale: torch.Tensor,
    transmittance_beta: float,
    transmittance_kappa: float,
    bary_softness_tau: float,
    t_softness_tau: float,
    blocker_chunk_size: int,
    device: torch.device,
    eps: float,
) -> torch.Tensor:
    """
    Soft barycentric + transmittance visibility for one chunk.

    Returns:
        (C,) face visibility values in [0, 1].
    """
    C = end - start
    K = sampled_points.shape[1]

    chunk_points = sampled_points[start:end]  # (C, K, 3)
    chunk_normals = normals[start:end]  # (C, 3)

    ray_origins = chunk_points + offset * chunk_normals[:, None, :] + L * sun  # (C, K, 3)
    ray_origins_flat = ray_origins.reshape(C * K, 3)
    ray_face_indices = torch.arange(start, end, device=device).repeat_interleave(K)  # (C*K,)

    R = ray_origins_flat.shape[0]
    occlusion_mass = torch.zeros((R,), device=device, dtype=tri_v0.dtype)
    F_count = tri_v0.shape[0]

    for occluder_start in range(0, F_count, blocker_chunk_size):
        occluder_end = min(occluder_start + blocker_chunk_size, F_count)

        t_block, u_block, v_block, parallel_mask = _moller_trumbore_components(
            ray_origins=ray_origins_flat,
            ray_dir=ray_dir,
            tri_v0=tri_v0[occluder_start:occluder_end],
            tri_v1=tri_v1[occluder_start:occluder_end],
            tri_v2=tri_v2[occluder_start:occluder_end],
            eps=eps,
        )

        w_u = torch.sigmoid(bary_softness_tau * u_block)
        w_v = torch.sigmoid(bary_softness_tau * v_block)
        w_w = torch.sigmoid(bary_softness_tau * (1.0 - u_block - v_block))
        w_t = torch.sigmoid(t_softness_tau * t_block)
        w_parallel = (~parallel_mask).to(dtype=tri_v0.dtype)[None, :]
        w_hit = w_u * w_v * w_w * w_t * w_parallel

        # Remove self-intersection contribution for all samples of the source face.
        self_in_block = (ray_face_indices >= occluder_start) & (ray_face_indices < occluder_end)
        if torch.any(self_in_block):
            row_idx = torch.nonzero(self_in_block, as_tuple=False).squeeze(-1)
            col_idx = ray_face_indices[self_in_block] - occluder_start
            w_hit[row_idx, col_idx] = 0.0

        t_pos = torch.clamp(t_block, min=0.0)
        t_norm = t_pos / depth_scale
        depth_weight = torch.exp(-transmittance_beta * t_norm)

        occlusion_mass = occlusion_mass + torch.sum(w_hit * depth_weight, dim=1)

    vis_rays = torch.exp(-transmittance_kappa * occlusion_mass)  # (C*K,)
    vis_faces = vis_rays.reshape(C, K).mean(dim=1)
    return vis_faces


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mesh_visibility_v3(  # noqa: PLR0912, PLR0915 #sadly it's a big mess.
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

    v3 supports two mesh smoothing modes selected by mesh_cfg["smoothing_mode"]:

    1) "soft_barycentric_transmittance" (default):
       - softens hit logic with barycentric/depth sigmoids
       - combines blockers with transmittance accumulation
       - supports multiple ray samples per face for smoother neighboring faces

    2) "v2_sigmoid":
       - compatibility mode reproducing v2-style softmin + sigmoid behavior.

    Args:
        vertices:           (V, 3) mesh vertex positions
        faces:              (F, 3) triangle face indices
        sun_dir:            (3,) unit vector **toward** the sun
        normals:            (F, 3) optional precomputed face normals (overrides normal_source)
        sdf_fn:             SDF callable (required when normal_source="sdf")
        normal_source:      "mesh" — normals from cross product of edges
                            "sdf"  — normals from ∇f evaluated at face centres
        mesh_cfg:           mesh configuration dictionary.
                    Required in all modes:
                     - "origin_offset_eps", "sun_distance_factor", "face_chunk_size"
                    Optional mode selector:
                     - "smoothing_mode": "soft_barycentric_transmittance" | "v2_sigmoid"

                    v2_sigmoid keys:
                     - "beta", "gamma", "margin_factor", "miss_t"

                    soft_barycentric_transmittance keys (optional with defaults):
                     - "face_samples" (default 1)
                     - "sample_pattern" (default "centroid")
                     - "depth_normalization" ("bbox" | "ray_length", default "bbox")
                     - "transmittance_beta" (default 2.0)
                     - "transmittance_kappa" (default 1.0)
                     - "bary_softness_tau" (default 12.0)
                     - "t_softness_tau" (default 8.0)
                     - "eps" (default 1e-8)
                     - "miss_t" (accepted for compatibility, not used)

    Returns:
        visibility:  (F,) soft visibility per face in [0, 1]
        normals:     (F, 3) unit face normals used
        centroids:   (F, 3) face centroids
    """
    print("Using 'mesh_visibility' version 3")
    if normal_source not in {"mesh", "sdf"}:
        raise ValueError(f"Invalid normal_source '{normal_source}'. Must be 'mesh' or 'sdf'.")

    smoothing_mode = str(mesh_cfg.get("smoothing_mode", "soft_barycentric_transmittance")).lower()
    origin_offset_eps = float(mesh_cfg["origin_offset_eps"])
    sun_distance_factor = float(mesh_cfg["sun_distance_factor"])
    face_chunk_size = int(mesh_cfg["face_chunk_size"])
    blocker_chunk_size = int(mesh_cfg.get("blocker_chunk_size", face_chunk_size))
    miss_t = float(mesh_cfg.get("miss_t", 1e9))
    eps = float(mesh_cfg.get("eps", 1e-8))

    # v2 compatibility keys are only needed in v2_sigmoid mode.
    beta = float(mesh_cfg.get("beta", 1.0))
    gamma = float(mesh_cfg.get("gamma", 20.0))
    margin_factor = float(mesh_cfg.get("margin_factor", 1e-3))

    # v3 smoother-mode parameters.
    face_samples = int(mesh_cfg.get("face_samples", 1))
    sample_pattern = str(mesh_cfg.get("sample_pattern", "centroid")).lower()
    depth_normalization = str(mesh_cfg.get("depth_normalization", "bbox")).lower()
    transmittance_beta = float(mesh_cfg.get("transmittance_beta", 2.0))
    transmittance_kappa = float(mesh_cfg.get("transmittance_kappa", 1.0))
    bary_softness_tau = float(mesh_cfg.get("bary_softness_tau", 12.0))
    t_softness_tau = float(mesh_cfg.get("t_softness_tau", 8.0))

    if blocker_chunk_size <= 0:
        raise ValueError("mesh_cfg['blocker_chunk_size'] must be > 0")
    if face_samples <= 0:
        raise ValueError("mesh_cfg['face_samples'] must be > 0")
    if smoothing_mode not in {"soft_barycentric_transmittance", "v2_sigmoid"}:
        raise ValueError("mesh_cfg['smoothing_mode'] must be 'soft_barycentric_transmittance' or 'v2_sigmoid'")
    if sample_pattern not in {"centroid", "barycentric_ring"}:
        raise ValueError("mesh_cfg['sample_pattern'] must be 'centroid' or 'barycentric_ring'")
    if depth_normalization not in {"bbox", "ray_length"}:
        raise ValueError("mesh_cfg['depth_normalization'] must be 'bbox' or 'ray_length'")

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

    depth_scale = bbox_diag if depth_normalization == "bbox" else L
    depth_scale = depth_scale.clamp(min=1e-12)

    # -- Triangle vertices (all triangles are potential occluders) --------------
    tri_v0, tri_v1, tri_v2 = v0, v1, v2  # (F, 3) each

    # -- Optional multi-sample face points for v3 smooth mode -------------------
    bary = _sample_face_barycentric(
        face_samples=face_samples,
        sample_pattern=sample_pattern,
        device=device,
        dtype=vertices.dtype,
    )
    sampled_points = _face_sample_points(v0, v1, v2, bary)

    visibility_chunks: list[torch.Tensor] = []

    print(f"Computing visibility in chunks: rays={face_chunk_size}, blockers={blocker_chunk_size}, faces={F_count}")

    for start in range(0, F_count, face_chunk_size):
        end = min(start + face_chunk_size, F_count)
        if smoothing_mode == "v2_sigmoid":
            chunk_visibility = _compute_visibility_for_chunk_v2(
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
        else:
            chunk_visibility = _compute_visibility_for_chunk_soft(
                start=start,
                end=end,
                sampled_points=sampled_points,
                normals=normals,
                sun=sun,
                ray_dir=ray_dir,
                tri_v0=tri_v0,
                tri_v1=tri_v1,
                tri_v2=tri_v2,
                offset=offset,
                L=L,
                depth_scale=depth_scale,
                transmittance_beta=transmittance_beta,
                transmittance_kappa=transmittance_kappa,
                bary_softness_tau=bary_softness_tau,
                t_softness_tau=t_softness_tau,
                blocker_chunk_size=blocker_chunk_size,
                device=device,
                eps=eps,
            )
        visibility_chunks.append(chunk_visibility)

    visibility = torch.cat(visibility_chunks, dim=0)  # (F,)

    return visibility, normals, centroids


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
    """Backward-compatible alias to v3 implementation."""
    return mesh_visibility_v3(
        vertices=vertices,
        faces=faces,
        sun_dir=sun_dir,
        mesh_cfg=mesh_cfg,
        normals=normals,
        sdf_fn=sdf_fn,
        normal_source=normal_source,
    )
