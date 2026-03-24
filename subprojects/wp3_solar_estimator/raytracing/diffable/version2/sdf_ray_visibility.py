# ruff: noqa: RUF002, RUF003
"""
Differentiable SDF-based visibility via ray marching.

Computes per-face soft visibility v_i ∈ [0,1] by marching fixed-size steps
along the sun direction and accumulating a smooth transmittance through the
signed distance field.

Two formulations are provided (selectable via ``method``):

1. **sigmoid** (default) — product-of-sigmoids transmittance:
       V_i = Π_k σ(α · f(x_k))
   computed in log-space for numerical stability.

2. **absorption** — Beer-Lambert style volumetric occlusion:
       O_i = Σ_k exp(-β · f(x_k)) · Δt
       V_i = exp(-κ · O_i)
   smoother gradients, slightly less sharp shadows.

Both are fully differentiable through the SDF network weights θ.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Normal computation helpers
# ---------------------------------------------------------------------------


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
    """
    pts = points.detach().requires_grad_(True)
    sdf_vals = sdf_fn(pts)  # (Q,)
    grad_outputs = torch.ones_like(sdf_vals)
    (grads,) = torch.autograd.grad(sdf_vals, pts, grad_outputs=grad_outputs, create_graph=True)
    return grads / grads.norm(dim=-1, keepdim=True).clamp(min=eps)


# ---------------------------------------------------------------------------
# Core: fixed-step SDF ray march
# ---------------------------------------------------------------------------


def _march_sigmoid(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    origins: torch.Tensor,
    sun_dir: torch.Tensor,
    num_steps: int,
    step_size: float,
    alpha: float,
    point_chunk_size: int,
) -> torch.Tensor:
    """
    Product-of-sigmoids transmittance (log-space accumulation).

    Returns visibility (R,) in [0, 1].
    """
    R = origins.shape[0]
    device = origins.device
    log_vis = torch.zeros(R, device=device, dtype=origins.dtype)

    x = origins.clone()  # (R, 3)

    for _k in range(num_steps):
        # Evaluate SDF in chunks to limit peak memory
        sdf_vals_list: list[torch.Tensor] = []
        for i in range(0, R, point_chunk_size):
            chunk = x[i : i + point_chunk_size]
            sdf_vals_list.append(sdf_fn(chunk))
        d = torch.cat(sdf_vals_list, dim=0)  # (R,)

        # log σ(α·d) — use log_sigmoid for numerical stability
        log_vis = log_vis + torch.nn.functional.logsigmoid(alpha * d)

        # Fixed step along sun direction
        x = x + step_size * sun_dir.unsqueeze(0)

    return log_vis.exp()  # (R,)


def _march_absorption(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    origins: torch.Tensor,
    sun_dir: torch.Tensor,
    num_steps: int,
    step_size: float,
    beta: float,
    kappa: float,
    point_chunk_size: int,
) -> torch.Tensor:
    """
    Beer-Lambert volumetric absorption transmittance.

    Returns visibility (R,) in [0, 1].
    """
    R = origins.shape[0]
    device = origins.device
    occlusion = torch.zeros(R, device=device, dtype=origins.dtype)

    x = origins.clone()

    for _k in range(num_steps):
        sdf_vals_list: list[torch.Tensor] = []
        for i in range(0, R, point_chunk_size):
            chunk = x[i : i + point_chunk_size]
            sdf_vals_list.append(sdf_fn(chunk))
        d = torch.cat(sdf_vals_list, dim=0)  # (R,)

        # density = exp(-β · d) ;  high near the surface (d ≈ 0)
        occlusion = occlusion + torch.exp(-beta * d) * step_size

        x = x + step_size * sun_dir.unsqueeze(0)

    return torch.exp(-kappa * occlusion)  # (R,)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sdf_visibility(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    sun_dir: torch.Tensor,
    *,
    normal_source: str = "mesh",  # expects "mesh" or "sdf"
    method: str = "sigmoid",  # expects "sigmoid" or "absorption"
    # Sigmoid method params
    sdf_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable per-face visibility via SDF ray marching.

    For every mesh face, march along the sun direction starting just above
    the surface and accumulate a soft transmittance by querying the SDF.

    Args:
        sdf_fn:              SDF callable: (Q, 3) → (Q,)
        vertices:            (V, 3) mesh vertex positions
        faces:               (F, 3) triangle face indices
        sun_dir:             (3,) unit vector **toward** the sun
        normal_source:       "mesh" — normals from cross product of edges
                             "sdf"  — normals from ∇f evaluated at face centres
        method:              "sigmoid"    — Π σ(α f(x_k))  (sharper)
                             "absorption" — exp(-κ Σ exp(-β f(x_k)) Δt)  (smoother)
        sdf_cfg:             SDF configuration dictionary with the following keys:
                             - "alpha": sigmoid softness (sigmoid method)
                             - "beta": density sharpness (absorption method)
                             - "kappa": absorption coefficient (absorption method)
                             - "num_steps": number of march steps
                             - "step_size_factor": step size as fraction of bounding-box diagonal
                             - "origin_offset_factor": initial offset along normal as fraction of bbox diagonal
        max_march_factor:    stop marching after this × bbox diagonal
        point_chunk_size:    SDF query batch size (controls GPU memory)

    Returns:
        visibility:  (F,) soft visibility per face in [0, 1]
        normals:     (F, 3) unit face normals used
        centroids:   (F, 3) face centroids
    """

    print("Using 'sdf_visibility' version 2")

    assert sdf_fn is not None, "sdf_fn cannot be None for sdf_ray_visibility computation."

    alpha = float(sdf_cfg["alpha"])
    beta = float(sdf_cfg["beta"])
    kappa = float(sdf_cfg["kappa"])
    num_steps = int(sdf_cfg["num_steps"])
    step_size_factor = float(sdf_cfg["step_size_factor"])
    origin_offset_factor = float(sdf_cfg["origin_offset_factor"])
    max_march_factor = float(sdf_cfg["max_march_factor"])
    point_chunk_size = int(sdf_cfg["point_chunk_size"])

    device = vertices.device

    # -- Sun direction (unit) --------------------------------------------------
    sun = sun_dir.to(device)
    sun = sun / sun.norm().clamp(min=1e-12)

    # -- Face geometry ---------------------------------------------------------
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0  # (F, 3)

    # -- Normals ---------------------------------------------------------------
    normals = _normals_from_sdf(sdf_fn, centroids) if normal_source == "sdf" else _normals_from_mesh(vertices, faces)

    # -- Bounding-box scale ----------------------------------------------------
    bbox_diag = (vertices.max(dim=0).values - vertices.min(dim=0).values).norm()
    step_size = step_size_factor * bbox_diag
    offset = origin_offset_factor * bbox_diag
    max_dist = max_march_factor * bbox_diag

    # Clamp num_steps so we don't march past the bounding region
    effective_steps = min(num_steps, max(1, int((max_dist / step_size).item())))

    # -- Ray origins: slightly above the surface along the normal ---------------
    origins = centroids + offset * normals  # (F, 3)

    # -- March -----------------------------------------------------------------
    if method == "sigmoid":
        visibility = _march_sigmoid(sdf_fn, origins, sun, effective_steps, step_size, alpha, point_chunk_size)
    elif method == "absorption":
        visibility = _march_absorption(sdf_fn, origins, sun, effective_steps, step_size, beta, kappa, point_chunk_size)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'sigmoid' or 'absorption'.")

    return visibility, normals, centroids
