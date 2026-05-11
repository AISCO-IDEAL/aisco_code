# ruff: noqa: RUF003
"""
Differentiable SDF-based visibility via ray marching (version 3).

Fixes over v2:
- Ray origin is offset along the sun direction (not the face normal), so the
  starting point is always ``offset * bbox_diag`` clear of the surface in the
  exact direction of travel regardless of face orientation.
- The first SDF evaluation is taken one step *away* from the origin, not at the
  origin itself, eliminating the spurious per-face occlusion penalty that v2
  incurred right at the surface.
- A warning is emitted when ``max_march_factor / step_size_factor`` clamps
  ``effective_steps`` below the requested ``num_steps``.

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

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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

    # Start one step away from the origin so the first eval is not at the surface.
    x = origins + step_size * sun_dir.unsqueeze(0)  # (R, 3)

    for _k in range(num_steps):
        # Evaluate SDF in chunks to limit peak memory
        sdf_vals_list: list[torch.Tensor] = []
        for i in range(0, R, point_chunk_size):
            chunk = x[i : i + point_chunk_size]
            sdf_vals_list.append(sdf_fn(chunk))
        d = torch.cat(sdf_vals_list, dim=0)  # (R,)

        # log σ(α·d) — use log_sigmoid for numerical stability
        log_vis += torch.nn.functional.logsigmoid(alpha * d)

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

    # Start one step away from the origin so the first eval is not at the surface.
    x = origins + step_size * sun_dir.unsqueeze(0)

    for _k in range(num_steps):
        sdf_vals_list: list[torch.Tensor] = []
        for i in range(0, R, point_chunk_size):
            chunk = x[i : i + point_chunk_size]
            sdf_vals_list.append(sdf_fn(chunk))
        d = torch.cat(sdf_vals_list, dim=0)  # (R,)

        # density = exp(-β · d) ;  high near the surface (d ≈ 0)
        occlusion += torch.exp(-beta * d) * step_size

        x = x + step_size * sun_dir.unsqueeze(0)

    return torch.exp(-kappa * occlusion)  # (R,)


def _sample_sdf_along_z_axis(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    offset: float,
    num_points: int = 1000,
    bbox_width: float | None = None,
    bbox_length: float | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample the SDF along the z-axis for debugging.

    Args:
        sdf_fn: SDF callable: (Q, 3) → (Q,)
        offset: initial offset along z-axis
        num_points: number of points to sample

    Returns:
        xs: (G,) x-coordinates of the sampled grid at given z
        ys: (G,) y-coordinates of the sampled grid at given z
        sdf_vals: (G, G) SDF values sampled on the xy grid at given z
    """
    if bbox_width is None or bbox_length is None:
        raise ValueError("bbox_width and bbox_length must be provided for sampling SDF along z-axis.")

    sample_device = device if device is not None else torch.device("cpu")
    grid_size = max(2, int(np.sqrt(num_points)))
    xs = torch.linspace(-bbox_width / 2, bbox_width / 2, grid_size, device=sample_device)
    ys = torch.linspace(-bbox_length / 2, bbox_length / 2, grid_size, device=sample_device)

    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    grid_z = torch.full_like(grid_x, offset)
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=-1)
    sdf_vals = sdf_fn(points).reshape(grid_size, grid_size)
    return xs, ys, sdf_vals


def _plot_function(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    vertices: torch.Tensor,
    device: torch.device | None = None,
    debug: int = 1,
    save_dir: Path | None = None,
) -> None:
    """
    Plot the SDF values along the z-axis for debugging.

    Args:
        sdf_fn: SDF callable: (Q, 3) → (Q,)
        vertices: (V, 3) mesh vertex positions to determine bounding box for sampling
        device: device to use for sampling
        debug: number of debug layers to plot
    """

    if save_dir is not None and debug > 0:
        output_dir = save_dir / "sdf_rt_debug"
    else:
        raise ValueError("save_dir must be provided and debug must be > 0 to plot SDF values.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # bounding box height determined from bbox_diag
    bbox_min = vertices.min(dim=0).values
    bbox_max = vertices.max(dim=0).values
    # ensure bbox dimensions are float and non-zero to avoid issues with sampling
    bbox_width = float((bbox_max[0] - bbox_min[0]).clamp(min=1e-12))
    bbox_length = float((bbox_max[1] - bbox_min[1]).clamp(min=1e-12))
    bbox_height = float((bbox_max[2] - bbox_min[2]).clamp(min=1e-12))
    # step size should be bbox height / debug, so we sample exactly debug layers until max_march_factor
    step_size = bbox_height / (debug + 1)
    for i in range(1, debug + 1):
        sample = _sample_sdf_along_z_axis(
            sdf_fn, i * step_size, num_points=1000, bbox_width=bbox_width, bbox_length=bbox_length, device=device
        )
        # Debug plotting should not keep autograd history.
        x_coords = sample[0].detach().cpu().numpy()
        y_coords = sample[1].detach().cpu().numpy()
        sdf_vals = sample[2].detach().cpu().numpy()
        extent: tuple[float, float, float, float] = (
            float(np.min(x_coords)),
            float(np.max(x_coords)),
            float(np.min(y_coords)),
            float(np.max(y_coords)),
        )
        fig = plt.imshow(
            sdf_vals,
            extent=extent,
            origin="lower",
            interpolation="bicubic",
        )
        fig.axes.set_autoscale_on(False)
        sample_x, sample_y = np.meshgrid(x_coords, y_coords, indexing="xy")
        plt.scatter(sample_x.ravel(), sample_y.ravel(), 10, facecolors="none", linewidths=0.2)
        plt.colorbar()
        plt.savefig(output_dir / f"sdf_debug_layer_{i:03d}.png")
        plt.close()

    print(f"Saved SDF debug plots to {output_dir}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sdf_visibility_v3(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    vertices: torch.Tensor,
    faces: torch.Tensor,
    sun_dir: torch.Tensor,
    *,
    save_dir: Path | None = None,
    normal_source: str = "mesh",
    sdf_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable per-face visibility via SDF ray marching.

    For every mesh face, march along the sun direction starting just above
    the surface and accumulate a soft transmittance by querying the SDF.

    Args:
        sdf_fn:        SDF callable: (Q, 3) → (Q,)
        vertices:      (V, 3) mesh vertex positions
        faces:         (F, 3) triangle face indices
        sun_dir:       (3,) unit vector **toward** the sun
        normal_source: "mesh" — normals from cross product of edges
                       "sdf"  — normals from ∇f evaluated at face centres
        sdf_cfg:       SDF configuration dictionary with the following keys:
                       - "method": "sigmoid" or "absorption"
                       - "alpha": sigmoid softness (sigmoid method)
                       - "beta": density sharpness (absorption method)
                       - "kappa": absorption coefficient (absorption method)
                       - "num_steps": number of march steps
                       - "step_size_factor": step size as fraction of bounding-box diagonal
                       - "origin_offset_factor": initial offset along sun dir as fraction of bbox diagonal
                       - "max_march_factor": stop marching after this × bbox diagonal
                       - "point_chunk_size": SDF query batch size (controls GPU memory)
                       - "debug": number of debug SDF layers to plot (0 = off)

    Returns:
        visibility:  (F,) soft visibility per face in [0, 1]
        normals:     (F, 3) unit face normals used
        centroids:   (F, 3) face centroids
    """

    print("Using 'sdf_visibility' version 3")

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

    # -- Debug: sample SDF along z-axis ----------------------------------------
    if save_dir is None:
        debug = 0
        print("No save_dir provided, skipping SDF debug plots, in sdf_ray_visibility_v3")
    else:
        debug = int(sdf_cfg["debug"])
    if debug < 0:
        raise ValueError(f"Invalid debug value: {debug}. Must be non-negative integer.")

    if bool(debug):
        print("Debug mode is ON for sdfRT")
        _plot_function(sdf_fn, vertices, device, debug, save_dir)
    else:
        print("Debug mode is OFF for sdfRT")

    # -- Clamp num_steps so we don't march past the bounding region ------------
    effective_steps = min(num_steps, max(1, int(float(max_dist / step_size))))
    if effective_steps < num_steps:
        warnings.warn(
            f"sdf_visibility_v3: max_march_factor={max_march_factor} / step_size_factor={step_size_factor} "
            f"limits the march to {effective_steps} steps (requested {num_steps}). "
            "Increase max_march_factor or decrease step_size_factor to avoid missed shadows.",
            stacklevel=2,
        )

    # -- Ray origins: offset along sun direction to clear the surface ----------
    # Offsetting along sun (not normals) ensures the starting point is
    # `offset * bbox_diag` clear in the exact march direction for every face.
    # Back-facing faces land inside the body (negative SDF), driving their
    # visibility toward zero without explicit back-face culling.
    origins = centroids + offset * sun  # (F, 3)

    # -- March -----------------------------------------------------------------
    method = str(sdf_cfg["method"]).lower()
    print(f"Marching with method: {method!r}")
    if method == "sigmoid":
        visibility = _march_sigmoid(sdf_fn, origins, sun, effective_steps, step_size, alpha, point_chunk_size)
    elif method == "absorption":
        visibility = _march_absorption(sdf_fn, origins, sun, effective_steps, step_size, beta, kappa, point_chunk_size)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'sigmoid' or 'absorption'.")

    return visibility, normals, centroids
