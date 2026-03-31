"""8x4 cross-section visualization for BWB SDF training.

8 extreme geometries x (3 spanwise XZ slices + 1 top-down XY slice at z=0).
Each cell: predicted SDF contourf + zero contour + GT mesh overlay.
"""

import os

import numpy as np
import torch
from torch import Tensor


def _get_half_span(mesh, points: Tensor, geom_idx: int) -> float:
    """Get half-span (max Y extent) for a geometry.

    Uses VTK mesh bounds if available, else falls back to point cloud.

    Args:
        mesh: pyvista mesh or None
        points: [N, P, 3] all points
        geom_idx: geometry index

    Returns:
        half-span (max Y value)
    """
    if mesh is not None:
        bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        return bounds[3]  # ymax
    else:
        # fallback: use training point cloud extent
        y_vals = points[geom_idx, :, 1]
        return y_vals.max().item()


def _slice_mesh_at_y(mesh, y_val: float):
    """Slice VTK mesh at given Y to get XZ polygon.

    Args:
        mesh: pyvista PolyData
        y_val: spanwise station

    Returns:
        (x_coords, z_coords) numpy arrays, or (None, None) if slice fails
    """
    if mesh is None:
        return None, None

    try:
        slc = mesh.slice(normal="y", origin=(0, y_val, 0))
        if slc.n_points < 3:
            return None, None
        pts = slc.points  # [M, 3]
        return pts[:, 0], pts[:, 2]
    except Exception:
        return None, None


def _slice_mesh_at_z(mesh, z_val: float):
    """Slice VTK mesh at given Z to get XY polygon (top-down view).

    Args:
        mesh: pyvista PolyData
        z_val: vertical station

    Returns:
        (x_coords, y_coords) numpy arrays, or (None, None) if slice fails
    """
    if mesh is None:
        return None, None

    try:
        slc = mesh.slice(normal="z", origin=(0, 0, z_val))
        if slc.n_points < 3:
            return None, None
        pts = slc.points  # [M, 3]
        return pts[:, 0], pts[:, 1]
    except Exception:
        return None, None


@torch.no_grad()
def save_cross_section_figure(
    model,
    points: Tensor,
    params: Tensor,
    param_min: Tensor,
    param_max: Tensor,
    vis_geom_indices: list[int],
    geom_ids: list[str],
    vtk_meshes: list,
    cfg: dict,
    step: int,
    device: torch.device,
) -> str | None:
    """Save 8x3 cross-section visualization.

    Args:
        model: BWBSDFNet
        points: [N, P, 3] all training points
        params: [N, 9] all raw parameters
        param_min, param_max: [9] normalization stats
        vis_geom_indices: 8 geometry indices to visualize
        geom_ids: list of geometry ID strings
        vtk_meshes: list of 8 pyvista meshes (or empty)
        cfg: training config
        step: current training step
        device: compute device

    Returns:
        path to saved image, or None on failure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()

    n_geoms = len(vis_geom_indices)
    y_fractions = [0.0, 0.5, 0.9]  # root, 50%, 90%
    n_cols = len(y_fractions) + 1  # +1 for top-down z=0 slice
    res = 256

    # compute global bounds from VTK meshes (avoids far-field outliers in points)
    if vtk_meshes and any(m is not None for m in vtk_meshes):
        # mesh.bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
        g_x_min = min(m.bounds[0] for m in vtk_meshes if m is not None)
        g_x_max = max(m.bounds[1] for m in vtk_meshes if m is not None)
        g_y_min = min(m.bounds[2] for m in vtk_meshes if m is not None)
        g_y_max = max(m.bounds[3] for m in vtk_meshes if m is not None)
        g_z_min = min(m.bounds[4] for m in vtk_meshes if m is not None)
        g_z_max = max(m.bounds[5] for m in vtk_meshes if m is not None)
    else:
        # fallback: 2nd-98th percentile of training points
        all_pts = torch.stack([points[gi] for gi in vis_geom_indices])
        flat = all_pts.reshape(-1, 3)
        lo = torch.quantile(flat, 0.02, dim=0)
        hi = torch.quantile(flat, 0.98, dim=0)
        g_x_min, g_y_min, g_z_min = lo[0].item(), lo[1].item(), lo[2].item()
        g_x_max, g_y_max, g_z_max = hi[0].item(), hi[1].item(), hi[2].item()
    dx = (g_x_max - g_x_min) * 0.15
    dy = (g_y_max - g_y_min) * 0.15
    dz = max((g_z_max - g_z_min) * 0.15, 0.02)
    g_x_min -= dx; g_x_max += dx
    g_y_min -= dy; g_y_max += dy
    g_z_min -= dz; g_z_max += dz

    # precompute global grids
    xs_xz = torch.linspace(g_x_min, g_x_max, res, device=device)
    zs_xz = torch.linspace(g_z_min, g_z_max, res, device=device)
    xs_xy = torch.linspace(g_x_min, g_x_max, res, device=device)
    ys_xy = torch.linspace(g_y_min, g_y_max, res, device=device)

    fig, axes = plt.subplots(n_geoms, n_cols, figsize=(4 * n_cols, 3 * n_geoms))
    if n_geoms == 1:
        axes = axes[np.newaxis, :]

    for row, geom_idx in enumerate(vis_geom_indices):
        mesh = vtk_meshes[row] if vtk_meshes else None
        half_span = _get_half_span(mesh, points, geom_idx)

        # normalize this geometry's params
        raw_params = params[geom_idx].unsqueeze(0).to(device)  # [1, 9]
        cond_norm = 2.0 * (raw_params - param_min) / (param_max - param_min + 1e-8) - 1.0

        # --- columns 0-2: spanwise XZ slices ---
        for col, y_frac in enumerate(y_fractions):
            ax = axes[row, col]
            y_val = y_frac * half_span

            gx, gz = torch.meshgrid(xs_xz, zs_xz, indexing="xy")
            gy = torch.full_like(gx, y_val)
            grid = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
            grid = grid.unsqueeze(0)

            sdf_pred = model(grid, cond_norm)
            sdf_2d = sdf_pred.reshape(res, res).cpu().numpy()

            xs_np = xs_xz.cpu().numpy()
            zs_np = zs_xz.cpu().numpy()

            ax.contourf(
                xs_np, zs_np, sdf_2d,
                levels=50, cmap="RdBu_r", vmin=-0.15, vmax=0.15,
            )
            try:
                ax.contour(xs_np, zs_np, sdf_2d, levels=[0.0], colors="k", linewidths=2)
            except ValueError:
                pass

            gt_x, gt_z = _slice_mesh_at_y(mesh, y_val)
            if gt_x is not None:
                ax.scatter(gt_x, gt_z, c="gold", s=1, zorder=5)

            ax.set_xlim(g_x_min, g_x_max)
            ax.set_ylim(g_z_min, g_z_max)
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(f"Y={y_frac:.0%} span", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{geom_ids[geom_idx]}", fontsize=8)
            ax.tick_params(labelsize=6)

        # --- column 3: top-down XY slice at z=0 ---
        ax = axes[row, n_cols - 1]

        gx, gy = torch.meshgrid(xs_xy, ys_xy, indexing="xy")
        gz = torch.zeros_like(gx)
        grid = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
        grid = grid.unsqueeze(0)

        sdf_pred = model(grid, cond_norm)
        sdf_2d = sdf_pred.reshape(res, res).cpu().numpy()

        xs_np = xs_xy.cpu().numpy()
        ys_np = ys_xy.cpu().numpy()

        ax.contourf(
            xs_np, ys_np, sdf_2d,
            levels=50, cmap="RdBu_r", vmin=-0.15, vmax=0.15,
        )
        try:
            ax.contour(xs_np, ys_np, sdf_2d, levels=[0.0], colors="k", linewidths=2)
        except ValueError:
            pass

        gt_x, gt_y = _slice_mesh_at_z(mesh, 0.0)
        if gt_x is not None:
            ax.scatter(gt_x, gt_y, c="gold", s=1, zorder=5)

        ax.set_xlim(g_x_min, g_x_max)
        ax.set_ylim(g_y_min, g_y_max)
        ax.set_aspect("equal")
        if row == 0:
            ax.set_title("top-down Z=0", fontsize=9)
        ax.tick_params(labelsize=6)

    fig.suptitle(f"step {step} — black=pred zero, gold=GT mesh", fontsize=11)
    fig.tight_layout()

    path = os.path.join(cfg["output_dir"], f"viz_step_{step:06d}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")

    model.train()
    return path
