#!/usr/bin/env python3
"""
Visualize .pt mesh files for Euler, Ahmed, and DrivAer datasets.
Creates multi-view plots showing mesh geometry colored by pressure (x),
wall shear stress magnitude, and Cp (if available).
"""
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT_DIR = "/cluster/home/elihuber/AISCO_code/visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = {
    "Euler (run_1)": "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler_pressure/euler_run_1/data.pt",
    "Ahmed (run_1)": "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_ahmed_pressure/ahmed_run_1/data.pt",
    "DrivAer (run_1)": "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_drivaer_pressure/drivaer_run_1/data.pt",
}

# Subsample large meshes for plotting
MAX_POINTS = 80_000


def subsample(data_dict, max_pts):
    """Subsample point cloud data for visualization."""
    n = data_dict["pos"].shape[0]
    if n <= max_pts:
        return data_dict, n
    idx = np.random.choice(n, max_pts, replace=False) # Noqa: NPY002
    idx.sort()
    idx_t = torch.from_numpy(idx)
    out = {}
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == n:
            out[k] = v[idx_t]
        else:
            out[k] = v
    return out, n


def make_overview_figure(name, data, n_original):
    """Create a multi-panel overview figure for one dataset."""
    pos = data["pos"].numpy()             # [N, 3]
    x_pressure = data["x"].numpy()[:, 0]  # [N]  pressure
    ws = data["wall_shear"].numpy()        # [N, 3]
    ws_mag = np.linalg.norm(ws, axis=1)

    has_cp = "cp" in data and data["cp"] is not None
    cp = data["cp"].numpy()[:, 0] if has_cp else None

    n_fields = 3 if has_cp else 2
    n_views = 3  # top, side, front

    fig, axes = plt.subplots(n_fields, n_views, figsize=(6 * n_views, 5 * n_fields),
                             subplot_kw={"projection": "3d"})
    if n_fields == 1:
        axes = axes[np.newaxis, :]

    point_size = max(0.05, min(1.0, 30_000 / pos.shape[0]))

    fields = [
        ("Pressure (x)", x_pressure, "RdBu_r", True),
        ("Wall Shear |τ|", ws_mag, "viridis", False),
    ]
    if has_cp:
        fields.append(("Cp", cp, "RdBu_r", True))

    view_angles = [
        ("Top (XY)", 90, -90),    # elev=90, azim=-90  → looking down Z
        ("Side (XZ)", 0, -90),    # elev=0, azim=-90   → looking along Y
        ("Front (YZ)", 0, 0),     # elev=0, azim=0     → looking along X
    ]

    for row, (field_name, values, cmap, diverging) in enumerate(fields):
        # Clip outliers for better color range
        vmin, vmax = np.percentile(values, [2, 98])
        norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax) if diverging and vmin < 0 < vmax else Normalize(vmin=vmin, vmax=vmax)

        for col, (view_name, elev, azim) in enumerate(view_angles):
            ax = axes[row, col]
            sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                            c=values, cmap=cmap, norm=norm,
                            s=point_size, alpha=0.6, edgecolors="none", rasterized=True)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            if col == 0:
                ax.set_ylabel(f"{field_name}\nY")
            ax.set_title(f"{view_name}")
            fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)

    force_val = data["force"].item() if not np.isnan(data["force"].item()) else "NaN"
    cd_val = data["cd"].item() if not np.isnan(data["cd"].item()) else "NaN"

    fig.suptitle(
        f"{name}\n"
        f"Vertices: {n_original:,} (showing {pos.shape[0]:,})  |  "
        f"Faces: {data['face'].shape[1]:,}  |  "
        f"Force: {force_val}  |  Cd: {cd_val}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


def make_histogram_figure(name, data, n_original):
    """Create histograms of all field values."""
    x_pressure = data["x"].numpy()[:, 0]
    ws = data["wall_shear"].numpy()
    ws_mag = np.linalg.norm(ws, axis=1)
    has_cp = "cp" in data and data["cp"] is not None

    n_fields = 4 if has_cp else 3
    fig, axes = plt.subplots(1, n_fields, figsize=(5 * n_fields, 4))

    # Pressure histogram
    axes[0].hist(x_pressure, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].set_title("Pressure (x)")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Count")
    axes[0].axvline(x_pressure.mean(), color="red", ls="--", label=f"mean={x_pressure.mean():.2f}")
    axes[0].legend(fontsize=8)

    # Wall shear magnitude
    axes[1].hist(ws_mag, bins=100, color="green", edgecolor="none", alpha=0.8)
    axes[1].set_title("Wall Shear |τ|")
    axes[1].set_xlabel("Value")

    # Normal coordinates c
    c = data["c"].numpy()
    for i, (label, color) in enumerate(zip(["cx", "cy", "cz"], ["r", "g", "b"], strict=False)):
        axes[2].hist(c[:, i], bins=50, color=color, edgecolor="none", alpha=0.5, label=label)
    axes[2].set_title("Surface Normals (c)")
    axes[2].legend(fontsize=8)

    if has_cp:
        cp = data["cp"].numpy()[:, 0]
        axes[3].hist(cp, bins=100, color="orange", edgecolor="none", alpha=0.8)
        axes[3].set_title("Cp (pressure coefficient)")
        axes[3].set_xlabel("Value")
        axes[3].axvline(cp.mean(), color="red", ls="--", label=f"mean={cp.mean():.2f}")
        axes[3].legend(fontsize=8)

    fig.suptitle(f"{name} — Field Distributions (n={n_original:,})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def make_coordinate_figure(name, data, n_original):
    """Create 2D scatter plots of coordinate distributions."""
    pos = data["pos"].numpy()
    x_pressure = data["x"].numpy()[:, 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmin, vmax = np.percentile(x_pressure, [2, 98])

    for ax, (xi, yi, xlabel, ylabel, title) in zip(axes, [
        (0, 1, "X", "Y", "Top View (XY)"),
        (0, 2, "X", "Z", "Side View (XZ)"),
        (1, 2, "Y", "Z", "Front View (YZ)"),
    ], strict=False):
        sc = ax.scatter(pos[:, xi], pos[:, yi], c=x_pressure, cmap="RdBu_r",
                        s=0.1, alpha=0.5, vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.8)

    fig.suptitle(f"{name} — 2D Projections colored by Pressure (n={n_original:,})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    np.random.seed(42) # Noqa: NPY002

    for name, path in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"  Path: {path}")

        if not os.path.exists(path):
            print("  SKIPPED — file not found")
            continue

        data = torch.load(path, weights_only=False)

        # Print summary
        print("  Fields:")
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k:15s} shape={v.shape!s:25s} dtype={v.dtype}  "
                      f"range=[{v.min().item():.4f}, {v.max().item():.4f}]")
            else:
                print(f"    {k:15s} = {v}")

        # Subsample for visualization
        data_sub, n_orig = subsample(data, MAX_POINTS)
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")

        # 1) 2D projection plots (fast, informative)
        print("  Creating 2D projection plot...")
        fig2d = make_coordinate_figure(name, data_sub, n_orig)
        fig2d.savefig(os.path.join(OUT_DIR, f"{safe_name}_2d_projections.png"),
                      dpi=150, bbox_inches="tight")
        plt.close(fig2d)

        # 2) Histograms
        print("  Creating histogram plot...")
        fig_hist = make_histogram_figure(name, data_sub, n_orig)
        fig_hist.savefig(os.path.join(OUT_DIR, f"{safe_name}_histograms.png"),
                         dpi=150, bbox_inches="tight")
        plt.close(fig_hist)

        # 3) 3D views (slower but useful)
        print("  Creating 3D overview plot...")
        fig3d = make_overview_figure(name, data_sub, n_orig)
        fig3d.savefig(os.path.join(OUT_DIR, f"{safe_name}_3d_overview.png"),
                      dpi=150, bbox_inches="tight")
        plt.close(fig3d)

        print(f"  Done! Saved to {OUT_DIR}/{safe_name}_*.png")

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {OUT_DIR}/")
