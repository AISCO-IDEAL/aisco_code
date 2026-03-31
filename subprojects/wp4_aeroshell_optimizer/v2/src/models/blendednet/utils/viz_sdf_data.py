"""Sanity check the precomputed SDF dataset.

Usage:
    python utils/viz_sdf_data.py                           # uses default local data
    python utils/viz_sdf_data.py --data /path/to/train_dataset.pt
    python utils/viz_sdf_data.py --data /path/to/train_dataset.pt --out /tmp/plots
"""

import argparse
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="visualize precomputed SDF dataset")
    parser.add_argument("--data", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "data", "train_dataset.pt"),
                        help="path to train_dataset.pt")
    parser.add_argument("--out", type=str, default=None,
                        help="output dir for plots (default: same dir as data)")
    args = parser.parse_args()

    data = torch.load(args.data, weights_only=False)
    out_dir = args.out or os.path.dirname(args.data)
    os.makedirs(out_dir, exist_ok=True)

    points = data["points"]   # [999, 25000, 3]
    sdf = data["sdf"]         # [999, 25000]
    params = data["params"]   # [999, 9]
    geom_ids = data["geom_ids"]
    param_names = data["param_names"]

    print(f"Dataset: {points.shape[0]} geometries × {points.shape[1]} points")
    print(f"Params: {param_names}")
    print("Param ranges:")
    for i, name in enumerate(param_names):
        print(f"  {name:5s}: [{params[:, i].min():.4f}, {params[:, i].max():.4f}]")

    print("\nSDF stats (all geometries):")
    print(f"  min:    {sdf.min():.4f}")
    print(f"  max:    {sdf.max():.4f}")
    print(f"  mean:   {sdf.mean():.4f}")
    print(f"  % inside (sdf<0): {(sdf < 0).float().mean():.1%}")
    print(f"  % near surface (|sdf|<0.01): {(sdf.abs() < 0.01).float().mean():.1%}")

    # Plot 4 geometries: SDF cross-sections at Y=0 (centerline)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Pick 4 diverse geometries
    idxs = [0, 250, 500, 750]

    for col, gi in enumerate(idxs):
        pts = points[gi].numpy()   # [25000, 3]
        sd = sdf[gi].numpy()       # [25000]
        par = params[gi].numpy()

        # Top row: XZ slice near Y=0 (centerline)
        y_center = 0.0
        mask = np.abs(pts[:, 1] - y_center) < 0.05
        ax = axes[0, col]
        if mask.sum() > 10:
            sc = ax.scatter(pts[mask, 0], pts[mask, 2], c=sd[mask],
                           cmap="RdBu_r", vmin=-0.15, vmax=0.15, s=2)
            ax.set_aspect("equal")
            plt.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_title(f"{geom_ids[gi]}\nXZ @ Y≈0 (n={mask.sum()})")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")

        # Bottom row: XY slice near Z=0 (planform view)
        z_center = 0.0
        mask = np.abs(pts[:, 2] - z_center) < 0.02
        ax = axes[1, col]
        if mask.sum() > 10:
            sc = ax.scatter(pts[mask, 0], pts[mask, 1], c=sd[mask],
                           cmap="RdBu_r", vmin=-0.15, vmax=0.15, s=2)
            ax.set_aspect("equal")
            plt.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_title(f"XY @ Z≈0 (n={mask.sum()})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.suptitle("SDF samples: blue=inside (-), red=outside (+)", fontsize=14)
    plt.tight_layout()
    path1 = os.path.join(out_dir, "sdf_sanity_check.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"\nSaved: {path1}")

    # SDF histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(sdf[0].numpy(), bins=100, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("SDF value")
    ax.set_ylabel("Count")
    ax.set_title(f"SDF distribution ({geom_ids[0]})")
    ax.axvline(0, color="red", ls="--", lw=1)

    ax = axes[1]
    # Show near-surface detail
    near_mask = sdf[0].abs() < 0.05
    ax.hist(sdf[0][near_mask].numpy(), bins=100, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("SDF value")
    ax.set_ylabel("Count")
    ax.set_title("Near-surface SDF (|sdf|<0.05)")
    ax.axvline(0, color="red", ls="--", lw=1)

    plt.tight_layout()
    path2 = os.path.join(out_dir, "sdf_histogram.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Saved: {path2}")


if __name__ == "__main__":
    main()
