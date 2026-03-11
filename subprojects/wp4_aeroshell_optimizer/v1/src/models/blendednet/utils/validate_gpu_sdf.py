"""
Validate GPU SDF against VTK's vtkImplicitPolyDataDistance.
"""

import time

import numpy as np
import pyvista as pv
import torch
import vtk
from gpu_sdf import load_vtk_to_torch, mesh_sdf


def vtk_sdf(query_np, surf):
    """Reference SDF using VTK (CPU)."""
    imp_dist = vtk.vtkImplicitPolyDataDistance()
    imp_dist.SetInput(surf)
    sdf = np.array([imp_dist.EvaluateFunction(p.tolist()) for p in query_np])
    return sdf


def main():
    vtk_path = "/Users/taebersold/Downloads/blended_dataset/train/vtk/case_0000.vtk"
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load mesh
    vertices, faces, surf = load_vtk_to_torch(vtk_path, device=device)
    print(f"Mesh: {vertices.shape[0]:,} vertices, {faces.shape[0]:,} triangles")

    # Generate test points: mix of near-surface, interior, and far-field
    np.random.seed(42)
    pts = np.array(surf.points)
    centroid = pts.mean(axis=0)

    # Near-surface: jitter mesh vertices
    n_near = 500
    idx = np.random.choice(len(pts), n_near)
    near_pts = pts[idx] + np.random.randn(n_near, 3) * 0.01

    # Interior: points near centroid
    n_interior = 200
    interior_pts = centroid + np.random.randn(n_interior, 3) * 0.03

    # Far-field
    n_far = 300
    far_pts = centroid + np.random.randn(n_far, 3) * 1.0

    query_np = np.vstack([near_pts, interior_pts, far_pts]).astype(np.float32)
    n_total = len(query_np)
    print(f"Query points: {n_total} ({n_near} near + {n_interior} interior + {n_far} far)")

    # --- VTK reference ---
    print("\nComputing VTK reference SDF...")
    t0 = time.time()
    sdf_vtk = vtk_sdf(query_np, surf)
    t_vtk = time.time() - t0
    print(f"  VTK time: {t_vtk:.2f}s ({n_total/t_vtk:.0f} pts/sec)")

    # --- GPU SDF ---
    query_gpu = torch.tensor(query_np, dtype=torch.float32, device=device)

    # Warmup
    _ = mesh_sdf(query_gpu[:10], vertices, faces, chunk_size=512)

    print("\nComputing GPU SDF...")
    t0 = time.time()
    sdf_gpu = mesh_sdf(query_gpu, vertices, faces, chunk_size=512)
    if device != "cpu":
        torch.mps.synchronize() if device == "mps" else torch.cuda.synchronize()
    t_gpu = time.time() - t0
    print(f"  GPU time: {t_gpu:.2f}s ({n_total/t_gpu:.0f} pts/sec)")

    sdf_gpu_np = sdf_gpu.cpu().numpy()

    # --- Compare ---
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    abs_err = np.abs(sdf_gpu_np - sdf_vtk)
    rel_err = abs_err / (np.abs(sdf_vtk) + 1e-6)

    # Sign agreement
    sign_agree = np.sign(sdf_gpu_np) == np.sign(sdf_vtk)
    # Exclude points very close to surface (sign is ambiguous)
    on_surface = np.abs(sdf_vtk) < 1e-4
    sign_agree_away = sign_agree[~on_surface].mean()
    sign_agree_all = sign_agree.mean()

    print(f"  Abs error:  mean={abs_err.mean():.6f}  max={abs_err.max():.6f}  median={np.median(abs_err):.6f}")
    print(f"  Rel error:  mean={rel_err.mean():.4f}  max={rel_err.max():.4f}  median={np.median(rel_err):.4f}")
    print(f"  Sign agree: {sign_agree_all:.1%} (all)  {sign_agree_away:.1%} (away from surface)")
    print(f"  On-surface points (|sdf| < 1e-4): {on_surface.sum()}")

    # Breakdown by region
    for label, mask in [("Near-surface", slice(0, n_near)),
                        ("Interior", slice(n_near, n_near+n_interior)),
                        ("Far-field", slice(n_near+n_interior, None))]:
        ae = abs_err[mask]
        sa = sign_agree[mask].mean()
        print(f"  {label:15s}  abs_err: mean={ae.mean():.6f} max={ae.max():.6f}  sign: {sa:.1%}")

    print(f"\nSpeedup: {t_vtk/t_gpu:.1f}x")

    # Scatter plot of GPU vs VTK
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.scatter(sdf_vtk, sdf_gpu_np, s=2, alpha=0.5)
        lim = max(abs(sdf_vtk.min()), abs(sdf_vtk.max()), abs(sdf_gpu_np.min()), abs(sdf_gpu_np.max()))
        ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1)
        ax.set_xlabel("VTK SDF")
        ax.set_ylabel("GPU SDF")
        ax.set_title("GPU vs VTK SDF")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.hist(abs_err, bins=50, edgecolor='black', linewidth=0.5)
        ax.set_xlabel("Absolute error")
        ax.set_ylabel("Count")
        ax.set_title(f"Error distribution (mean={abs_err.mean():.6f})")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("/Users/taebersold/Code/GLUE2/blendednet/sdf_validation.png", dpi=150)
        plt.show()
        print("Saved: blendednet/sdf_validation.png")
    except ImportError:
        print("(matplotlib not available, skipping plot)")


if __name__ == "__main__":
    main()
