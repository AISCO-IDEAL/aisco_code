"""
Inspect BlendedNet VTK files for SDF feasibility.

Usage:
    python inspect_vtk.py /path/to/some_case.vtk
    python inspect_vtk.py /path/to/BlendedNet/        # inspects first 5 VTK files in dir
"""

import sys
from pathlib import Path

import numpy as np
import pyvista as pv


def inspect_vtk(path: Path):
    print(f"\n{'='*70}")
    print(f"File: {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    print(f"{'='*70}")

    mesh = pv.read(str(path))

    # --- Basic info ---
    print(f"\nMesh type:       {type(mesh).__name__}")
    print(f"Points:          {mesh.n_points:,}")
    print(f"Cells:           {mesh.n_cells:,}")
    print(f"Cell types:      {set(mesh.celltypes) if hasattr(mesh, 'celltypes') else 'N/A'}")
    bounds = mesh.bounds
    print(f"Bounds X:        [{bounds[0]:.4f}, {bounds[1]:.4f}]  (span {bounds[1]-bounds[0]:.4f})")
    print(f"Bounds Y:        [{bounds[2]:.4f}, {bounds[3]:.4f}]  (span {bounds[3]-bounds[2]:.4f})")
    print(f"Bounds Z:        [{bounds[4]:.4f}, {bounds[5]:.4f}]  (span {bounds[5]-bounds[4]:.4f})")

    # --- Data arrays (Cp, Cf, etc.) ---
    print(f"\nPoint data arrays ({mesh.n_arrays}):")
    for name in mesh.point_data:
        arr = mesh.point_data[name]
        print(f"  {name:30s}  shape={arr.shape!s:15s}  "
              f"range=[{arr.min():.4f}, {arr.max():.4f}]  dtype={arr.dtype}")

    if mesh.cell_data:
        print(f"\nCell data arrays ({len(mesh.cell_data.keys())}):")
        for name in mesh.cell_data:
            arr = mesh.cell_data[name]
            print(f"  {name:30s}  shape={arr.shape!s:15s}  "
                  f"range=[{arr.min():.4f}, {arr.max():.4f}]  dtype={arr.dtype}")

    # --- Watertightness checks ---
    print("\n--- Watertightness ---")
    surf = mesh.extract_surface() if not isinstance(mesh, pv.PolyData) else mesh

    boundary = surf.extract_feature_edges(
        boundary_edges=True, non_manifold_edges=False,
        feature_edges=False, manifold_edges=False,
    )
    nonmanifold = surf.extract_feature_edges(
        boundary_edges=False, non_manifold_edges=True,
        feature_edges=False, manifold_edges=False,
    )
    n_boundary = boundary.n_cells
    n_nonmanifold = nonmanifold.n_cells

    print(f"Surface triangles: {surf.n_cells:,}")
    print(f"Boundary edges:    {n_boundary}  {'(OPEN!)' if n_boundary > 0 else '(closed)'}")
    print(f"Non-manifold edges:{n_nonmanifold}  {'(non-manifold!)' if n_nonmanifold > 0 else '(clean)'}")

    is_watertight = n_boundary == 0 and n_nonmanifold == 0
    print(f"Watertight:        {'YES' if is_watertight else 'NO'}")

    # --- Normals consistency ---
    surf_with_normals = surf.compute_normals(
        cell_normals=True, point_normals=False, consistent_normals=False
    )
    normals = surf_with_normals.cell_data.get("Normals", None)
    if normals is not None:
        centers = surf.cell_centers().points
        centroid = centers.mean(axis=0)
        outward = centers - centroid
        dots = np.sum(normals * outward, axis=1)
        frac_outward = (dots > 0).mean()
        print(f"Normals outward:   {frac_outward:.1%}  (expect ~100% or ~0% if consistent)")

    # --- Trailing edge detail ---
    pts = np.array(surf.points)
    x_max = pts[:, 0].max()
    te_mask = pts[:, 0] > x_max - 0.01 * (x_max - pts[:, 0].min())
    n_te_pts = te_mask.sum()
    if n_te_pts > 0:
        te_pts = pts[te_mask]
        te_thickness = te_pts[:, 2].max() - te_pts[:, 2].min()
        print("\nTrailing edge (top 1% of X):")
        print(f"  Points:          {n_te_pts}")
        print(f"  Z thickness:     {te_thickness:.6f}  {'(very thin!)' if te_thickness < 0.01 else ''}")

    return mesh, surf, is_watertight


def quick_sdf_test(surf):
    """Try computing SDF at a few test points using VTK's implicit distance."""
    print("\n--- Quick SDF test (vtkImplicitPolyDataDistance) ---")
    try:
        import vtk

        imp_dist = vtk.vtkImplicitPolyDataDistance()
        imp_dist.SetInput(surf)

        pts = np.array(surf.points)
        centroid = pts.mean(axis=0)
        far_pt = np.array([pts[:, 0].max() + 1.0, 0.0, 0.0])

        d_centroid = imp_dist.EvaluateFunction(centroid)
        d_far = imp_dist.EvaluateFunction(far_pt)

        # Sample a line through the middle in Z
        x_mid = (pts[:, 0].min() + pts[:, 0].max()) / 2
        y_mid = (pts[:, 1].min() + pts[:, 1].max()) / 2
        z_range = np.linspace(pts[:, 2].min() - 0.1, pts[:, 2].max() + 0.1, 50)

        sdf_line = np.array([
            imp_dist.EvaluateFunction([x_mid, y_mid, z])
            for z in z_range
        ])
        n_sign_changes = np.sum(np.diff(np.sign(sdf_line)) != 0)

        print(f"  SDF at centroid:     {d_centroid:+.4f}  ({'inside' if d_centroid < 0 else 'outside'})")
        print(f"  SDF far away:        {d_far:+.4f}  ({'inside' if d_far < 0 else 'outside'})")
        print(f"  Z-line sign changes: {n_sign_changes}  (expect 2 for closed surface)")
        print(f"  SDF range on line:   [{sdf_line.min():.4f}, {sdf_line.max():.4f}]")

        if d_centroid < 0 and d_far > 0 and n_sign_changes == 2:
            print("  --> SDF looks GOOD")
        elif n_sign_changes == 0:
            print("  --> SDF looks BROKEN (no zero crossings, mesh probably not closed)")
        else:
            print("  --> SDF is AMBIGUOUS (check visually)")

    except Exception as e:
        print(f"  Failed: {e}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_dir():
        vtk_files = sorted(path.glob("**/*.vtk"))
        total = len(vtk_files)
        vtk_files = vtk_files[:5]
        if not vtk_files:
            vtk_files = sorted(path.glob("**/*.vt[kpu]"))[:5]
        if not vtk_files:
            print(f"No VTK files found in {path}")
            sys.exit(1)
        print(f"Found {total} VTK files total, inspecting first {len(vtk_files)}")
    else:
        vtk_files = [path]

    for vf in vtk_files:
        mesh, surf, is_watertight = inspect_vtk(vf)
        quick_sdf_test(surf)

    if len(vtk_files) > 1:
        print(f"\n{'='*70}")
        print("KEY QUESTIONS:")
        print("  1. Watertight? (boundary edges = 0)")
        print("  2. SDF inside/outside correct? (centroid negative, far positive)")
        print("  3. Trailing edge thickness? (thin TE needs special handling)")
        print()
        print("If watertight:     VTK's built-in SDF is the fastest path")
        print("If NOT watertight: need mesh_to_sdf (handles open meshes)")


if __name__ == "__main__":
    main()
