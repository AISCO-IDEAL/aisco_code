"""Tests for process_openfoam_pressure.py — OpenFOAM .pt extraction."""

import math
import statistics
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
import trimesh

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.dataset.openfoam.process_openfoam_pressure import (
    _count_boundary_edges,
    _recover_surface_mesh,
    extract_patch_vtk,
    parse_cd_stats,
    parse_convergence_status,
    parse_openfoam_forces,
    process_all_runs,
    process_single_run,
)

# ---------------------------------------------------------------------------
#  Helpers: create synthetic OpenFOAM output fixtures
# ---------------------------------------------------------------------------


def _create_synthetic_vtk(case_dir: Path, time_step: int = 300) -> None:
    """Create a synthetic VTK file mimicking foamToVTK output for car_surface.

    OpenFOAM 2306 foamToVTK writes boundary patches to:
        VTK/{case_name}_{time_step}/boundary/{patch_name}.vtp
    """
    run_name = case_dir.name  # e.g. "RUN_001"
    vtk_dir = case_dir / "VTK" / f"{run_name}_{time_step}" / "boundary"
    vtk_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple box surface mesh (8 verts, 12 triangles)
    box = pv.Box(bounds=(-2, 2, -0.9, 0.9, 0, 1.5))
    # Triangulate to ensure consistent face topology
    box = box.triangulate()

    n_cells = box.n_cells

    # Add synthetic pressure (cell data — face-centered, as foamToVTK outputs)
    # Use a realistic gauge pressure range for external aero (~-500 to +200 Pa)
    rng = np.random.default_rng(42)
    pressure = rng.uniform(-500.0, 200.0, n_cells).astype(np.float64)
    box.cell_data["p"] = pressure

    # Add synthetic wall shear stress (cell data, 3-component vector)
    wss = rng.uniform(-5.0, 5.0, (n_cells, 3)).astype(np.float64)
    box.cell_data["wallShearStress"] = wss

    # Save as .vtp (VTK PolyData XML) — the format foamToVTK uses
    vtk_path = vtk_dir / "car_surface.vtp"
    box.save(str(vtk_path))


def _create_open_bottom_vtk(case_dir: Path, time_step: int = 300) -> None:
    """Create a synthetic VTK patch with the underside removed.

    This mimics the historical OpenFOAM extraction artifact where the car sat on
    the floor plane and the exported `car_surface` patch was open on the bottom.
    """
    run_name = case_dir.name
    vtk_dir = case_dir / "VTK" / f"{run_name}_{time_step}" / "boundary"
    vtk_dir.mkdir(parents=True, exist_ok=True)

    box = pv.Box(bounds=(-2, 2, -0.9, 0.9, 0, 1.5)).triangulate()
    bottom_cell_ids = np.where(np.isclose(box.cell_normals[:, 2], -1.0, atol=1e-6))[0]
    keep_cell_ids = np.setdiff1d(np.arange(box.n_cells), bottom_cell_ids)
    open_box = box.extract_cells(keep_cell_ids).extract_surface().triangulate()

    rng = np.random.default_rng(7)
    pressure = rng.uniform(-500.0, 200.0, open_box.n_cells).astype(np.float64)
    wall_shear = rng.uniform(-5.0, 5.0, (open_box.n_cells, 3)).astype(np.float64)
    open_box.cell_data["p"] = pressure
    open_box.cell_data["wallShearStress"] = wall_shear
    vtk_path = vtk_dir / "car_surface.vtp"
    open_box.save(str(vtk_path))


def _create_closed_stl(case_dir: Path) -> None:
    """Create the closed STL surface that downstream recovery should project onto."""
    tri_surface_dir = case_dir / "constant" / "triSurface"
    tri_surface_dir.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=[4.0, 1.8, 1.5])
    mesh.apply_translation((0.0, 0.0, 0.75))
    mesh.export(str(tri_surface_dir / "car_surface.stl"))


def _create_vtk_with_minor_boundary_edges(case_dir: Path, time_step: int = 300) -> int:
    """Create a high-res VTK patch with minor boundary edges (< 1% of faces).

    Builds a sphere mesh with ~1700 faces, removes 2 triangles to create
    ~4-6 boundary edges (ratio ~0.3%), mimicking the normal snappyHexMesh
    artifact where boundary patch meets domain boundaries.

    Args:
        case_dir: Case directory (must already exist).
        time_step: VTK time-step subdirectory.

    Returns:
        Number of points in the created VTK mesh.
    """
    run_name = case_dir.name
    vtk_dir = case_dir / "VTK" / f"{run_name}_{time_step}" / "boundary"
    vtk_dir.mkdir(parents=True, exist_ok=True)

    sphere = pv.Sphere(radius=1.0, center=(0, 0, 0.75), theta_resolution=30, phi_resolution=30)
    sphere = sphere.triangulate().clean()

    # Remove 2 triangles → creates ~4-6 boundary edges on a ~1700-face mesh
    keep = np.ones(sphere.n_cells, dtype=bool)
    keep[0] = False
    keep[1] = False
    mesh = sphere.extract_cells(keep).extract_surface().triangulate().clean()

    rng = np.random.default_rng(42)
    mesh.cell_data["p"] = rng.uniform(-500.0, 200.0, mesh.n_cells).astype(np.float64)
    mesh.cell_data["wallShearStress"] = rng.uniform(-5.0, 5.0, (mesh.n_cells, 3)).astype(np.float64)

    mesh.save(str(vtk_dir / "car_surface.vtp"))
    return int(mesh.n_points)


def _create_low_res_stl(case_dir: Path) -> int:
    """Create a low-resolution closed STL (12 triangles / 8 vertices).

    Args:
        case_dir: Case directory (must already exist).

    Returns:
        Number of vertices in the STL.
    """
    tri_surface_dir = case_dir / "constant" / "triSurface"
    tri_surface_dir.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=[2.0, 2.0, 1.5])
    mesh.apply_translation((0.0, 0.0, 0.75))
    mesh.export(str(tri_surface_dir / "car_surface.stl"))
    return len(mesh.vertices)


def _create_synthetic_forces(case_dir: Path) -> None:
    """Create synthetic forceCoeffs output matching OpenFOAM format."""
    forces_dir = case_dir / "postProcessing" / "forceCoeffs" / "0"
    forces_dir.mkdir(parents=True, exist_ok=True)

    # OpenFOAM forceCoeffs.dat format (v2306):
    # Time  Cd  Cs  Cl  CmRoll  CmPitch  CmYaw  Cd(f)  Cd(r)  Cs(f)  Cs(r)  Cl(f)  Cl(r)
    content = """# Force coefficients
# CofR        : (0 0 0)
# liftDir     : (0 0 1)
# dragDir     : (-1 0 0)
# pitchAxis   : (0 1 0)
# magUInf     : 25
# lRef        : 4
# Aref        : 1.35
# Lookup fields: (p)
# Time  Cd  Cs  Cl  CmRoll  CmPitch  CmYaw  Cd(f)  Cd(r)  Cs(f)  Cs(r)  Cl(f)  Cl(r)
1       0.450   0.001   0.120   0.001   0.05    0.001   0.225   0.225   0.001   0.001   0.06    0.06
100     0.380   0.001   0.110   0.001   0.04    0.001   0.190   0.190   0.001   0.001   0.055   0.055
200     0.355   0.001   0.105   0.001   0.04    0.001   0.178   0.178   0.001   0.001   0.053   0.053
300     0.342   0.001   0.102   0.001   0.03    0.001   0.171   0.171   0.001   0.001   0.051   0.051
"""
    (forces_dir / "coefficient.dat").write_text(content)

    # Also create forces.dat for drag force extraction.
    # OpenFOAM 2306 format: Time total_x total_y total_z pressure_x pressure_y pressure_z viscous_x viscous_y viscous_z
    forces_dir2 = case_dir / "postProcessing" / "forces" / "0"
    forces_dir2.mkdir(parents=True, exist_ok=True)
    forces_content = """# Force         
# CofR          : (0.00000000e+00 0.00000000e+00 0.00000000e+00)
#
# Time                  total_x total_y total_z pressure_x pressure_y pressure_z viscous_x viscous_y viscous_z
1     -1.500000e+02    1.20e+00   5.00e+01   -1.400000e+02    1.10e+00   4.95e+01   -1.000000e+01    1.00e-01    5.00e-01
100   -1.300000e+02    1.00e+00   4.50e+01   -1.210000e+02    9.50e-01   4.46e+01   -9.000000e+00    5.00e-02    4.00e-01
300   -1.288000e+02    8.00e-01   4.20e+01   -1.205000e+02    7.50e-01   4.16e+01   -8.300000e+00    5.00e-02    4.00e-01
"""
    # At t=300: total_x=-128.8, pressure_x=-120.5, viscous_x=-8.3
    # total_x = pressure_x + viscous_x = -120.5 + -8.3 = -128.8  (self-consistent)
    # Correct drag force = -total_x = 128.8
    (forces_dir2 / "force.dat").write_text(forces_content)


def _create_synthetic_restart_forces(case_dir: Path) -> None:
    """Create force outputs split across restart directories.

    OpenFOAM writes restarted functionObject output under a new postProcessing
    time directory (e.g. `forceCoeffs/300/coefficient.dat`) instead of
    appending to `forceCoeffs/0/coefficient.dat`. The parser must select the
    latest available directory, not assume `0`.
    """
    coeff_0 = case_dir / "postProcessing" / "forceCoeffs" / "0"
    coeff_300 = case_dir / "postProcessing" / "forceCoeffs" / "300"
    forces_0 = case_dir / "postProcessing" / "forces" / "0"
    forces_300 = case_dir / "postProcessing" / "forces" / "300"

    coeff_0.mkdir(parents=True, exist_ok=True)
    coeff_300.mkdir(parents=True, exist_ok=True)
    forces_0.mkdir(parents=True, exist_ok=True)
    forces_300.mkdir(parents=True, exist_ok=True)

    (coeff_0 / "coefficient.dat").write_text(
        """# Time  Cd  Cs  Cl\n"
        "300     0.356005   0.001   0.102\n"
        """
    )
    (coeff_300 / "coefficient.dat").write_text(
        """# Time  Cd  Cs  Cl\n"
        "600     0.276729   0.001   0.050\n"
        """
    )

    (forces_0 / "force.dat").write_text(
        """# Time total_x total_y total_z pressure_x pressure_y pressure_z viscous_x viscous_y viscous_z\n"
        "300   -2.97681134e+02  1.48e+02  -1.61e+02  -2.97022094e+02  1.47e+02  -1.61e+02  -6.59040207e-01  3.19e-01  -3.99e-02\n"
        """
    )
    (forces_300 / "force.dat").write_text(
        """# Time total_x total_y total_z pressure_x pressure_y pressure_z viscous_x viscous_y viscous_z\n"
        "600   -2.31479203e+02  9.10e+01  -8.55e+01  -2.30820197e+02  9.07e+01  -8.54e+01  -6.59006000e-01  3.00e-01  -1.00e-01\n"
        """
    )


def _create_synthetic_case(cases_dir: Path) -> Path:
    """Create a complete synthetic OpenFOAM case directory for testing."""
    case_dir = cases_dir / "RUN_001"
    case_dir.mkdir(parents=True)
    _create_synthetic_vtk(case_dir)
    _create_synthetic_forces(case_dir)
    _write_converged_log(case_dir)
    return case_dir


class TestParseOpenfoamForces:
    def test_parses_cd_from_coefficient_dat(self) -> None:
        """Extracts last Cd from forceCoeffs coefficient.dat."""
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            _create_synthetic_forces(case_dir)
            cd, _force = parse_openfoam_forces(case_dir)
            assert cd is not None
            # Last line has Cd=0.342
            assert abs(cd - 0.342) < 0.001

    def test_parses_drag_force_from_force_dat(self) -> None:
        """Drag force equals negative total_x from the last time step row.

        force.dat columns (OF 2306): time total_x total_y total_z pressure_x ...
        Correct: force = -total_x  (dragDir = (-1 0 0)).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            _create_synthetic_forces(case_dir)
            _cd, force = parse_openfoam_forces(case_dir)
            # t=300 fixture: total_x = -128.80  →  force = 128.80
            assert force is not None
            assert abs(force - 128.80) < 0.01, (
                f"Expected force ≈ 128.80 (-total_x), got {force:.4f}. "
                "Likely the column mapping is wrong (double-counted pressure)."
            )

    def test_force_not_double_counted(self) -> None:
        """force must be -total_x, not -(total_x + pressure_x).

        The buggy formula -(total_x + pressure_x) ~= 2x drag because
        total_x = pressure_x + viscous_x, so summing them double-counts pressure.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            _create_synthetic_forces(case_dir)
            _cd, force = parse_openfoam_forces(case_dir)
            # -(total_x + pressure_x) = -(-128.80 + -120.50) = 249.30
            assert force is not None
            assert abs(force - 249.30) > 1.0, (
                f"force={force:.4f} matches the double-counted value 249.30. "
                "The buggy formula -(total_x + pressure_x) is still in use."
            )

    def test_returns_none_on_missing_files(self) -> None:
        """Returns (None, None) when force files don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cd, force = parse_openfoam_forces(Path(tmpdir))
            assert cd is None
            assert force is None

    def test_uses_latest_restart_force_directory(self) -> None:
        """Restarted runs must read the latest force/coeff directory, not hardcoded `0`."""
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            _create_synthetic_restart_forces(case_dir)
            cd, force = parse_openfoam_forces(case_dir)

            assert cd is not None
            assert force is not None
            assert abs(cd - 0.276729) < 1e-6
            assert abs(force - 231.479203) < 1e-6

    def test_reads_parallel_coefficient_0_dat(self) -> None:
        """Reads coefficient_0.dat when only the parallel-run file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            coeff_dir = case_dir / "postProcessing" / "forceCoeffs" / "0"
            coeff_dir.mkdir(parents=True)
            content = "# Time  Cd  Cs  Cl\n3000    0.270   0.001   0.050\n"
            (coeff_dir / "coefficient_0.dat").write_text(content)
            cd, _force = parse_openfoam_forces(case_dir)
            assert cd is not None
            assert abs(cd - 0.270) < 0.001

    def test_reads_parallel_force_0_dat(self) -> None:
        """Reads force_0.dat when only the parallel-run file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            forces_dir = case_dir / "postProcessing" / "forces" / "0"
            forces_dir.mkdir(parents=True)
            content = "# Time total_x total_y total_z\n3000   -2.21600e+02  1.00e+00  5.00e+01\n"
            (forces_dir / "force_0.dat").write_text(content)
            _cd, force = parse_openfoam_forces(case_dir)
            assert force is not None
            assert abs(force - 221.6) < 0.1

    def test_prefers_parallel_over_serial_when_both_exist(self) -> None:
        """When both coefficient.dat and coefficient_0.dat exist, use _0 (parallel)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            coeff_dir = case_dir / "postProcessing" / "forceCoeffs" / "0"
            coeff_dir.mkdir(parents=True)
            forces_dir = case_dir / "postProcessing" / "forces" / "0"
            forces_dir.mkdir(parents=True)

            # Serial (old, stale data from 1200 iter)
            (coeff_dir / "coefficient.dat").write_text("# Time  Cd\n1200    0.450   0.001   0.050\n")
            # Parallel (new, correct data from 3000 iter)
            (coeff_dir / "coefficient_0.dat").write_text("# Time  Cd\n3000    0.270   0.001   0.050\n")

            # Serial force (old)
            (forces_dir / "force.dat").write_text("# Time total_x\n1200   -4.45800e+02  1.00e+00  5.00e+01\n")
            # Parallel force (new)
            (forces_dir / "force_0.dat").write_text("# Time total_x\n3000   -2.21600e+02  1.00e+00  5.00e+01\n")

            cd, force = parse_openfoam_forces(case_dir)
            assert cd is not None
            assert force is not None
            # Must read the _0 (parallel) values, NOT the serial ones
            assert abs(cd - 0.270) < 0.001, f"Expected Cd=0.270 (parallel), got {cd}"
            assert abs(force - 221.6) < 0.1, f"Expected force=221.6 (parallel), got {force}"


class TestExtractPatchVtk:
    def test_extracts_mesh_and_fields(self) -> None:
        """Extracts vertices, faces, pressure, and wallShearStress from VTK."""
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "RUN_001"
            case_dir.mkdir()
            _create_synthetic_vtk(case_dir)
            data = extract_patch_vtk(case_dir, time_step=300)
            assert "vertices" in data
            assert "faces" in data
            assert "pressure" in data
            assert "wall_shear" in data
            assert data["vertices"].shape[1] == 3
            assert data["pressure"].ndim == 1
            assert data["wall_shear"].shape[1] == 3

    def test_pressure_is_per_vertex(self) -> None:
        """Pressure is interpolated to vertices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "RUN_001"
            case_dir.mkdir()
            _create_synthetic_vtk(case_dir)
            data = extract_patch_vtk(case_dir, time_step=300)
            assert len(data["pressure"]) == len(data["vertices"])

    def test_falls_back_to_latest_vtk_when_time_step_missing(self) -> None:
        """If the requested time_step VTK dir is absent, uses the latest available.

        This supports runs that converge early via residualControl (e.g. t=546
        rather than t=600), so the extractor can still find their output without
        knowing the exact convergence iteration.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "RUN_001"
            case_dir.mkdir()
            # Create VTK at t=600, not t=300
            _create_synthetic_vtk(case_dir, time_step=600)
            # Requesting t=300 should fall back to the available t=600
            data = extract_patch_vtk(case_dir, time_step=300)
            assert "vertices" in data
            assert data["vertices"].shape[1] == 3


class TestRecoverSurfaceMesh:
    """Tests for the boundary-edge threshold in _recover_surface_mesh."""

    def test_preserves_vtk_when_boundary_edges_are_minor(self) -> None:
        """VTK mesh with minor boundary edges (<1% of faces) is returned as-is.

        snappyHexMesh produces 0.01-0.05% boundary edges at domain boundaries.
        These are normal mesh artifacts and should NOT trigger the STL fallback
        that destroys surface resolution.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "RUN_001"
            case_dir.mkdir(parents=True)

            vtk_n_points = _create_vtk_with_minor_boundary_edges(case_dir)
            stl_n_verts = _create_low_res_stl(case_dir)

            # Load the VTK mesh the same way extract_patch_vtk does
            vtk_path = case_dir / "VTK" / "RUN_001_300" / "boundary" / "car_surface.vtp"
            source_mesh = pv.read(str(vtk_path)).triangulate().clean()

            # Confirm the mesh actually has boundary edges
            n_boundary = _count_boundary_edges(source_mesh)
            assert n_boundary > 0, "Test mesh must have boundary edges"
            ratio = n_boundary / source_mesh.n_cells
            assert ratio < 0.01, f"Test mesh boundary ratio {ratio:.4f} should be < 0.01"

            # _recover_surface_mesh should preserve the VTK mesh
            result = _recover_surface_mesh(case_dir, source_mesh)
            assert result.n_points == vtk_n_points, (
                f"Expected VTK resolution ({vtk_n_points} pts) but got "
                f"{result.n_points} pts — STL fallback was incorrectly triggered"
            )
            assert result.n_points > stl_n_verts, (
                f"Result ({result.n_points} pts) ≤ STL ({stl_n_verts} pts) — STL fallback destroyed VTK resolution"
            )

    def test_retains_vtk_for_genuinely_open_mesh_and_warns(self) -> None:
        """VTK mesh with major boundary edges (>1%) is retained with a WARNING logged.

        The STL fallback was removed (analysis 026 showed it produced all-zero
        pressure).  High boundary-edge ratio is now logged as a WARNING only;
        the VTK mesh is always returned unchanged.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "RUN_001"
            case_dir.mkdir(parents=True)
            _create_open_bottom_vtk(case_dir)
            _create_closed_stl(case_dir)

            vtk_path = case_dir / "VTK" / "RUN_001_300" / "boundary" / "car_surface.vtp"
            source_mesh = pv.read(str(vtk_path)).triangulate().clean()

            n_boundary = _count_boundary_edges(source_mesh)
            ratio = n_boundary / source_mesh.n_cells
            assert ratio > 0.01, f"Test mesh boundary ratio {ratio:.4f} should be > 0.01 for genuinely open mesh"

            # _recover_surface_mesh must return the original VTK mesh (not STL)
            result = _recover_surface_mesh(case_dir, source_mesh)
            assert result.n_points == source_mesh.n_points, (
                f"Expected VTK mesh ({source_mesh.n_points} pts) to be returned unchanged, "
                f"got {result.n_points} pts — STL fallback must be removed"
            )

    def test_preserves_field_data_when_below_threshold(self) -> None:
        """Point fields (p, wallShearStress) are preserved on the VTK mesh."""
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "RUN_001"
            case_dir.mkdir(parents=True)
            _create_vtk_with_minor_boundary_edges(case_dir)
            _create_low_res_stl(case_dir)

            vtk_path = case_dir / "VTK" / "RUN_001_300" / "boundary" / "car_surface.vtp"
            source_mesh = pv.read(str(vtk_path)).triangulate().clean()

            result = _recover_surface_mesh(case_dir, source_mesh)
            # Fields should be interpolated to point data
            assert "p" in result.point_data, "Pressure field missing from result"
            assert "wallShearStress" in result.point_data, "Wall shear stress missing from result"
            assert len(result.point_data["p"]) == result.n_points


class TestProcessSingleRun:
    def test_produces_pt_file_with_correct_schema(self) -> None:
        """Produces a .pt file with all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cases_dir = Path(tmpdir) / "CASES"
            output_dir = Path(tmpdir) / "output"
            cases_dir.mkdir()
            output_dir.mkdir()

            _create_synthetic_case(cases_dir)

            result = process_single_run(1, str(cases_dir), str(output_dir))
            assert result is not None

            # Load and verify schema
            pt_path = output_dir / result / "data.pt"
            assert pt_path.exists()
            data = torch.load(pt_path, weights_only=False)

            # Required fields from Fluent extractor schema
            assert hasattr(data, "pos")
            assert hasattr(data, "x")  # Cp
            assert hasattr(data, "c")  # normals
            assert hasattr(data, "face")
            assert hasattr(data, "wall_shear")
            assert hasattr(data, "force")
            assert hasattr(data, "cd")
            assert hasattr(data, "v_inf")
            assert hasattr(data, "conditioning")
            assert hasattr(data, "filename")
            assert hasattr(data, "converged")
            assert hasattr(data, "final_residuals")

    def test_tensor_shapes_match_fluent_schema(self) -> None:
        """Tensor shapes match the Fluent extractor output format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cases_dir = Path(tmpdir) / "CASES"
            output_dir = Path(tmpdir) / "output"
            cases_dir.mkdir()
            output_dir.mkdir()

            _create_synthetic_case(cases_dir)

            result = process_single_run(1, str(cases_dir), str(output_dir))
            data = torch.load(output_dir / result / "data.pt", weights_only=False)

            n_verts = data.pos.shape[0]
            assert data.pos.shape == (n_verts, 3)
            assert data.x.shape == (n_verts, 1)
            assert data.c.shape == (n_verts, 3)
            assert data.face.shape[0] == 3  # [3, F]
            assert data.wall_shear.shape == (n_verts, 3)
            assert data.force.shape == (1,)
            assert data.cd.shape == (1,)
            assert data.v_inf.shape == (1,)
            assert data.conditioning.shape == (1, 7)
            assert data.converged.shape == (1,)
            assert data.converged.dtype == torch.bool
            assert data.final_residuals.shape == (4,)
            assert data.final_residuals.dtype == torch.float32

    def test_returns_none_on_missing_case(self) -> None:
        """Returns None when case directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = process_single_run(999, tmpdir, tmpdir)
            assert result is None

    def test_saves_vtk_mesh_when_vtk_patch_is_open(self) -> None:
        """When the VTK patch has high boundary-edge ratio the .pt is still written
        using the VTK mesh (STL fallback removed — analysis 026).

        The saved mesh may have boundary edges (the open VTK patch is used
        as-is), but pressure and wall_shear fields must be finite.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cases_dir = Path(tmpdir) / "CASES"
            output_dir = Path(tmpdir) / "output"
            cases_dir.mkdir()
            output_dir.mkdir()

            case_dir = cases_dir / "RUN_001"
            case_dir.mkdir(parents=True)
            _create_open_bottom_vtk(case_dir)
            _create_synthetic_forces(case_dir)
            _create_closed_stl(case_dir)

            result = process_single_run(1, str(cases_dir), str(output_dir))
            assert result is not None, "process_single_run returned None — extraction failed"
            data = torch.load(output_dir / result / "data.pt", weights_only=False)

            assert torch.isfinite(data.pressure).all(), "Pressure field contains non-finite values"
            assert torch.isfinite(data.wall_shear).all(), "Wall shear field contains non-finite values"


def _create_run(cases_dir: Path, run_id: int) -> None:
    """Helper: create a full synthetic case for a given run ID."""
    case_dir = cases_dir / f"RUN_{run_id:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    _create_synthetic_vtk(case_dir)
    _create_synthetic_forces(case_dir)
    _write_converged_log(case_dir)


class TestProcessAllRuns:
    def test_processes_multiple_runs_and_writes_order_file(self) -> None:
        """Processes multiple runs and writes order_.txt with sorted names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cases_dir = Path(tmpdir) / "CASES"
            output_dir = Path(tmpdir) / "output"
            cases_dir.mkdir()
            output_dir.mkdir()
            _create_run(cases_dir, 1)
            _create_run(cases_dir, 2)

            process_all_runs(
                cases_dir=str(cases_dir),
                output_dir=str(output_dir),
                run_ids=[1, 2],
                num_workers=1,
            )

            assert (output_dir / "euler_run_1" / "data.pt").exists()
            assert (output_dir / "euler_run_2" / "data.pt").exists()
            order_file = output_dir / "order_.txt"
            assert order_file.exists()
            lines = order_file.read_text().strip().splitlines()
            assert "euler_run_1" in lines
            assert "euler_run_2" in lines

    def test_max_runs_limits_batch(self) -> None:
        """max_runs truncates the run list before processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cases_dir = Path(tmpdir) / "CASES"
            output_dir = Path(tmpdir) / "output"
            cases_dir.mkdir()
            output_dir.mkdir()
            for run_id in [1, 2, 3]:
                _create_run(cases_dir, run_id)

            process_all_runs(
                cases_dir=str(cases_dir),
                output_dir=str(output_dir),
                run_ids=[1, 2, 3],
                max_runs=2,
                num_workers=1,
            )

            assert (output_dir / "euler_run_1" / "data.pt").exists()
            assert (output_dir / "euler_run_2" / "data.pt").exists()
            assert not (output_dir / "euler_run_3").exists()

    def test_skips_missing_runs_without_aborting(self) -> None:
        """A missing case directory is skipped; other runs still complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cases_dir = Path(tmpdir) / "CASES"
            output_dir = Path(tmpdir) / "output"
            cases_dir.mkdir()
            output_dir.mkdir()
            _create_run(cases_dir, 1)  # Run 2 is intentionally absent

            process_all_runs(
                cases_dir=str(cases_dir),
                output_dir=str(output_dir),
                run_ids=[1, 2],
                num_workers=1,
            )

            assert (output_dir / "euler_run_1" / "data.pt").exists()
            assert not (output_dir / "euler_run_2").exists()

    def test_autodetects_run_ids_from_directories(self) -> None:
        """When run_ids=None, auto-detects available RUN_XXX directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cases_dir = Path(tmpdir) / "CASES"
            output_dir = Path(tmpdir) / "output"
            cases_dir.mkdir()
            output_dir.mkdir()
            _create_run(cases_dir, 5)  # Non-trivial ID to verify detection

            process_all_runs(
                cases_dir=str(cases_dir),
                output_dir=str(output_dir),
                run_ids=None,
                num_workers=1,
            )

            assert (output_dir / "euler_run_5" / "data.pt").exists()


# ---------------------------------------------------------------------------
#  Convergence log fixtures
# ---------------------------------------------------------------------------


def _write_converged_log(case_dir: Path) -> None:
    """Write a synthetic log.simpleFoam that indicates early convergence."""
    log = case_dir / "log.simpleFoam"
    log.write_text(
        """\
Time = 845

smoothSolver:  Solving for Ux, Initial residual = 1.23e-06, Final residual = 5.67e-08, No Iterations 1
smoothSolver:  Solving for Uy, Initial residual = 2.34e-06, Final residual = 8.90e-08, No Iterations 1
smoothSolver:  Solving for Uz, Initial residual = 3.45e-06, Final residual = 1.11e-07, No Iterations 1
GAMG:  Solving for p, Initial residual = 4.56e-06, Final residual = 2.22e-07, No Iterations 3
smoothSolver:  Solving for k, Initial residual = 5.67e-05, Final residual = 3.33e-06, No Iterations 1
smoothSolver:  Solving for omega, Initial residual = 6.78e-05, Final residual = 4.44e-06, No Iterations 1

SIMPLE solution converged in 845 iterations

End
"""
    )


def _write_unconverged_log(case_dir: Path) -> None:
    """Write a synthetic log.simpleFoam that hits max iterations without converging."""
    log = case_dir / "log.simpleFoam"
    log.write_text(
        """\
Time = 1200

smoothSolver:  Solving for Ux, Initial residual = 1.00e-04, Final residual = 5.00e-05, No Iterations 3
smoothSolver:  Solving for Uy, Initial residual = 2.00e-04, Final residual = 6.00e-05, No Iterations 3
smoothSolver:  Solving for Uz, Initial residual = 3.00e-04, Final residual = 7.00e-05, No Iterations 3
GAMG:  Solving for p, Initial residual = 8.00e-04, Final residual = 1.50e-04, No Iterations 5
smoothSolver:  Solving for k, Initial residual = 9.00e-04, Final residual = 2.50e-04, No Iterations 3
smoothSolver:  Solving for omega, Initial residual = 1.00e-03, Final residual = 3.50e-04, No Iterations 3

End
"""
    )


# ---------------------------------------------------------------------------
#  parse_convergence_status
# ---------------------------------------------------------------------------


class TestParseConvergenceStatus:
    def test_converged_log_returns_true_and_residuals(self) -> None:
        """A log containing 'SIMPLE solution converged' returns converged=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            _write_converged_log(case_dir)
            result = parse_convergence_status(case_dir)

            assert result["converged"] is True
            assert abs(result["final_residuals"]["p"] - 2.22e-07) < 1e-10
            assert abs(result["final_residuals"]["Ux"] - 5.67e-08) < 1e-11
            assert abs(result["final_residuals"]["k"] - 3.33e-06) < 1e-09
            assert abs(result["final_residuals"]["omega"] - 4.44e-06) < 1e-09

    def test_unconverged_log_returns_false(self) -> None:
        """A log reaching max iterations without 'SIMPLE solution converged' returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            _write_unconverged_log(case_dir)
            result = parse_convergence_status(case_dir)

            assert result["converged"] is False
            assert abs(result["final_residuals"]["p"] - 1.50e-04) < 1e-07
            assert abs(result["final_residuals"]["Ux"] - 5.00e-05) < 1e-08

    def test_missing_log_returns_false_with_nan_residuals(self) -> None:
        """Missing log.simpleFoam returns converged=False with NaN residuals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = parse_convergence_status(Path(tmpdir))

            assert result["converged"] is False

            for field in ("p", "Ux", "k", "omega"):
                assert math.isnan(result["final_residuals"][field])


# ---------------------------------------------------------------------------
#  parse_cd_stats
# ---------------------------------------------------------------------------


def _write_coefficient_dat(path: Path, cd_values: list[float]) -> None:
    """Write a synthetic coefficient.dat with the given Cd values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Time Cd Cs Cl CmRoll CmPitch CmYaw\n"]
    for i, cd in enumerate(cd_values, start=1):
        lines.append(f"{i}    {cd:.6f}  0.001  0.050  0.000  0.000  0.000\n")
    path.write_text("".join(lines))


class TestParseCdStats:
    def test_returns_correct_mean_and_std_for_50_entries(self) -> None:
        """With exactly 50 Cd entries, mean50 and std50 are computed over all of them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            cd_values = [0.20 + 0.001 * i for i in range(50)]
            _write_coefficient_dat(
                case_dir / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
                cd_values,
            )
            mean50, std50 = parse_cd_stats(case_dir)

            assert mean50 is not None
            assert std50 is not None
            expected_mean = sum(cd_values) / len(cd_values)
            assert abs(mean50 - expected_mean) < 1e-6, f"mean50={mean50} vs expected {expected_mean}"
            expected_std = statistics.stdev(cd_values)
            assert abs(std50 - expected_std) < 1e-6, f"std50={std50} vs expected {expected_std}"

    def test_uses_last_50_when_more_entries_exist(self) -> None:
        """With >50 entries, only the last 50 are used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            # First 100 entries have Cd≈1.0 (old, noisy), last 50 have Cd≈0.25 (converged)
            early = [1.0 + 0.01 * i for i in range(100)]
            late = [0.25 + 0.001 * i for i in range(50)]
            _write_coefficient_dat(
                case_dir / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
                early + late,
            )
            mean50, _std50 = parse_cd_stats(case_dir)

            assert mean50 is not None
            # Mean should be close to 0.25, not 1.0
            assert 0.24 < mean50 < 0.28, f"mean50={mean50} should be near 0.25 (last 50)"

    def test_uses_all_entries_when_fewer_than_50(self) -> None:
        """When fewer than 50 entries, computes stats over all available entries (no crash)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            cd_values = [0.18, 0.19, 0.20, 0.21]  # only 4 entries (e.g. early-converging run)
            _write_coefficient_dat(
                case_dir / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
                cd_values,
            )
            mean50, std50 = parse_cd_stats(case_dir)

            assert mean50 is not None
            assert std50 is not None
            assert abs(mean50 - sum(cd_values) / len(cd_values)) < 1e-6

    def test_returns_nan_when_no_coefficient_file(self) -> None:
        """Returns (nan, nan) when no coefficient file is present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mean50, std50 = parse_cd_stats(Path(tmpdir))
            assert mean50 is None or math.isnan(mean50)
            assert std50 is None or math.isnan(std50)

    def test_reads_parallel_coefficient_0_dat_for_stats(self) -> None:
        """Uses coefficient_0.dat (parallel) over coefficient.dat when both exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            coeff_dir = case_dir / "postProcessing" / "forceCoeffs" / "0"
            # Serial (stale) — mean would be 1.0
            _write_coefficient_dat(coeff_dir / "coefficient.dat", [1.0] * 50)
            # Parallel (current) — mean should be 0.25
            _write_coefficient_dat(coeff_dir / "coefficient_0.dat", [0.25] * 50)

            mean50, _std50 = parse_cd_stats(case_dir)

            assert mean50 is not None
            assert abs(mean50 - 0.25) < 1e-4, f"mean50={mean50} should read from _0 file"


class TestProcessSingleRunCdStats:
    def test_pt_contains_cd_stats_fields(self) -> None:
        """process_single_run must write cd_mean50 and cd_std50 to the .pt file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cases_dir = Path(tmpdir) / "cases"
            output_dir = Path(tmpdir) / "output"
            case_dir = cases_dir / "RUN_001"
            case_dir.mkdir(parents=True)

            _create_synthetic_vtk(case_dir)
            _write_converged_log(case_dir)

            # Write synthetic coefficient.dat with 50 entries
            cd_values = [0.30 + 0.001 * i for i in range(50)]
            _write_coefficient_dat(
                case_dir / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
                cd_values,
            )
            # Write a minimal force.dat too
            force_dir = case_dir / "postProcessing" / "forces" / "0"
            force_dir.mkdir(parents=True)
            (force_dir / "force.dat").write_text("# Time total_x total_y total_z\n1200   -150.0  1.0  5.0\n")

            process_single_run(1, cases_dir=str(cases_dir), output_dir=str(output_dir))

            pt_path = output_dir / "euler_run_1" / "data.pt"
            assert pt_path.exists(), "data.pt not created"
            data = torch.load(pt_path, map_location="cpu", weights_only=False)

            assert hasattr(data, "cd_mean50") or "cd_mean50" in data, "data.pt missing cd_mean50"
            assert hasattr(data, "cd_std50") or "cd_std50" in data, "data.pt missing cd_std50"

            cd_mean50 = data.cd_mean50 if hasattr(data, "cd_mean50") else data["cd_mean50"]

            expected_mean = sum(cd_values) / len(cd_values)
            assert abs(float(cd_mean50) - expected_mean) < 1e-4, f"cd_mean50={float(cd_mean50)} vs expected {expected_mean}"
