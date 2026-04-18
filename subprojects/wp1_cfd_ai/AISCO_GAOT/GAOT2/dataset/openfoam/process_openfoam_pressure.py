#!/usr/bin/env python3
"""Process OpenFOAM simulation results into GAOT-compatible .pt files.

Reads VTK output (from foamToVTK) and force reports to produce
.pt files with identical schema to the Fluent extractor.

For each run, extracts:
  - pos:        [N, 3]   vertex coordinates (surface nodes on car_surface)
  - x:          [N, 1]   pressure coefficient at each vertex
  - c:          [N, 3]   outward unit normals
  - face:       [3, F']  triangulated face connectivity
  - wall_shear: [N, 3]   wall shear stress at each vertex
  - pressure:   [N, 1]   raw pressure at each vertex
  - force:      [1]      drag force
  - cd:         [1]      drag coefficient
  - v_inf:      [1]      freestream speed
  - conditioning:[1,7]   [v_inf, extent_xyz, aspect_xyz]
  - filename:   str

Usage:
    python process_openfoam_pressure.py                           # all runs
    python process_openfoam_pressure.py --run-ids 1 2 3           # specific
    python process_openfoam_pressure.py --max-runs 5              # first 5
"""

import argparse
import glob
import logging
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_preprocessing import (
    AIR_DENSITY,
    V_INF_EULER,
    compute_geom_conditioning,
    compute_pressure_coefficient,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
# v040 paths — old CASES / processed_euler_openfoam_cp_full are kept as-is for reference.
CASES_DIR = "/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/CASES_V2"
OUTPUT_DIR = "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler_openfoam_cp_v040"
NUM_WORKERS = 4
DEFAULT_TIME_STEP = 3000

# Boundary edge ratio below which a VTK mesh is treated as effectively closed.
# snappyHexMesh produces minor boundary edges at domain boundaries (typically
# 0.01-0.05% of faces).  Only genuinely open meshes (ratio > 1%) should
# trigger the STL fallback.
_BOUNDARY_EDGE_RATIO_THRESHOLD = 0.01


def _count_boundary_edges(mesh: pv.PolyData) -> int:
    """Count open boundary edges on a triangulated surface mesh.

    Args:
        mesh: Surface mesh to inspect.

    Returns:
        Number of boundary edges present on the surface.
    """
    boundary_edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    return int(boundary_edges.n_cells)


def _ensure_point_fields(mesh: pv.PolyData) -> pv.PolyData:
    """Ensure surface fields are available as point data.

    Args:
        mesh: Surface mesh with OpenFOAM arrays in either point or cell data.

    Returns:
        Mesh with point-data versions of the fields used by the extractor.
    """
    if "p" in mesh.cell_data or "wallShearStress" in mesh.cell_data:
        return mesh.cell_data_to_point_data()
    return mesh


def _recover_surface_mesh(case_dir: Path, source_mesh: pv.PolyData) -> pv.PolyData:
    """Return the VTK surface mesh with point fields ensured.

    Previously this function sampled pressure from the VTK patch onto the
    coarse input STL when boundary-edge ratio exceeded a threshold.  That
    approach was removed because:

    1. The input STL is the *coarsest* geometry source (90× fewer triangles
       than the VTK patch), so sampling onto it discards resolution.
    2. PyVista ``PolyData.sample()`` from 2-D onto 2-D topology returns
       all-zero pressure fields for this geometry class (verified).
    3. snappyHexMesh always produces a topologically closed car_surface patch;
       the fallback has never been triggered in production.

    High boundary-edge ratios are now logged as a WARNING so they remain
    visible in extraction logs without silently replacing the geometry.

    Args:
        case_dir: Path to the OpenFOAM case directory (used only for logging).
        source_mesh: Triangulated surface read from foamToVTK.

    Returns:
        The source mesh with point data fields ensured.
    """
    boundary_edge_count = _count_boundary_edges(source_mesh)
    if boundary_edge_count == 0:
        return _ensure_point_fields(source_mesh)

    n_faces = max(source_mesh.n_cells, 1)
    boundary_ratio = boundary_edge_count / n_faces

    if boundary_ratio >= _BOUNDARY_EDGE_RATIO_THRESHOLD:
        logger.warning(
            "%s: VTK patch has %d boundary edges (%.3f%% of %d faces) — "
            "exceeds %.1f%% threshold; retaining VTK mesh (STL fallback removed)",
            case_dir.name,
            boundary_edge_count,
            boundary_ratio * 100,
            n_faces,
            _BOUNDARY_EDGE_RATIO_THRESHOLD * 100,
        )
    else:
        logger.info(
            "%s: VTK patch has %d boundary edges (%.3f%% of %d faces) — "
            "below %.1f%% threshold, normal snappyHexMesh artifact",
            case_dir.name,
            boundary_edge_count,
            boundary_ratio * 100,
            n_faces,
            _BOUNDARY_EDGE_RATIO_THRESHOLD * 100,
        )

    return _ensure_point_fields(source_mesh)


# ---------------------------------------------------------------------------
#  Convergence status parser
# ---------------------------------------------------------------------------

_RESIDUAL_FIELDS = ("p", "Ux", "k", "omega")


# ---------------------------------------------------------------------------
#  Geometry & pressure quality helpers (M2 — extraction telemetry)
# ---------------------------------------------------------------------------

_STL_QUALITY_WARN_THRESHOLD = 10_000  # n_pts below this triggers a WARNING


def _compute_stl_quality_metrics(case_dir: Path) -> dict[str, float | int]:
    """Compute geometry quality metrics from the case's input STL.

    Reads ``constant/triSurface/car_surface.stl`` and returns point count,
    median edge length (mm), and distinct normal direction count.  All values
    are ``-1`` when the STL is absent or unreadable.

    Args:
        case_dir: OpenFOAM case directory.

    Returns:
        Dict with keys ``n_stl_pts``, ``stl_median_edge_mm``,
        ``distinct_normal_dirs``.
    """
    stl_path = case_dir / "constant" / "triSurface" / "car_surface.stl"
    defaults: dict[str, float | int] = {
        "n_stl_pts": -1,
        "stl_median_edge_mm": -1.0,
        "distinct_normal_dirs": -1,
    }
    if not stl_path.exists():
        return defaults

    try:
        stl = pv.read(str(stl_path)).triangulate().clean()
        n_pts = int(stl.n_points)
        pos = np.array(stl.points)
        if stl.n_cells > 0:
            faces = stl.faces.reshape(-1, 4)[:, 1:]
            v0, v1, v2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
            edge_lengths = np.concatenate([
                np.linalg.norm(v1 - v0, axis=1),
                np.linalg.norm(v2 - v1, axis=1),
                np.linalg.norm(v0 - v2, axis=1),
            ])
            median_edge_mm = float(np.median(edge_lengths) * 1000.0)

            # Count distinct normal directions (0.01 rad tolerance)
            face_normals = np.cross(v1 - v0, v2 - v0)
            norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            unit_normals = face_normals / norms
            # Round to 2 decimal places to cluster similar normals
            rounded = np.round(unit_normals, 2)
            distinct_dirs = int(len(np.unique(rounded, axis=0)))
        else:
            median_edge_mm = -1.0
            distinct_dirs = -1

        return {
            "n_stl_pts": n_pts,
            "stl_median_edge_mm": round(median_edge_mm, 1),
            "distinct_normal_dirs": distinct_dirs,
        }
    except Exception as exc:
        logger.warning("%s: failed to compute STL quality metrics: %s", case_dir.name, exc)
        return defaults


def _compute_pressure_validity(pressure: np.ndarray) -> dict[str, float]:
    """Compute pressure field validity metrics.

    Args:
        pressure: 1-D pressure array (raw kinematic pressure, m²/s²).

    Returns:
        Dict with keys ``p_min``, ``p_max``, ``p_zero_frac``.
    """
    if len(pressure) == 0:
        return {"p_min": float("nan"), "p_max": float("nan"), "p_zero_frac": 1.0}
    return {
        "p_min": float(np.min(pressure)),
        "p_max": float(np.max(pressure)),
        "p_zero_frac": float(np.sum(pressure == 0.0) / len(pressure)),
    }


def parse_convergence_status(case_dir: Path) -> dict[str, bool | dict[str, float]]:
    """Parse convergence status and final residuals from log.simpleFoam.

    Args:
        case_dir: OpenFOAM case directory containing ``log.simpleFoam``.

    Returns:
        Dictionary with keys:
        - ``converged`` (bool): True if the log contains
          ``SIMPLE solution converged``.
        - ``final_residuals`` (dict[str, float]): Final residual for each
          of ``p``, ``Ux``, ``k``, ``omega``.  Values are ``float('nan')``
          when the log is missing or a field was not found.
    """
    log_path = case_dir / "log.simpleFoam"
    nan_residuals = {f: float("nan") for f in _RESIDUAL_FIELDS}

    if not log_path.exists():
        logger.warning(
            "%s: log.simpleFoam not found — marking as unconverged with NaN residuals",
            case_dir.name,
        )
        return {"converged": False, "final_residuals": nan_residuals}

    text = log_path.read_text()
    converged = "SIMPLE solution converged" in text

    # Parse the *last* final residual for each field.
    # simpleFoam format: Solving for <field>, Initial residual = ..., Final residual = <val>, ...

    final_residuals: dict[str, float] = {}
    for field in _RESIDUAL_FIELDS:
        pattern = rf"Solving for {re.escape(field)},.*?Final residual = ([0-9eE.+-]+)"
        matches = re.findall(pattern, text)
        if matches:
            final_residuals[field] = float(matches[-1])
        else:
            final_residuals[field] = float("nan")

    return {"converged": converged, "final_residuals": final_residuals}


# ---------------------------------------------------------------------------
#  Force report parser
# ---------------------------------------------------------------------------
def _find_postprocessing_file(base_dir: Path, serial_name: str, parallel_name: str) -> Path | None:
    """Find the best postProcessing data file, preferring parallel over serial.

    OpenFOAM parallel MPI runs write force/coefficient data with a ``_0``
    suffix (e.g. ``coefficient_0.dat``).  When both serial and parallel
    files coexist the ``_0`` variant contains the most recent data.

    Within each naming variant the *latest* time subdirectory is selected
    (matching the existing restart-directory logic).

    Args:
        base_dir: e.g. ``case_dir / "postProcessing" / "forceCoeffs"``.
        serial_name: Filename for serial runs (e.g. ``"coefficient.dat"``).
        parallel_name: Filename for parallel runs (e.g. ``"coefficient_0.dat"``).

    Returns:
        Path to the best file, or ``None`` if nothing is found.
    """
    if not base_dir.is_dir():
        return None

    # Prefer parallel (_0) files when they exist
    parallel_candidates = sorted(
        base_dir.glob(f"*/{parallel_name}"),
        key=lambda p: float(p.parent.name),
    )
    if parallel_candidates:
        return parallel_candidates[-1]

    serial_candidates = sorted(
        base_dir.glob(f"*/{serial_name}"),
        key=lambda p: float(p.parent.name),
    )
    if serial_candidates:
        return serial_candidates[-1]

    return None


def parse_cd_stats(case_dir: Path) -> tuple[float | None, float | None]:
    """Compute mean and std of the last 50 Cd values from forceCoeffs coefficient file.

    Uses ``min(50, len(entries))`` entries so runs that converge early (few
    write times) are handled gracefully without raising an error.

    The parallel ``coefficient_0.dat`` is preferred over ``coefficient.dat``
    when both exist, consistent with ``parse_openfoam_forces``.

    Args:
        case_dir: Path to the OpenFOAM case directory.

    Returns:
        Tuple of ``(cd_mean50, cd_std50)``. Both are ``None`` when the
        coefficient file is absent or contains no valid numeric rows.
    """
    coeff_file = _find_postprocessing_file(
        case_dir / "postProcessing" / "forceCoeffs",
        serial_name="coefficient.dat",
        parallel_name="coefficient_0.dat",
    )
    if coeff_file is None or not coeff_file.exists():
        return None, None

    cd_entries: list[float] = []
    with open(coeff_file) as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    cd_entries.append(float(parts[1]))
                except ValueError:
                    continue

    if not cd_entries:
        return None, None

    window = cd_entries[-min(50, len(cd_entries)) :]
    mean_val = sum(window) / len(window)
    if len(window) > 1:
        variance = sum((v - mean_val) ** 2 for v in window) / (len(window) - 1)
        std_val = variance**0.5
    else:
        std_val = 0.0

    return mean_val, std_val


def parse_openfoam_forces(case_dir: Path) -> tuple[float | None, float | None]:
    """Parse drag coefficient and drag force from OpenFOAM postProcessing.

    Reads the forceCoeffs coefficient.dat (or coefficient_0.dat for parallel
    MPI runs) for Cd, and forces force.dat (or force_0.dat) for the actual
    drag force.  When both serial and parallel files coexist, the parallel
    ``_0`` variant is preferred — it contains the most recent data.

    Args:
        case_dir: Path to the OpenFOAM case directory.

    Returns:
        Tuple of (cd, drag_force). Either may be None if not found.
    """
    cd_val: float | None = None
    force_val: float | None = None

    # Parse Cd from forceCoeffs
    coeff_file = _find_postprocessing_file(
        case_dir / "postProcessing" / "forceCoeffs",
        serial_name="coefficient.dat",
        parallel_name="coefficient_0.dat",
    )
    if coeff_file is not None and coeff_file.exists():
        with open(coeff_file) as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        cd_val = float(parts[1])
                    except ValueError:
                        continue

    # Parse drag force from forces
    force_file = _find_postprocessing_file(
        case_dir / "postProcessing" / "forces",
        serial_name="force.dat",
        parallel_name="force_0.dat",
    )
    if force_file is not None and force_file.exists():
        with open(force_file) as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                # OpenFOAM 2306 force.dat columns (space-delimited, no parentheses):
                #   time  total_x  total_y  total_z  pressure_x  pressure_y  pressure_z
                #         viscous_x  viscous_y  viscous_z
                # dragDir = (-1 0 0), so F_drag = -total_x (numbers[1]).
                numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", stripped)
                if len(numbers) >= 2:
                    try:
                        force_val = -float(numbers[1])  # negate total_x
                    except (ValueError, IndexError):
                        continue

    return cd_val, force_val


# ---------------------------------------------------------------------------
#  VTK patch data extraction
# ---------------------------------------------------------------------------
def extract_patch_vtk(
    case_dir: Path,
    time_step: int = DEFAULT_TIME_STEP,
    patch_name: str = "car_surface",
) -> dict:
    """Extract mesh and field data from foamToVTK output for a patch.

    Args:
        case_dir: Path to the OpenFOAM case directory.
        time_step: Time step to extract (default: 1200).
        patch_name: Name of the boundary patch.

    Returns:
        Dict with keys: vertices [N,3], faces [F,3], pressure [N],
        wall_shear [N,3] (all per-vertex after interpolation).

    Raises:
        FileNotFoundError: If the VTK file is not found.
    """
    # OpenFOAM 2306 foamToVTK writes boundary patches to:
    #   VTK/{case_name}_{time_step}/boundary/{patch_name}.vtp
    run_name = case_dir.name  # e.g. "RUN_001"
    vtk_path = case_dir / "VTK" / f"{run_name}_{time_step}" / "boundary" / f"{patch_name}.vtp"

    if not vtk_path.exists():
        # Fall back to the latest available VTK time directory.
        # This handles cases where residualControl stopped the solver at a
        # different iteration than the default time_step (e.g. t=546 instead of 600).
        vtk_base = case_dir / "VTK"
        candidates = sorted(
            vtk_base.glob(f"{run_name}_*/boundary/{patch_name}.vtp"),
            key=lambda p: int(p.parent.parent.name.split("_")[-1]),
        )
        if candidates:
            vtk_path = candidates[-1]
            actual_time = int(vtk_path.parent.parent.name.split("_")[-1])
            logger.info(
                "%s: requested t=%d not found; using latest available t=%d",
                run_name,
                time_step,
                actual_time,
            )
        else:
            raise FileNotFoundError(
                f"VTK file not found at {vtk_path}. Ensure foamToVTK was run with -patches '({patch_name})' -latestTime"
            )

    mesh = pv.read(str(vtk_path)).triangulate().clean()
    mesh = _recover_surface_mesh(case_dir, mesh)

    # Extract vertices
    vertices = np.array(mesh.points, dtype=np.float32)

    # Extract triangulated faces
    # pyvista stores faces as [n, v0, v1, v2, n, v0, v1, ...]
    faces_pv = mesh.faces.reshape(-1, 4)[:, 1:4]  # Drop the "3" prefix per tri
    faces = np.array(faces_pv, dtype=np.int64)

    # Extract pressure — may be cell data or point data
    if "p" in mesh.point_data:
        pressure = np.array(mesh.point_data["p"], dtype=np.float32)
    elif "p" in mesh.cell_data:
        # Interpolate cell data to point data
        mesh_interp = mesh.cell_data_to_point_data()
        pressure = np.array(mesh_interp.point_data["p"], dtype=np.float32)
    else:
        logger.warning("No pressure field found in VTK, using zeros")
        pressure = np.zeros(len(vertices), dtype=np.float32)

    # Extract wall shear stress
    if "wallShearStress" in mesh.point_data:
        wall_shear = np.array(mesh.point_data["wallShearStress"], dtype=np.float32)
    elif "wallShearStress" in mesh.cell_data:
        mesh_interp = mesh.cell_data_to_point_data()
        wall_shear = np.array(mesh_interp.point_data["wallShearStress"], dtype=np.float32)
    else:
        logger.warning("No wallShearStress field found in VTK, using zeros")
        wall_shear = np.zeros((len(vertices), 3), dtype=np.float32)

    return {
        "vertices": vertices,
        "faces": faces,
        "pressure": pressure,
        "wall_shear": wall_shear,
    }


def _compute_vertex_normals(pos: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Compute area-weighted vertex normals from triangulated mesh.

    Args:
        pos: Vertex positions [N, 3].
        triangles: Triangle face list [F, 3].

    Returns:
        Unit normals [N, 3].
    """
    v0 = pos[triangles[:, 0]]
    v1 = pos[triangles[:, 1]]
    v2 = pos[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    vertex_normals = np.zeros_like(pos)
    for j in range(3):
        np.add.at(vertex_normals, triangles[:, j], face_normals)

    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (vertex_normals / norms).astype(np.float32)


# ---------------------------------------------------------------------------
#  Single-run processor
# ---------------------------------------------------------------------------
def process_single_run(
    run_id: int,
    cases_dir: str = CASES_DIR,
    output_dir: str = OUTPUT_DIR,
    time_step: int = DEFAULT_TIME_STEP,
) -> str | None:
    """Process one OpenFOAM run into a GAOT-compatible .pt file.

    Args:
        run_id: Numeric run ID (e.g. 1 for RUN_001).
        cases_dir: Parent directory containing RUN_XXX case directories.
        output_dir: Directory for .pt output.
        time_step: Time step to extract.

    Returns:
        Save name on success, None on failure.
    """
    case_dir = Path(cases_dir) / f"RUN_{run_id:03d}"

    if not case_dir.exists():
        logger.warning("RUN_%03d: case directory not found at %s", run_id, case_dir)
        return None

    # Extract patch data from VTK
    try:
        patch_data = extract_patch_vtk(case_dir, time_step)
    except FileNotFoundError as e:
        logger.warning("RUN_%03d: %s", run_id, e)
        return None

    pos_np = patch_data["vertices"]
    faces_np = patch_data["faces"]
    pressure_np = patch_data["pressure"]
    wall_shear_np = patch_data["wall_shear"]

    # Compute normals
    normals_np = _compute_vertex_normals(pos_np, faces_np)

    # Parse forces
    cd_val, force_val = parse_openfoam_forces(case_dir)
    if cd_val is None:
        cd_val = float("nan")
    if force_val is None:
        force_val = float("nan")

    # Cd stability stats over last 50 iterations
    cd_mean50_val, cd_std50_val = parse_cd_stats(case_dir)
    cd_mean50_float = cd_mean50_val if cd_mean50_val is not None else float("nan")
    cd_std50_float = cd_std50_val if cd_std50_val is not None else float("nan")

    # Compute Cp.
    # simpleFoam stores kinematic pressure p_kin = p_static / rho  [m²/s²].
    # compute_pressure_coefficient divides by (0.5 * rho * v_inf²).
    # For kinematic pressure the correct denominator is (0.5 * v_inf²), i.e. rho=1.
    # Passing rho=1.0 gives: Cp = p_kin / (0.5 * v_inf²) = p_static / (0.5 * rho * v_inf²) ✓
    pressure_tensor = torch.tensor(pressure_np, dtype=torch.float32).unsqueeze(-1)
    cp_tensor = compute_pressure_coefficient(pressure_tensor, V_INF_EULER, rho=1.0)

    pos_tensor = torch.tensor(pos_np, dtype=torch.float32)
    geom_extent, geom_aspect, conditioning = compute_geom_conditioning(pos_tensor, V_INF_EULER)

    # Sanity-check: force / (Q_inf * frontal_area) should match Cd from coefficient.dat.
    # geom_extent = [x_range, y_range, z_range]; frontal area = y_range * z_range.
    if not (np.isnan(cd_val) or np.isnan(force_val)):
        q_inf = 0.5 * AIR_DENSITY * V_INF_EULER**2
        aref = float(geom_extent[1]) * float(geom_extent[2])  # Y x Z frontal area
        if aref > 0:
            cd_from_force = force_val / (q_inf * aref)
            rel_err = abs(cd_from_force - cd_val) / (abs(cd_val) + 1e-12)
            if rel_err > 0.02:
                logger.warning(
                    "RUN_%03d: force/Cd sanity check failed — "
                    "force=%.4f N, cd=%.5f, cd_from_force=%.5f (%.1f%% off). "
                    "Check parse_openfoam_forces column mapping.",
                    run_id,
                    force_val,
                    cd_val,
                    cd_from_force,
                    rel_err * 100,
                )

    # -----------------------------------------------------------------------
    #  Extraction quality telemetry (M2)
    # -----------------------------------------------------------------------
    stl_quality = _compute_stl_quality_metrics(case_dir)
    n_stl_pts = stl_quality["n_stl_pts"]
    logger.info(
        "RUN_%03d: geometry_quality n_stl_pts=%d stl_median_edge_mm=%.1f distinct_normal_dirs=%d",
        run_id,
        n_stl_pts,
        stl_quality["stl_median_edge_mm"],
        stl_quality["distinct_normal_dirs"],
    )
    if isinstance(n_stl_pts, int) and n_stl_pts < _STL_QUALITY_WARN_THRESHOLD:
        logger.warning(
            "RUN_%03d: geometry_quality_low n_stl_pts=%d < threshold=%d — "
            "coarse input geometry; surface faceting expected",
            run_id,
            n_stl_pts,
            _STL_QUALITY_WARN_THRESHOLD,
        )

    p_validity = _compute_pressure_validity(pressure_np)
    logger.info(
        "RUN_%03d: pressure_validity p_min=%.2f p_max=%.2f p_zero_frac=%.4f",
        run_id,
        p_validity["p_min"],
        p_validity["p_max"],
        p_validity["p_zero_frac"],
    )
    if p_validity["p_zero_frac"] > 0.5:
        logger.error(
            "RUN_%03d: pressure_all_zero p_zero_frac=%.4f > 0.5 — "
            "likely sampling failure; .pt will contain invalid Cp fields",
            run_id,
            p_validity["p_zero_frac"],
        )

    # Build PyG Data
    data = Data(
        pos=pos_tensor,
        x=cp_tensor,
        c=torch.tensor(normals_np, dtype=torch.float32),
        face=torch.tensor(faces_np.T, dtype=torch.long),  # [3, F]
        wall_shear=torch.tensor(wall_shear_np, dtype=torch.float32),
        force=torch.tensor([force_val], dtype=torch.float32),
        cd=torch.tensor([cd_val], dtype=torch.float32),
    )
    data.pressure = pressure_tensor
    data.cp = cp_tensor
    data.v_inf = torch.tensor([V_INF_EULER], dtype=torch.float32)
    data.geom_extent = geom_extent
    data.geom_aspect = geom_aspect
    data.conditioning = conditioning
    data.filename = f"RUN_{run_id:03d}"
    data.cd_mean50 = torch.tensor([cd_mean50_float], dtype=torch.float32)
    data.cd_std50 = torch.tensor([cd_std50_float], dtype=torch.float32)

    # Convergence telemetry
    conv = parse_convergence_status(case_dir)
    data.converged = torch.tensor([conv["converged"]], dtype=torch.bool)
    residuals = conv["final_residuals"]
    assert isinstance(residuals, dict)
    data.final_residuals = torch.tensor(
        [residuals["p"], residuals["Ux"], residuals["k"], residuals["omega"]],
        dtype=torch.float32,
    )

    save_name = f"euler_run_{run_id}"
    save_dir = os.path.join(output_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(data, os.path.join(save_dir, "data.pt"))
    logger.info("RUN_%03d: saved to %s", run_id, save_dir)
    return save_name


# ---------------------------------------------------------------------------
#  Batch processing
# ---------------------------------------------------------------------------
def process_all_runs(
    cases_dir: str = CASES_DIR,
    output_dir: str = OUTPUT_DIR,
    run_ids: list[int] | None = None,
    max_runs: int | None = None,
    num_workers: int = NUM_WORKERS,
    time_step: int = DEFAULT_TIME_STEP,
) -> None:
    """Process all (or selected) OpenFOAM runs into .pt files.

    Args:
        cases_dir: Parent directory containing RUN_XXX case directories.
        output_dir: Directory for .pt output.
        run_ids: Specific run numbers. If None, auto-detect.
        max_runs: Process at most this many runs.
        num_workers: Number of parallel workers.
        time_step: Time step to extract.
    """
    os.makedirs(output_dir, exist_ok=True)

    if run_ids is None:
        case_dirs = sorted(glob.glob(os.path.join(cases_dir, "RUN_*")))
        run_ids = []
        for cd in case_dirs:
            m = re.search(r"RUN_(\d+)", os.path.basename(cd))
            if m:
                run_ids.append(int(m.group(1)))
        run_ids.sort()

    if max_runs is not None:
        run_ids = run_ids[:max_runs]

    print(f"Processing {len(run_ids)} OpenFOAM runs with {num_workers} workers...")
    print(f"  Cases:  {cases_dir}")
    print(f"  Output: {output_dir}")

    order_list: list[str] = []

    if num_workers <= 1:
        for rid in tqdm(run_ids, desc="Processing runs"):
            try:
                result = process_single_run(rid, cases_dir, output_dir, time_step)
                if result is not None:
                    order_list.append(result)
            except Exception as exc:
                print(f"  [FAIL] RUN_{rid:03d}: {exc}")
                traceback.print_exc()
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_run, rid, cases_dir, output_dir, time_step): rid for rid in run_ids}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing runs"):
                rid = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        order_list.append(result)
                except Exception as exc:
                    print(f"  [FAIL] RUN_{rid:03d}: {exc}")
                    traceback.print_exc()

    order_list.sort(key=lambda name: int(name.split("_")[-1]))

    order_path = os.path.join(output_dir, "order_.txt")
    with open(order_path, "w") as f:
        f.writelines(f"{name}\n" for name in order_list)

    print(f"\nDone! Processed {len(order_list)}/{len(run_ids)} runs.")
    print(f"Order file: {order_path}")


def main() -> None:
    """CLI entry point for OpenFOAM .pt extraction."""
    parser = argparse.ArgumentParser(description="Extract OpenFOAM results to .pt files")
    parser.add_argument("--cases-dir", type=str, default=CASES_DIR, help="OpenFOAM cases directory")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for .pt files")
    parser.add_argument("--run-ids", type=int, nargs="+", default=None, help="Specific run IDs")
    parser.add_argument("--max-runs", type=int, default=None, help="Max runs to process")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Parallel workers")
    parser.add_argument("--time-step", type=int, default=DEFAULT_TIME_STEP, help="Time step to extract")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    process_all_runs(
        cases_dir=args.cases_dir,
        output_dir=args.output_dir,
        run_ids=args.run_ids,
        max_runs=args.max_runs,
        num_workers=args.workers,
        time_step=args.time_step,
    )


if __name__ == "__main__":
    main()
