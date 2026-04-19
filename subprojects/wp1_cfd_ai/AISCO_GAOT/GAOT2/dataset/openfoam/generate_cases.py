#!/usr/bin/env python3
"""Generate OpenFOAM case directories from template and geometry manifest.

For each converted PLY→STL run in the geometry manifest, creates a
complete OpenFOAM case directory with per-run domain sizing, BOI box,
and STL placement.

Usage:
    python generate_cases.py                              # all converted runs (Euler, v_inf=25)
    python generate_cases.py --run-ids 1 2 3              # specific runs
    python generate_cases.py --vinf 1 \\
        --manifest .../OPENFOAM/ahmed_geometry_manifest.json \\
        --output .../OPENFOAM/AHMED_CASES \\
        --cases-manifest .../OPENFOAM/ahmed_cases_manifest.json
"""

import argparse
import json
import logging
import math
import shutil
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

# Background cell size matching Fluent HexMaxCellLength
_CELL_SIZE = 0.164

# Domain padding multipliers (car-lengths / car-widths / car-heights)
_UPSTREAM_LENGTHS = 2.0
_DOWNSTREAM_LENGTHS = 5.0
_LATERAL_WIDTHS = 2.0
_VERTICAL_HEIGHTS = 3.0

# BOI box padding (meters)
_BOI_UPSTREAM = 0.5
_BOI_DOWNSTREAM = 1.5
_BOI_LATERAL = 0.3
_BOI_VERTICAL = 0.3

# Default MPI tasks for decomposition
# Mesh and solve use the same proc count to avoid decomposeParDict swap.
_NPROCS = 128

# Freestream velocity for Euler (AISCO) cases.  Ahmed cases use V_INF_AHMED=1.
_DEFAULT_V_INF = 25.0

# Turbulence inlet BCs are derived from v_inf to preserve turbulence intensity
# and length scale across datasets.  The reference values at v_inf=25 m/s are:
#   k_ref = 6.25e-4 m²/s², omega_ref = 25 s⁻¹
# Both scale with v_inf so that TI = sqrt(2k/3)/v_inf stays constant and the
# turbulent length scale L = sqrt(k) / (C_mu^0.25 * omega) is geometry-invariant.
#   k(v) = k_ref * (v / v_ref)²  =>  k(v) = 1e-6 * v²
#   omega(v) = omega_ref * (v / v_ref)  =>  omega(v) = v
_K_INLET_V_REF = 6.25e-4
_OMEGA_INLET_V_REF = 25.0
_V_REF = 25.0

# Default paths on Euler
_DEFAULT_MANIFEST = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/geometry_manifest.json")
_DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "case_template"
_DEFAULT_OUTPUT = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/CASES_v3")
_DEFAULT_CASES_MANIFEST = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/cases_manifest.json")


def compute_domain_extents(
    bounds_min: list[float],
    bounds_max: list[float],
) -> dict[str, float | int]:
    """Compute wind tunnel domain extents from geometry bounds.

    Domain extends:
    - Upstream (-x): ~2 car-lengths before geometry min
    - Downstream (+x): ~5 car-lengths after geometry max
    - Lateral (y): ±2 car-widths from geometry center
    - Vertical (z): 0 (ground) to ~3 car heights

    Args:
        bounds_min: [x, y, z] minimum bounds of geometry (meters).
        bounds_max: [x, y, z] maximum bounds of geometry (meters).

    Returns:
        Dict with xMin, xMax, yMin, yMax, zMin, zMax, xCells, yCells, zCells.
    """
    car_length = bounds_max[0] - bounds_min[0]
    car_width = bounds_max[1] - bounds_min[1]
    car_height = bounds_max[2] - bounds_min[2]

    x_min = bounds_min[0] - _UPSTREAM_LENGTHS * car_length
    x_max = bounds_max[0] + _DOWNSTREAM_LENGTHS * car_length
    y_min = bounds_min[1] - _LATERAL_WIDTHS * car_width
    y_max = bounds_max[1] + _LATERAL_WIDTHS * car_width
    z_min = 0.0
    z_max = max(bounds_max[2] + _VERTICAL_HEIGHTS * car_height, 5.0)  # at least 5m

    x_cells = max(1, math.ceil((x_max - x_min) / _CELL_SIZE))
    y_cells = max(1, math.ceil((y_max - y_min) / _CELL_SIZE))
    z_cells = max(1, math.ceil((z_max - z_min) / _CELL_SIZE))

    return {
        "xMin": round(x_min, 4),
        "xMax": round(x_max, 4),
        "yMin": round(y_min, 4),
        "yMax": round(y_max, 4),
        "zMin": z_min,
        "zMax": round(z_max, 4),
        "xCells": x_cells,
        "yCells": y_cells,
        "zCells": z_cells,
    }


def compute_boi_box(
    bounds_min: list[float],
    bounds_max: list[float],
) -> dict[str, float]:
    """Compute BOI (body of influence) refinement box from geometry bounds.

    Applies fixed padding around the geometry bounding box:
    - Upstream: +0.5m
    - Downstream: +1.5m
    - Lateral: +0.3m each side
    - Vertical: +0.3m above, ground at 0

    Args:
        bounds_min: [x, y, z] minimum bounds of geometry (meters).
        bounds_max: [x, y, z] maximum bounds of geometry (meters).

    Returns:
        Dict with xMin, xMax, yMin, yMax, zMin, zMax.
    """
    return {
        "xMin": round(bounds_min[0] - _BOI_UPSTREAM, 4),
        "xMax": round(bounds_max[0] + _BOI_DOWNSTREAM, 4),
        "yMin": round(bounds_min[1] - _BOI_LATERAL, 4),
        "yMax": round(bounds_max[1] + _BOI_LATERAL, 4),
        "zMin": 0.0,
        "zMax": round(bounds_max[2] + _BOI_VERTICAL, 4),
    }


def _replace_placeholders(content: str, replacements: Mapping[str, str | float | int]) -> str:
    """Replace __PLACEHOLDER__ tokens in file content.

    Args:
        content: File content with placeholders.
        replacements: Mapping of placeholder name (without __) → value.

    Returns:
        Content with placeholders replaced.
    """
    for key, value in replacements.items():
        content = content.replace(f"__{key}__", str(value))
    return content


def compute_turbulence_inlet(v_inf: float) -> tuple[float, float]:
    """Compute inlet turbulence BCs (k, omega) for a given freestream speed.

    Scales from the reference Euler values so that turbulence intensity
    and length scale remain constant across v_inf settings.

    Args:
        v_inf: Freestream speed in m/s.

    Returns:
        Tuple of (k_inlet, omega_inlet) in SI units.
    """
    k_inlet = _K_INLET_V_REF * (v_inf / _V_REF) ** 2
    omega_inlet = _OMEGA_INLET_V_REF * (v_inf / _V_REF)
    return k_inlet, omega_inlet


def instantiate_case(
    run_id: str,
    stl_path: Path,
    bounds_min: list[float],
    bounds_max: list[float],
    template_dir: Path,
    output_dir: Path,
    v_inf: float = _DEFAULT_V_INF,
    nprocs: int = _NPROCS,
) -> dict:
    """Create a complete OpenFOAM case directory for one run.

    Args:
        run_id: Run identifier, e.g. "RUN_001".
        stl_path: Path to the STL geometry file.
        bounds_min: [x, y, z] minimum geometry bounds in meters.
        bounds_max: [x, y, z] maximum geometry bounds in meters.
        template_dir: Path to the case template directory.
        output_dir: Parent directory for case directories.
        v_inf: Freestream speed in m/s.  Controls U, k, omega BCs and
            the forceCoeffs magUInf.  Defaults to 25.0 (Euler dataset).
        nprocs: Number of MPI processes for both meshing and solving.

    Returns:
        Dict with run_id, case_path, status.
    """
    case_dir = output_dir / run_id
    entry = {"run_id": run_id, "case_path": str(case_dir), "status": "pending"}

    # Copy template
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(template_dir, case_dir)

    # Place STL in constant/triSurface/
    tri_surface_dir = case_dir / "constant" / "triSurface"
    tri_surface_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stl_path, tri_surface_dir / "car_surface.stl")

    # Compute domain and BOI
    domain = compute_domain_extents(bounds_min, bounds_max)
    boi = compute_boi_box(bounds_min, bounds_max)

    # Geometry extents for reference area/length in controlDict
    car_length = bounds_max[0] - bounds_min[0]
    car_width = bounds_max[1] - bounds_min[1]
    car_height = bounds_max[2] - bounds_min[2]
    # Frontal area approximation (width x height)
    a_ref = round(car_width * car_height, 4)
    l_ref = round(car_length, 4)

    # locationInMesh: a point inside the domain but outside the car
    # Place it upstream of the geometry, at mid-height
    loc_x = round(bounds_min[0] - 1.0, 4)
    loc_y = round((bounds_min[1] + bounds_max[1]) / 2.0, 4)
    loc_z = round((bounds_min[2] + bounds_max[2]) / 2.0, 4)

    # Turbulence inlet BCs scaled from reference v_inf=25 values
    k_inlet, omega_inlet = compute_turbulence_inlet(v_inf)

    # Build replacement dict for all placeholders
    replacements = {
        # blockMeshDict
        "XMIN": domain["xMin"],
        "XMAX": domain["xMax"],
        "YMIN": domain["yMin"],
        "YMAX": domain["yMax"],
        "ZMAX": domain["zMax"],
        "XCELLS": domain["xCells"],
        "YCELLS": domain["yCells"],
        "ZCELLS": domain["zCells"],
        # snappyHexMeshDict BOI
        "BOI_XMIN": boi["xMin"],
        "BOI_XMAX": boi["xMax"],
        "BOI_YMIN": boi["yMin"],
        "BOI_YMAX": boi["yMax"],
        "BOI_ZMIN": boi["zMin"],
        "BOI_ZMAX": boi["zMax"],
        # snappyHexMeshDict location in mesh
        "LOCATION_IN_MESH_X": loc_x,
        "LOCATION_IN_MESH_Y": loc_y,
        "LOCATION_IN_MESH_Z": loc_z,
        # controlDict / forceCoeffs
        "LREF": l_ref,
        "AREF": a_ref,
        "VINF": v_inf,
        # Initial / inlet boundary conditions
        "K_INLET": f"{k_inlet:.6e}",
        "OMEGA_INLET": f"{omega_inlet:.6g}",
        # decomposeParDict
        "NPROCS": nprocs,
    }

    # Replace placeholders in all text files
    for file_path in case_dir.rglob("*"):
        if file_path.is_file():
            try:
                content = file_path.read_text()
            except UnicodeDecodeError:
                continue  # skip binary files (e.g. STL)
            new_content = _replace_placeholders(content, replacements)
            if new_content != content:
                file_path.write_text(new_content)

    entry["status"] = "created"
    logger.info("%s: case created at %s", run_id, case_dir)
    return entry


def generate_all_cases(
    manifest_path: Path,
    template_dir: Path,
    output_dir: Path,
    v_inf: float = _DEFAULT_V_INF,
    run_ids: list[int] | None = None,
    cases_manifest_path: Path | None = None,
) -> list[dict]:
    """Generate OpenFOAM cases for all converted runs in the manifest.

    Args:
        manifest_path: Path to geometry_manifest.json from ply_to_stl.
        template_dir: Path to case template directory.
        output_dir: Parent directory for case directories.
        v_inf: Freestream speed in m/s.  All cases in this batch use the
            same v_inf.  Defaults to 25.0 (Euler dataset).
        run_ids: Optional list of specific run IDs to process.
        cases_manifest_path: Path to write cases manifest JSON.

    Returns:
        List of case entry dicts with run_id, case_path, status.
    """
    with open(manifest_path) as f:
        geometry_manifest = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for entry in geometry_manifest:
        if entry["status"] != "converted":
            logger.info("%s: skipping (status=%s)", entry["run_id"], entry["status"])
            continue

        # Filter by run_ids if specified
        if run_ids is not None:
            # Extract numeric ID from "RUN_001" format
            run_num = int(entry["run_id"].split("_")[1])
            if run_num not in run_ids:
                continue

        result = instantiate_case(
            run_id=entry["run_id"],
            stl_path=Path(entry["stl_path"]),
            bounds_min=entry["bounds_min"],
            bounds_max=entry["bounds_max"],
            template_dir=template_dir,
            output_dir=output_dir,
            v_inf=v_inf,
        )
        results.append(result)

    # Append-safe manifest: merge new results with any existing entries
    if cases_manifest_path is None:
        cases_manifest_path = output_dir.parent / "cases_manifest.json"
    cases_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if cases_manifest_path.exists():
        with open(cases_manifest_path) as f:
            existing = json.load(f)
    merged = {e["run_id"]: e for e in existing}
    for r in results:
        merged[r["run_id"]] = r
    all_entries = list(merged.values())
    with open(cases_manifest_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    logger.info("Generated %d cases, manifest at %s", len(results), cases_manifest_path)
    return results


def main() -> None:
    """CLI entry point for case generation."""
    _AHMED_MANIFEST = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/ahmed_geometry_manifest.json")
    _AHMED_OUTPUT = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/AHMED_CASES")
    _AHMED_CASES_MANIFEST = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/ahmed_cases_manifest.json")

    parser = argparse.ArgumentParser(description="Generate OpenFOAM cases from geometry manifest")
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST, help="Path to geometry_manifest.json")
    parser.add_argument("--template", type=Path, default=_DEFAULT_TEMPLATE, help="Path to case template directory")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help="Output directory for cases")
    parser.add_argument("--cases-manifest", type=Path, default=_DEFAULT_CASES_MANIFEST, help="Output cases manifest path")
    parser.add_argument(
        "--vinf",
        type=float,
        default=_DEFAULT_V_INF,
        help=(
            f"Freestream speed in m/s (default: {_DEFAULT_V_INF}).  "
            "Use 1.0 for Ahmed body cases.  "
            "When set to 1.0, defaults for --manifest, --output, and --cases-manifest "
            "switch to Ahmed body paths unless overridden explicitly."
        ),
    )
    parser.add_argument("--run-ids", type=int, nargs="+", default=None, help="Specific run IDs")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # When vinf=1 (Ahmed preset) and the user has not overridden paths, switch defaults.
    if args.vinf == 1.0:
        if args.manifest == _DEFAULT_MANIFEST:
            args.manifest = _AHMED_MANIFEST
        if args.output == _DEFAULT_OUTPUT:
            args.output = _AHMED_OUTPUT
        if args.cases_manifest == _DEFAULT_CASES_MANIFEST:
            args.cases_manifest = _AHMED_CASES_MANIFEST

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    generate_all_cases(
        manifest_path=args.manifest,
        template_dir=args.template,
        output_dir=args.output,
        v_inf=args.vinf,
        run_ids=args.run_ids,
        cases_manifest_path=args.cases_manifest,
    )


if __name__ == "__main__":
    main()
