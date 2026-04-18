#!/usr/bin/env python3
"""Convert PLY geometry files to STL for OpenFOAM snappyHexMesh consumption.

Handles unit detection (mm vs m), mesh repair for watertightness,
and batch processing with a JSON manifest for downstream consumption
by generate_cases.py.

Usage:
    python ply_to_stl.py                              # all PLYs
    python ply_to_stl.py --input-dir /path/to/plys    # custom input
    python ply_to_stl.py --run-ids 1 2 3              # specific runs
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import cast

import trimesh

logger = logging.getLogger(__name__)

# Threshold: if max bounding box extent > 20, geometry is in mm
_MM_THRESHOLD = 20.0
_GROUND_CLEARANCE_M = 0.1

# Default paths on Euler
_DEFAULT_INPUT_DIR = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/FLUENT/GEOMETRY")
_DEFAULT_OUTPUT_DIR = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/GEOMETRY")
_DEFAULT_MANIFEST = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/geometry_manifest.json")


def compute_ground_translation(z_min: float, target_clearance: float = _GROUND_CLEARANCE_M) -> float:
    """Compute the Z translation to place the mesh at exactly the target clearance.

    Args:
        z_min: Current minimum Z coordinate of the mesh, in meters.
        target_clearance: Exact gap between the lowest vertex and the OpenFOAM
            ground plane, in meters.

    Returns:
        Translation to apply in meters. Positive lifts the mesh, negative
        lowers it.
    """
    return target_clearance - z_min


def detect_units(mesh: trimesh.Trimesh) -> str:
    """Detect whether a mesh is in millimeters or meters.

    Args:
        mesh: A trimesh object to check.

    Returns:
        "mm" if max bounding box extent > 20, "m" otherwise.
    """
    max_extent = float(mesh.bounding_box.extents.max())
    if max_extent > _MM_THRESHOLD:
        return "mm"
    return "m"


def repair_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, bool]:
    """Attempt to repair a mesh to make it watertight.

    Applies fill_holes and fix_winding. Returns the mesh and whether
    it is watertight after repair.

    Args:
        mesh: A trimesh object to repair.

    Returns:
        Tuple of (repaired mesh, is_watertight).
    """
    if mesh.is_watertight:
        return mesh, True

    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)

    return mesh, mesh.is_watertight


def convert_single_ply(
    ply_path: Path,
    output_dir: Path,
) -> dict:
    """Convert a single PLY file to a meter-scale STL.

    Args:
        ply_path: Path to the input PLY file.
        output_dir: Directory to write the output STL.

    Returns:
        A manifest entry dict with run_id, status, paths, bounds, etc.
    """
    run_id = ply_path.stem  # e.g. "RUN_001"
    stl_filename = f"{run_id}.stl"
    stl_path = output_dir / stl_filename

    entry: dict = {
        "run_id": run_id,
        "ply_path": str(ply_path),
        "stl_path": str(stl_path),
        "status": "pending",
        "detected_unit": None,
        "watertight_before_repair": None,
        "watertight_after_repair": None,
    }

    try:
        mesh = cast(trimesh.Trimesh, trimesh.load(str(ply_path), force="mesh"))

        # Detect units
        unit = detect_units(mesh)
        entry["detected_unit"] = unit

        # Convert mm → m if necessary
        if unit == "mm":
            mesh.vertices = mesh.vertices / 1000.0
            logger.info("%s: detected mm, scaled to meters", run_id)
        else:
            logger.info("%s: detected meters, no scaling", run_id)

        # Place the lowest vertex at exactly _GROUND_CLEARANCE_M above z=0
        # so the underside remains in the solved car_surface patch.
        z_min = float(mesh.bounds[0][2])
        z_shift = compute_ground_translation(z_min)
        entry["source_z_min_m"] = z_min
        entry["ground_clearance_m"] = _GROUND_CLEARANCE_M
        mesh.vertices[:, 2] += z_shift
        entry["z_shift_applied"] = z_shift
        logger.info(
            "%s: z_min=%.4f — translated Z by %+.4f m to exact %.3f m ground clearance",
            run_id,
            z_min,
            z_shift,
            _GROUND_CLEARANCE_M,
        )

        # Check/repair watertightness
        entry["watertight_before_repair"] = mesh.is_watertight
        mesh, is_watertight = repair_mesh(mesh)
        entry["watertight_after_repair"] = is_watertight

        if not is_watertight:
            # snappyHexMesh for external aero (locationInMesh outside car) can
            # handle non-watertight STL. Warn but continue rather than exclude.
            logger.warning(
                "%s: non-watertight after repair — converting anyway (snappyHexMesh may still succeed for external flow)",
                run_id,
            )

        # Record bounds in meters
        entry["bounds_min"] = mesh.bounds[0].tolist()
        entry["bounds_max"] = mesh.bounds[1].tolist()

        # Export as binary STL with solid name "car_surface"
        # trimesh doesn't support solid naming in binary STL directly,
        # so we set the metadata
        mesh.metadata["name"] = "car_surface"
        stl_data = trimesh.exchange.stl.export_stl(mesh)
        with open(stl_path, "wb") as f:
            f.write(stl_data)

        entry["status"] = "converted"
        logger.info("%s: converted successfully (bounds_m: %s → %s)", run_id, entry["bounds_min"], entry["bounds_max"])

    except Exception:
        entry["status"] = "excluded"
        entry["exclusion_reason"] = f"Error during conversion: {sys.exc_info()[1]}"
        logger.exception("%s: EXCLUDED — error during conversion", run_id)

    return entry


def batch_convert(
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    run_ids: list[int] | None = None,
) -> list[dict]:
    """Convert all PLY files in a directory to STL.

    Args:
        input_dir: Directory containing RUN_XXX.ply files.
        output_dir: Directory to write RUN_XXX.stl files.
        manifest_path: Path to write the JSON manifest.
        run_ids: Optional list of specific run IDs to process.
            If None, processes all RUN_*.ply files found.

    Returns:
        List of manifest entry dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_ids is not None:
        ply_files = []
        for rid in sorted(run_ids):
            p = input_dir / f"RUN_{rid:03d}.ply"
            if p.exists():
                ply_files.append(p)
            else:
                logger.warning("RUN_%03d.ply not found in %s, skipping", rid, input_dir)
    else:
        ply_files = sorted(input_dir.glob("RUN_*.ply"))

    if not ply_files:
        logger.warning("No PLY files found in %s", input_dir)
        return []

    logger.info("Processing %d PLY files from %s", len(ply_files), input_dir)

    results: list[dict] = []
    for ply_path in ply_files:
        result = convert_single_ply(ply_path, output_dir)
        results.append(result)

    # Write manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    converted = sum(1 for r in results if r["status"] == "converted")
    excluded = sum(1 for r in results if r["status"] == "excluded")
    logger.info("Batch complete: %d converted, %d excluded out of %d total", converted, excluded, len(results))

    return results


def main() -> None:
    """CLI entry point for PLY→STL batch conversion."""
    parser = argparse.ArgumentParser(description="Convert PLY geometries to STL for OpenFOAM")
    parser.add_argument("--input-dir", type=Path, default=_DEFAULT_INPUT_DIR, help="Directory with RUN_XXX.ply files")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR, help="Directory for output STL files")
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST, help="Path for output manifest JSON")
    parser.add_argument("--run-ids", type=int, nargs="+", default=None, help="Specific run IDs to process")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    batch_convert(args.input_dir, args.output_dir, args.manifest, args.run_ids)


if __name__ == "__main__":
    main()
