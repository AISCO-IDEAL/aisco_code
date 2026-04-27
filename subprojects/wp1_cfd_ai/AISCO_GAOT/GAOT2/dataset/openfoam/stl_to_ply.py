#!/usr/bin/env python3
"""Convert STL geometry files to PLY for pipeline append-mode consumption.

Handles unit detection (mm vs m), mesh repair for watertightness,
and batch processing with a JSON manifest. Output PLY files can be fed
directly into run_pipeline.py --mode append.

Usage:
    python stl_to_ply.py                              # all STLs in default input dir
    python stl_to_ply.py --input-dir /path/to/stls    # custom input
    python stl_to_ply.py --run-ids 1 2 3              # specific RUN_XXX.stl files
"""

from __future__ import annotations

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
_DEFAULT_INPUT_DIR = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/stl_input")
_DEFAULT_OUTPUT_DIR = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/ply_staging")
_DEFAULT_MANIFEST = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/stl_to_ply_manifest.json")


def compute_ground_translation(z_min: float, target_clearance: float = _GROUND_CLEARANCE_M) -> float:
    """Compute the Z translation to place the mesh at exactly the target clearance.

    Args:
        z_min: Current minimum Z coordinate of the mesh, in meters.
        target_clearance: Exact gap between the lowest vertex and the OpenFOAM
            ground plane, in meters.

    Returns:
        Translation to apply in meters. Positive lifts the mesh, negative lowers it.
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


def convert_single_stl(
    stl_path: Path,
    output_dir: Path,
) -> dict:
    """Convert a single STL file to a meter-scale PLY.

    Args:
        stl_path: Path to the input STL file.
        output_dir: Directory to write the output PLY.

    Returns:
        A manifest entry dict with stem, status, paths, bounds, etc.
    """
    stem = stl_path.stem  # e.g. "RUN_001" or "my_car"
    ply_path = output_dir / f"{stem}.ply"

    entry: dict = {
        "stem": stem,
        "stl_path": str(stl_path),
        "ply_path": str(ply_path),
        "status": "pending",
        "detected_unit": None,
        "watertight_before_repair": None,
        "watertight_after_repair": None,
    }

    try:
        mesh = cast(trimesh.Trimesh, trimesh.load(str(stl_path), force="mesh"))

        # Detect units
        unit = detect_units(mesh)
        entry["detected_unit"] = unit

        # Convert mm → m if necessary
        if unit == "mm":
            mesh.vertices = mesh.vertices / 1000.0
            logger.info("%s: detected mm, scaled to meters", stem)
        else:
            logger.info("%s: detected meters, no scaling", stem)

        # Place the lowest vertex at exactly _GROUND_CLEARANCE_M above z=0
        z_min = float(mesh.bounds[0][2])
        z_shift = compute_ground_translation(z_min)
        entry["source_z_min_m"] = z_min
        entry["ground_clearance_m"] = _GROUND_CLEARANCE_M
        mesh.vertices[:, 2] += z_shift
        entry["z_shift_applied"] = z_shift
        logger.info(
            "%s: z_min=%.4f — translated Z by %+.4f m to exact %.3f m ground clearance",
            stem,
            z_min,
            z_shift,
            _GROUND_CLEARANCE_M,
        )

        # Check/repair watertightness
        entry["watertight_before_repair"] = mesh.is_watertight
        mesh, is_watertight = repair_mesh(mesh)
        entry["watertight_after_repair"] = is_watertight

        if not is_watertight:
            logger.warning(
                "%s: non-watertight after repair — converting anyway (snappyHexMesh may still succeed for external flow)",
                stem,
            )

        # Record bounds in meters
        entry["bounds_min"] = mesh.bounds[0].tolist()
        entry["bounds_max"] = mesh.bounds[1].tolist()

        # Export as binary PLY
        ply_data = trimesh.exchange.ply.export_ply(mesh)
        with open(ply_path, "wb") as f:
            f.write(ply_data)

        entry["status"] = "converted"
        logger.info("%s: converted successfully (bounds_m: %s → %s)", stem, entry["bounds_min"], entry["bounds_max"])

    except Exception:
        entry["status"] = "excluded"
        entry["exclusion_reason"] = f"Error during conversion: {sys.exc_info()[1]}"
        logger.exception("%s: EXCLUDED — error during conversion", stem)

    return entry


def batch_convert(
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    run_ids: list[int] | None = None,
) -> list[dict]:
    """Convert all STL files in a directory to PLY.

    Args:
        input_dir: Directory containing STL files.
        output_dir: Directory to write PLY files.
        manifest_path: Path to write the JSON manifest.
        run_ids: Optional list of specific run IDs to process (expects RUN_XXX.stl naming).
            If None, processes all *.stl files found.

    Returns:
        List of manifest entry dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_ids is not None:
        stl_files = []
        for rid in sorted(run_ids):
            p = input_dir / f"RUN_{rid:03d}.stl"
            if p.exists():
                stl_files.append(p)
            else:
                logger.warning("RUN_%03d.stl not found in %s, skipping", rid, input_dir)
    else:
        stl_files = sorted(input_dir.glob("*.stl"))

    if not stl_files:
        logger.warning("No STL files found in %s", input_dir)
        return []

    logger.info("Processing %d STL files from %s", len(stl_files), input_dir)

    results: list[dict] = []
    for stl_path in stl_files:
        result = convert_single_stl(stl_path, output_dir)
        results.append(result)

    # Append-safe manifest: merge new results with any existing entries
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            existing = json.load(f)
    merged = {e["stem"]: e for e in existing}
    for r in results:
        merged[r["stem"]] = r
    all_entries = list(merged.values())
    with open(manifest_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    converted = sum(1 for r in results if r["status"] == "converted")
    excluded = sum(1 for r in results if r["status"] == "excluded")
    logger.info("Batch complete: %d converted, %d excluded out of %d total", converted, excluded, len(results))

    return results


def main() -> None:
    """CLI entry point for STL→PLY batch conversion."""
    parser = argparse.ArgumentParser(description="Convert STL geometries to PLY for pipeline append-mode")
    parser.add_argument("--input-dir", type=Path, default=_DEFAULT_INPUT_DIR, help="Directory with STL files")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR, help="Directory for output PLY files")
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST, help="Path for output manifest JSON")
    parser.add_argument(
        "--run-ids", type=int, nargs="+", default=None, help="Specific run IDs to process (expects RUN_XXX.stl naming)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    batch_convert(args.input_dir, args.output_dir, args.manifest, args.run_ids)


if __name__ == "__main__":
    main()
