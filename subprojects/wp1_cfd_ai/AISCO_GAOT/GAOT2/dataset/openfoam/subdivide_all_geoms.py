"""Subdivide all OpenFOAM geometry PLYs using adaptive Loop subdivision.

For each PLY, computes the number of Loop iterations needed to reach
approximately --target-verts (default 180,000) without exceeding it.
Each iteration multiplies vertex count by ~4x.

Usage:
    python subdivide_all_geoms.py \
        --input-dir /cluster/work/fuge/AISCO/wp1_cfd_ai/openfoam_geometry \
        --output-dir /cluster/work/fuge/AISCO/wp1_cfd_ai/of_geom_l2 \
        --target-verts 180000 \
        --workers 16
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import open3d as o3d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_iterations(num_verts: int, target: int) -> int:
    """Compute Loop subdivision iterations to get closest to target.

    Each Loop iteration ~4x vertex count. Picks the n where |V*4^n - target|
    is minimized (allowed to exceed target).

    Args:
        num_verts: Current vertex count.
        target: Target vertex count (~180K).

    Returns:
        Number of iterations (0 if already closest without subdividing).
    """
    if num_verts >= target:
        return 0
    # Find n that minimizes |V*4^n - target|
    n = round(math.log(target / num_verts) / math.log(4))
    return max(n, 0)


def subdivide_ply(
    input_path: Path,
    output_path: Path,
    target_verts: int,
) -> dict[str, int | str]:
    """Apply adaptive Loop subdivision to a single PLY file.

    Args:
        input_path: Path to input PLY.
        output_path: Path to write subdivided PLY.
        target_verts: Target vertex count (will not exceed).

    Returns:
        Dict with run_id, input/output vertex/face counts, iterations, and status.
    """
    run_id = input_path.stem
    try:
        mesh = o3d.io.read_triangle_mesh(str(input_path))
        if mesh.is_empty():
            return {"run_id": run_id, "status": "error", "message": "empty mesh"}

        verts_in = len(np.asarray(mesh.vertices))
        faces_in = len(np.asarray(mesh.triangles))

        iterations = compute_iterations(verts_in, target_verts)

        if iterations > 0:
            mesh = mesh.subdivide_loop(number_of_iterations=iterations)

        verts_out = len(np.asarray(mesh.vertices))
        faces_out = len(np.asarray(mesh.triangles))

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write PLY (binary for compactness)
        o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=False)

        return {
            "run_id": run_id,
            "status": "ok",
            "iterations": iterations,
            "verts_in": verts_in,
            "faces_in": faces_in,
            "verts_out": verts_out,
            "faces_out": faces_out,
        }
    except Exception as e:
        return {"run_id": run_id, "status": "error", "message": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Loop-subdivide all PLY geometries.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/openfoam_geometry"),
        help="Directory containing input PLY files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/of_geom_l2"),
        help="Directory to write subdivided PLY files.",
    )
    parser.add_argument(
        "--target-verts",
        type=int,
        default=180_000,
        help="Target max vertex count per mesh (default: 180000).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers.",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        logger.error(
            "Input directory does not exist: %s. "
            "Provide a valid --input-dir path.",
            args.input_dir,
        )
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(args.input_dir.glob("RUN_*.ply"))
    if not ply_files:
        logger.error("No RUN_*.ply files found in %s.", args.input_dir)
        sys.exit(1)

    logger.info(
        "Subdividing %d PLY files to target ~%d verts using %d workers.",
        len(ply_files),
        args.target_verts,
        args.workers,
    )

    results: list[dict[str, int | str]] = []
    errors: list[dict[str, int | str]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for ply_path in ply_files:
            out_path = args.output_dir / ply_path.name
            fut = executor.submit(subdivide_ply, ply_path, out_path, args.target_verts)
            futures[fut] = ply_path.name

        for fut in as_completed(futures):
            result = fut.result()
            if result["status"] == "ok":
                results.append(result)
                logger.info(
                    "%s: %d→%d verts (%d iters), %d→%d faces",
                    result["run_id"],
                    result["verts_in"],
                    result["verts_out"],
                    result["iterations"],
                    result["faces_in"],
                    result["faces_out"],
                )
            else:
                errors.append(result)
                logger.error(
                    "%s: FAILED — %s",
                    result["run_id"],
                    result.get("message", "unknown"),
                )

    logger.info("=== SUMMARY ===")
    logger.info("Success: %d / %d", len(results), len(ply_files))
    if errors:
        logger.warning("Failures: %d", len(errors))
        for e in errors:
            logger.warning("  %s: %s", e["run_id"], e.get("message", "unknown"))
    else:
        logger.info("All files subdivided successfully.")


if __name__ == "__main__":
    main()
