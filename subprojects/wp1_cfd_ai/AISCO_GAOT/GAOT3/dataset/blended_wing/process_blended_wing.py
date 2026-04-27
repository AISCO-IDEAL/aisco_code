"""
Process Blended Wing Body (BWB) CFD data into PyG-compatible .pt files.

Data layout:
  /cluster/work/fuge/AISCO/wp1_cfd_ai/blended_wing/train/
    vtk/case_NNNN.vtk   -- surface mesh with point arrays: cp, cf_x, cf_y, cf_z
    case_data.dat        -- tab-separated: case_name geom_name alt_kft Re M_inf
                           alpha_deg beta_deg CD CL CMy

Coordinate system (already z-up, x=flow):
  x: [0, 1]     -- chord direction (flow, leading edge=0, trailing edge=1)
  y: [-0.73, 0.73] -- span (lateral), chord-normalised
  z: [-0.09, 0.09] -- thickness (vertical), chord-normalised

Saved Data fields:
  pos    (N,3) float32  -- surface vertex coordinates (chord-normalised)
  c      (N,3) float32  -- outward surface normals
  x      (N,1) float32  -- pressure coefficient cp
  cd     (1,)  float32  -- drag coefficient CD from case_data.dat
  force  (1,)  float32  -- NaN (only Cd available, not absolute force)
  filename str          -- case identifier, e.g. "case_0000"
"""

import os
import sys
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyvista as pv
import torch
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_preprocessing import A_SEA_LEVEL, compute_geom_conditioning

DATA_ROOT = "/cluster/work/fuge/AISCO/wp1_cfd_ai/blended_wing/train"
VTK_DIR = os.path.join(DATA_ROOT, "vtk")
CASE_DATA = os.path.join(DATA_ROOT, "case_data_solar.dat")
OUTPUT_DIR = "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_blended_wing_cp_full_filtered"

NUM_WORKERS = 8


def load_case_data(path: str) -> dict:
    """Parse case_data.dat into dict with aerodynamic coefficients and Mach number."""
    case_df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=["case_name", "geom_name", "alt_kft", "Re", "M_inf", "alpha_deg", "beta_deg", "CD", "CL", "CMy"],
    )
    return {
        row["case_name"]: {
            "CD": row["CD"],
            "CL": row["CL"],
            "M_inf": row["M_inf"],
        }
        for _, row in case_df.iterrows()
    }


def _process_single(vtk_file: str, case_lookup: dict, output_dir: str):
    """Process one VTK file. Returns case_name on success, None on failure."""
    case_name = os.path.basename(vtk_file).replace(".vtk", "")  # e.g. case_0000

    if case_name not in case_lookup:
        return None  # no aerodynamic data for this case

    try:
        mesh = pv.read(vtk_file)
        if mesh.n_points == 0:
            return None

        faces = None
        if hasattr(mesh, "faces") and mesh.faces is not None and mesh.faces.size > 0:
            try:
                faces_raw = mesh.faces.reshape(-1, 4)
                faces = torch.tensor(faces_raw[:, 1:], dtype=torch.long).t().contiguous()
            except ValueError:
                mesh = mesh.triangulate()
                faces_raw = mesh.faces.reshape(-1, 4)
                faces = torch.tensor(faces_raw[:, 1:], dtype=torch.long).t().contiguous()

        mesh = mesh.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)

        pos = torch.tensor(mesh.points, dtype=torch.float32)  # (N,3)
        c = torch.tensor(mesh.point_data["Normals"], dtype=torch.float32)  # (N,3)
        cp = torch.tensor(mesh.point_data["cp"], dtype=torch.float32).unsqueeze(1)  # (N,1)

    except Exception as e:
        print(f"Error processing {vtk_file}: {e}")
        return None

    aero = case_lookup[case_name]
    v_inf = float(aero["M_inf"]) * A_SEA_LEVEL

    data = Data(pos=pos, c=c, x=cp, face=faces)

    data.cd = torch.tensor([aero["CD"]], dtype=torch.float32)
    data.force = torch.tensor([float("nan")], dtype=torch.float32)

    geom_extent, geom_aspect, conditioning = compute_geom_conditioning(pos, v_inf)
    data.v_inf = torch.tensor([v_inf], dtype=torch.float32)
    data.geom_extent = geom_extent
    data.geom_aspect = geom_aspect
    data.conditioning = conditioning
    data.filename = case_name

    torch.save(data, os.path.join(output_dir, f"{case_name}.pt"))
    return case_name


def _process_chunk(chunk, case_lookup, output_dir, num_workers):
    """
    Process a list of VTK files with a fresh ProcessPoolExecutor.
    Returns (successes, remaining_on_crash).
    'remaining_on_crash' is non-empty only if the pool died mid-chunk.
    """
    successes = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_single, f, case_lookup, output_dir): f for f in chunk}
        completed = set()
        for future in as_completed(futures):
            completed.add(future)
            try:
                result = future.result()
                if result is not None:
                    successes.append(result)
            except BrokenExecutor:
                break  # pool is dead — stop iterating, remaining futures never ran
            except Exception:
                pass  # legitimate per-file error, skip

    pending = [futures[f] for f in futures if f not in completed]
    return successes, pending


def process_blended_wing(vtk_dir=VTK_DIR, case_data_path=CASE_DATA, output_dir=OUTPUT_DIR, num_workers=NUM_WORKERS):
    os.makedirs(output_dir, exist_ok=True)

    case_lookup = load_case_data(case_data_path)
    print(f"Loaded {len(case_lookup)} cases from case_data.dat")

    vtk_files = [os.path.join(vtk_dir, f"{case_name}.vtk") for case_name in case_lookup]
    missing_vtk = [os.path.basename(vtk_file).replace(".vtk", "") for vtk_file in vtk_files if not os.path.exists(vtk_file)]
    vtk_files = [vtk_file for vtk_file in vtk_files if os.path.exists(vtk_file)]

    print(f"Found {len(vtk_files)} VTK files listed in case_data.dat. Processing with {num_workers} workers...")
    if missing_vtk:
        print(f"  Warning: {len(missing_vtk)} cases from case_data.dat have no matching VTK file and will be skipped.")

    # Skip already-processed files (safe to resume after crash)
    vtk_todo = [f for f in vtk_files if not os.path.exists(os.path.join(output_dir, os.path.basename(f).replace(".vtk", ".pt")))]
    print(f"  Skipping {len(vtk_files) - len(vtk_todo)} already-processed, {len(vtk_todo)} remaining.")

    order_list = []
    skipped_corrupt = 0
    chunk_size = num_workers * 4  # process in chunks; fresh pool per chunk

    pbar = tqdm(total=len(vtk_todo), desc="Processing Blended Wing")
    remaining = list(vtk_todo)

    while remaining:
        chunk, remaining = remaining[:chunk_size], remaining[chunk_size:]
        successes, pending = _process_chunk(chunk, case_lookup, output_dir, num_workers)
        order_list.extend(successes)
        pbar.update(len(chunk) - len(pending))

        if pending:
            # Pool crashed mid-chunk — process each pending file in isolation
            # (fresh single-worker pool per file) to identify and skip the bad one
            for f in pending:
                try:
                    successes2, _ = _process_chunk([f], case_lookup, output_dir, 1)
                    order_list.extend(successes2)
                except Exception:
                    pass
                finally:
                    # Check if the .pt was written (success) vs not (corrupt/crash)
                    case_name = os.path.basename(f).replace(".vtk", "")
                    if not os.path.exists(os.path.join(output_dir, f"{case_name}.pt")):
                        skipped_corrupt += 1
                        print(f"\n  Skipping corrupt file: {os.path.basename(f)}")
                pbar.update(1)

    pbar.close()

    order_path = os.path.join(output_dir, "order_.txt")
    existing_order = []
    if os.path.exists(order_path):
        with open(order_path) as f:
            existing_order = [line.strip() for line in f if line.strip()]

    current_order = [case_name for case_name in case_lookup if os.path.exists(os.path.join(output_dir, f"{case_name}.pt"))]
    all_pt = existing_order[:]
    seen = set(existing_order)
    for case_name in current_order:
        if case_name not in seen:
            all_pt.append(case_name)
            seen.add(case_name)

    with open(order_path, "w") as f:
        f.writelines(f"{name}\n" for name in all_pt)

    print(f"\nThis run:  {len(order_list)} processed, {skipped_corrupt} corrupt/skipped.")
    print(f"Total in output dir: {len(all_pt)} cases.")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    _slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    _default_workers = max(1, _slurm_cpus - 4) if _slurm_cpus > 0 else NUM_WORKERS

    parser = argparse.ArgumentParser(description="Preprocess Blended Wing Body VTK dataset")
    parser.add_argument("--vtk-dir", default=VTK_DIR)
    parser.add_argument("--case-data", default=CASE_DATA)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--num-workers", type=int, default=_default_workers)
    args = parser.parse_args()

    print(f"Using {args.num_workers} workers (SLURM_CPUS_PER_TASK={_slurm_cpus})")
    process_blended_wing(args.vtk_dir, args.case_data, args.output_dir, args.num_workers)
