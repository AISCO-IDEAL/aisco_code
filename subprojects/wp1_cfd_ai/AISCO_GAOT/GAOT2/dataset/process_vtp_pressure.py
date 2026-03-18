#!/usr/bin/env python3
"""
Process VTP boundary mesh files from drivaer_complete / ahmed_complete
into GAOT-compatible .pt files with pressure fields.

For each run, extracts from the .vtp file:
  - pos:          [N, 3]   vertex coordinates
  - x:            [N, 1]   pressure at each vertex (cell_data → point_data)
  - c:            [N, 3]   outward unit normals (computed)
  - face:         [3, F]   triangulated face connectivity
  - wall_shear:   [N, 3]   wall shear stress at each vertex
  - cp:           [N, 1]   pressure coefficient at each vertex (if available)
  - force:        [1]      drag force from CSV (drivaer only; NaN for ahmed)
  - cd:           [1]      drag coefficient from CSV
  - filename:     str

Supports two dataset layouts:
  - drivaer_complete: boundary_{id}.vtp, drag_param_{id}.csv
  - ahmed_complete:   boundary_{id}.vtp, force_mom_{id}.csv

Usage:
    python process_vtp_pressure.py drivaer                   # all 500 drivaer
    python process_vtp_pressure.py ahmed                     # all 500 ahmed
    python process_vtp_pressure.py drivaer --max_runs 5      # first 5
    python process_vtp_pressure.py ahmed --run_ids 1 2 3     # specific runs
    python process_vtp_pressure.py drivaer --workers 4       # parallel workers
"""

import argparse
import csv
import glob
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyvista as pv
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# ---------------------------------------------------------------------------
#  Dataset configurations
# ---------------------------------------------------------------------------
DATASETS = {
    "drivaer": {
        "input_dir": "/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_complete",
        "output_dir": "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_drivaer_pressure",
        "vtp_pattern": "boundary_{id}.vtp",
        "pressure_key": "pMeanTrim",  # cell data key for pressure
        "cp_key": "CpMeanTrim",  # cell data key for Cp (pressure coef)
        "shear_key": "wallShearStressMeanTrim",  # cell data key for wall shear
        "csv_type": "drag_param",  # CSV format type
        "prefix": "drivaer_run",  # output folder prefix
    },
    "ahmed": {
        "input_dir": "/cluster/work/fuge/AISCO/wp1_cfd_ai/ahmed_complete",
        "output_dir": "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_ahmed_pressure",
        "vtp_pattern": "boundary_{id}.vtp",
        "pressure_key": "pMean",
        "cp_key": "static(p)_coeffMean",
        "shear_key": "wallShearStressMean",
        "csv_type": "force_mom",
        "prefix": "ahmed_run",
    },
}

NUM_WORKERS = 2  # VTP reading is memory-heavy (8M+ pts); keep moderate


# ---------------------------------------------------------------------------
#  CSV parsers
# ---------------------------------------------------------------------------
def parse_drivaer_csv(run_dir, run_id):
    """Parse drag_param_{id}.csv → (cd, force)."""
    csv_path = os.path.join(run_dir, f"drag_param_{run_id}.csv")
    if not os.path.exists(csv_path):
        return float("nan"), float("nan")
    with open(csv_path) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            cd = float(row.get("Cd", "nan"))
            force = float(row.get("Drag_Force_N", "nan"))
            return cd, force
    return float("nan"), float("nan")


def parse_ahmed_csv(run_dir, run_id):
    """Parse force_mom_{id}.csv → (cd, force=NaN since not available)."""
    csv_path = os.path.join(run_dir, f"force_mom_{run_id}.csv")
    if not os.path.exists(csv_path):
        return float("nan"), float("nan")
    with open(csv_path) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            cd = float(row.get("cd", "nan"))
            # Ahmed doesn't have drag force in Newtons, only Cd
            return cd, float("nan")
    return float("nan"), float("nan")


CSV_PARSERS = {
    "drag_param": parse_drivaer_csv,
    "force_mom": parse_ahmed_csv,
}


# ---------------------------------------------------------------------------
#  Core processing
# ---------------------------------------------------------------------------
def process_single_vtp(run_id, dataset_cfg):
    """
    Process one VTP run into a GAOT .pt file with pressure field data.
    Returns the save name on success, None on failure.
    """
    input_dir = dataset_cfg["input_dir"]
    output_dir = dataset_cfg["output_dir"]
    prefix = dataset_cfg["prefix"]

    run_dir = os.path.join(input_dir, f"run_{run_id}")
    vtp_path = os.path.join(run_dir, dataset_cfg["vtp_pattern"].format(id=run_id))

    if not os.path.exists(vtp_path):
        print(f"  [SKIP] run_{run_id}: {vtp_path} not found")
        return None

    # ---- 1. Read VTP mesh ----
    mesh = pv.read(vtp_path)
    if mesh.n_points == 0:
        print(f"  [SKIP] run_{run_id}: empty mesh")
        return None

    # ---- 2. Triangulate (VTP may have quads/polygons) ----
    mesh_tri = mesh.triangulate()

    # ---- 3. Convert cell data → point data ----
    mesh_pt = mesh_tri.cell_data_to_point_data()

    # ---- 4. Compute normals ----
    mesh_pt = mesh_pt.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)

    # ---- 5. Extract arrays ----
    pos = torch.tensor(mesh_pt.points, dtype=torch.float32)  # [N, 3]
    normals = torch.tensor(mesh_pt.point_data["Normals"], dtype=torch.float32)  # [N, 3]

    # Pressure
    pressure_key = dataset_cfg["pressure_key"]
    if pressure_key in mesh_pt.point_data:
        pressure = torch.tensor(mesh_pt.point_data[pressure_key], dtype=torch.float32).unsqueeze(-1)  # [N, 1]
    else:
        print(f"  [WARN] run_{run_id}: '{pressure_key}' not found, using zeros")
        pressure = torch.zeros(pos.shape[0], 1, dtype=torch.float32)

    # Cp (pressure coefficient)
    cp_key = dataset_cfg.get("cp_key")
    cp = None
    if cp_key and cp_key in mesh_pt.point_data:
        cp = torch.tensor(mesh_pt.point_data[cp_key], dtype=torch.float32).unsqueeze(-1)  # [N, 1]

    # Wall shear stress
    shear_key = dataset_cfg["shear_key"]
    if shear_key in mesh_pt.point_data:
        wall_shear = torch.tensor(mesh_pt.point_data[shear_key], dtype=torch.float32)  # [N, 3]
    else:
        wall_shear = torch.zeros(pos.shape[0], 3, dtype=torch.float32)

    # ---- 6. Extract face connectivity ----
    faces_np = mesh_tri.faces.reshape(-1, 4)[:, 1:]  # strip the "3" prefix
    face = torch.tensor(faces_np.T, dtype=torch.long)  # [3, F]

    # ---- 7. Parse scalar targets from CSV ----
    csv_parser = CSV_PARSERS[dataset_cfg["csv_type"]]
    cd_val, force_val = csv_parser(run_dir, run_id)

    # ---- 8. Build PyG Data and save ----
    data = Data(
        pos=pos,  # [N, 3]
        x=pressure,  # [N, 1] — pressure field (GAOT target)
        c=normals,  # [N, 3]
        face=face,  # [3, F]
        wall_shear=wall_shear,  # [N, 3]
        force=torch.tensor([force_val], dtype=torch.float32),  # [1]
        cd=torch.tensor([cd_val], dtype=torch.float32),  # [1]
    )
    if cp is not None:
        data.cp = cp  # [N, 1]
    data.filename = f"run_{run_id}"

    save_name = f"{prefix}_{run_id}"
    save_dir = os.path.join(output_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(data, os.path.join(save_dir, "data.pt"))
    return save_name


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _detect_run_ids(input_dir: str) -> list:
    """Auto-detect run IDs by scanning run_* subdirectories."""
    run_dirs = sorted(glob.glob(os.path.join(input_dir, "run_*")))
    run_ids = []
    for rd in run_dirs:
        m = re.search(r"run_(\d+)$", rd)
        if m:
            run_ids.append(int(m.group(1)))
    return sorted(run_ids)


# ---------------------------------------------------------------------------
#  Batch processing
# ---------------------------------------------------------------------------
def process_all(dataset_name, run_ids=None, max_runs=None, num_workers=NUM_WORKERS):
    """Process all (or selected) runs for a given dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASETS.keys())}")

    cfg = DATASETS[dataset_name]
    input_dir = cfg["input_dir"]
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect runs
    if run_ids is None:
        run_ids = _detect_run_ids(input_dir)
        run_ids.sort()

    if max_runs is not None:
        run_ids = run_ids[:max_runs]

    print(f"Processing {len(run_ids)} {dataset_name} runs with {num_workers} workers...")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")

    order_list = []

    if num_workers <= 1:
        for rid in tqdm(run_ids, desc=f"Processing {dataset_name}"):
            try:
                result = process_single_vtp(rid, cfg)
                if result is not None:
                    order_list.append(result)
            except Exception as e:
                print(f"  [FAIL] run_{rid}: {e}")
                traceback.print_exc()
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_vtp, rid, cfg): rid for rid in run_ids}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {dataset_name}"):
                rid = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        order_list.append(result)
                except Exception as e:
                    print(f"  [FAIL] run_{rid}: {e}")
                    traceback.print_exc()

    # Sort numerically
    order_list.sort(key=lambda x: int(x.rsplit("_", 1)[-1]))

    order_path = os.path.join(output_dir, "order_.txt")
    with open(order_path, "w") as f:
        f.writelines(f"{name}\n" for name in order_list)

    print(f"\nDone! Processed {len(order_list)}/{len(run_ids)} runs.")
    print(f"Order file: {order_path}")
    return order_list


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Process VTP boundary meshes into GAOT .pt with pressure field.")
    parser.add_argument("dataset", choices=list(DATASETS.keys()), help="Which dataset to process (drivaer or ahmed)")
    parser.add_argument("--run_ids", nargs="+", type=int, default=None)
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    process_all(
        dataset_name=args.dataset,
        run_ids=args.run_ids,
        max_runs=args.max_runs,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
