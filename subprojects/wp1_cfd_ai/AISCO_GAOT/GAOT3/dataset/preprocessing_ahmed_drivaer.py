#!/usr/bin/env python3
"""
Process VTP boundary mesh files from drivaer_complete / ahmed_complete
into GAOT-compatible .pt files with canonical Cp fields and sample-level conditioning.

For each run, extracts from the .vtp file:
  - pos:          [N, 3]   vertex coordinates
    - x:            [N, 1]   pressure coefficient at each vertex
  - c:            [N, 3]   outward unit normals (computed)
  - face:         [3, F]   triangulated face connectivity
  - wall_shear:   [N, 3]   wall shear stress at each vertex
    - pressure:     [N, 1]   raw pressure at each vertex (if available)
  - force:        [1]      drag force from CSV (drivaer only; NaN for ahmed)
  - cd:           [1]      drag coefficient from CSV
    - v_inf:        [1]      freestream speed for the sample
    - geom_extent:  [3]      geometry extents before normalization
    - geom_aspect:  [3]      extent ratios before normalization
    - conditioning: [1, 7]   [v_inf, extent_xyz, aspect_xyz]
    - filename:     str

Supports two dataset layouts:
  - drivaer_complete: boundary_{id}.vtp, drag_param_{id}.csv
  - ahmed_complete:   boundary_{id}.vtp, force_mom_{id}.csv

Usage:
    python preprocessing_ahmed_drivaer.py drivaer                   # all 500 drivaer
    python preprocessing_ahmed_drivaer.py ahmed                     # all 500 ahmed
    python preprocessing_ahmed_drivaer.py drivaer --max_runs 5      # first 5
    python preprocessing_ahmed_drivaer.py ahmed --run_ids 1 2 3     # specific runs
    python preprocessing_ahmed_drivaer.py drivaer --workers 4       # parallel workers
    python preprocessing_ahmed_drivaer.py drivaer --target-reduction 0.6  # keep 40% of vertices
"""

import argparse
import csv
import glob
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyvista as pv
import scipy.spatial
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.dataset.common_preprocessing import (
    AIR_DENSITY,
    PREPROCESS_CONTRACT_VERSION,
    V_INF_AHMED,
    V_INF_DRIVAER,
    apply_canonical_contract,
    compute_geom_conditioning,
    compute_pressure_coefficient,
    validate_canonical_contract,
    write_dataset_manifest,
)

# ---------------------------------------------------------------------------
#  Dataset configurations
# ---------------------------------------------------------------------------
DATASETS = {
    "drivaer": {
        "input_dir": "/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_complete",
        "output_dir": "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_drivaer_cp_downsampled",
        "vtp_pattern": "boundary_{id}.vtp",
        "pressure_key": "pMeanTrim",
        "cp_key": "CpMeanTrim",
        "shear_key": "wallShearStressMeanTrim",
        "csv_type": "drag_param",
        "prefix": "drivaer_run",
        "v_inf": V_INF_DRIVAER,
        "target_reduction": 0.0,
    },
    "ahmed": {
        "input_dir": "/cluster/work/fuge/AISCO/wp1_cfd_ai/ahmed_complete",
        "output_dir": "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_ahmed_cp_full",
        "vtp_pattern": "boundary_{id}.vtp",
        "pressure_key": "pMean",
        "cp_key": "static(p)_coeffMean",
        "shear_key": "wallShearStressMean",
        "csv_type": "force_mom",
        "prefix": "ahmed_run",
        "v_inf": V_INF_AHMED,
        "target_reduction": 0.0,
    },
}

NUM_WORKERS = 2


# ---------------------------------------------------------------------------
#  CSV parsers
# ---------------------------------------------------------------------------
def parse_drivaer_csv(run_dir, run_id):
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
    csv_path = os.path.join(run_dir, f"force_mom_{run_id}.csv")
    if not os.path.exists(csv_path):
        return float("nan"), float("nan")
    with open(csv_path) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            cd = float(row.get("cd", "nan"))
            return cd, float("nan")
    return float("nan"), float("nan")


CSV_PARSERS = {
    "drag_param": parse_drivaer_csv,
    "force_mom": parse_ahmed_csv,
}


# ---------------------------------------------------------------------------
#  Core processing
# ---------------------------------------------------------------------------
def process_single_vtp(run_id, dataset_cfg, target_reduction=0.0):  # noqa: PLR0912, PLR0915 - legacy extraction path kept stable.
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

    # ---- 2. Triangulate ----
    mesh_tri = mesh.triangulate()

    # ---- 3. Convert cell data → point data (High-Res) ----
    mesh_pt_highres = mesh_tri.cell_data_to_point_data()
    mesh_pt = mesh_pt_highres  # Fallback

    # ---- 3b. Optional mesh decimation with nearest-neighbor field transfer ----
    if target_reduction > 0:
        # 1) Simplify geometry only.
        mesh_pt_lowres = mesh_pt_highres.decimate(target_reduction=target_reduction, volume_preservation=True)

        # 2) Build KDTree for robust transfer of point fields.
        tree = scipy.spatial.KDTree(mesh_pt_highres.points)
        _, nearest_indices = tree.query(mesh_pt_lowres.points)

        # 3) Copy all arrays from nearest high-resolution neighbors.
        for key in mesh_pt_highres.point_data:
            mesh_pt_lowres.point_data[key] = mesh_pt_highres.point_data[key][nearest_indices]

        mesh_pt = mesh_pt_lowres

    # ---- 4. Compute normals ----
    mesh_pt = mesh_pt.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)

    # ---- 5. Extract arrays ----
    pos = torch.tensor(mesh_pt.points, dtype=torch.float32)
    normals = torch.tensor(mesh_pt.point_data["Normals"], dtype=torch.float32)

    # Pressure
    pressure_key = dataset_cfg["pressure_key"]
    if pressure_key in mesh_pt.point_data:
        pressure = torch.tensor(mesh_pt.point_data[pressure_key], dtype=torch.float32).unsqueeze(-1)
    else:
        print(f"  [WARN] run_{run_id}: '{pressure_key}' not found")
        pressure = None

    # Cp
    cp_key = dataset_cfg.get("cp_key")
    cp = None
    if pressure is not None:
        cp = compute_pressure_coefficient(pressure, dataset_cfg["v_inf"], AIR_DENSITY)
    elif cp_key and cp_key in mesh_pt.point_data:
        cp = torch.tensor(mesh_pt.point_data[cp_key], dtype=torch.float32).unsqueeze(-1)
    else:
        cp = torch.full((pos.shape[0], 1), float("nan"), dtype=torch.float32)

    # Wall shear stress
    shear_key = dataset_cfg["shear_key"]
    wall_shear = torch.tensor(mesh_pt.point_data[shear_key], dtype=torch.float32) if shear_key in mesh_pt.point_data else None

    # ---- 6. Extract face connectivity ----
    faces_np = mesh_pt.faces.reshape(-1, 4)[:, 1:]
    face = torch.tensor(faces_np.T, dtype=torch.long)

    # ---- 7. Parse scalar targets from CSV ----
    csv_parser = CSV_PARSERS[dataset_cfg["csv_type"]]
    cd_val, force_val = csv_parser(run_dir, run_id)

    geom_extent, geom_aspect, conditioning = compute_geom_conditioning(pos, dataset_cfg["v_inf"])

    # ---- 8. Build PyG Data and save ----
    data = Data(
        pos=pos,
        x=cp,
        c=normals,
        face=face,
        force=torch.tensor([force_val], dtype=torch.float32),
        cd=torch.tensor([cd_val], dtype=torch.float32),
    )
    if pressure is not None:
        data.pressure = pressure
    if wall_shear is not None:
        data.wall_shear = wall_shear
    data.cp = cp
    data.v_inf = torch.tensor([dataset_cfg["v_inf"]], dtype=torch.float32)
    data.geom_extent = geom_extent
    data.geom_aspect = geom_aspect
    data.conditioning = conditioning
    data.filename = f"run_{run_id}"

    flow_conditions = {"v_inf": float(dataset_cfg["v_inf"])}
    apply_canonical_contract(
        data,
        source_dataset=dataset_cfg["dataset_name"],
        source_case_id=f"run_{run_id}",
        filename=f"run_{run_id}",
        geom_name="",
        flow_conditions=flow_conditions,
    )
    validate_canonical_contract(data)

    save_name = f"{prefix}_{run_id}"
    save_dir = os.path.join(output_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)

    final_path = os.path.join(save_dir, "data.pt")

    # Save and re-load once to guard against partially written files.
    try:
        torch.save(data, final_path)
        with open(final_path, "a") as f:
            os.fsync(f.fileno())

        # Sanity check to catch corrupt archives early.
        _ = torch.load(final_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [ERROR] Save/Verify failed for run_{run_id}: {e}")
        if os.path.exists(final_path):
            os.remove(final_path)
        return None

    return save_name


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _detect_run_ids(input_dir: str) -> list:
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
def process_all(  # noqa: PLR0912, PLR0915 - existing branch/statement structure retained for clarity and stable behavior.
    dataset_name,
    run_ids=None,
    max_runs=None,
    num_workers=NUM_WORKERS,
    output_dir=None,
    target_reduction=0.0,
    input_dir=None,
):
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASETS.keys())}")

    cfg = dict(DATASETS[dataset_name])
    cfg["dataset_name"] = dataset_name
    if input_dir is not None:
        cfg["input_dir"] = input_dir
    input_dir = str(cfg["input_dir"])
    output_dir = str(output_dir or cfg["output_dir"])

    if target_reduction > 0:
        keep_pct = int((1.0 - target_reduction) * 100)
        output_dir = output_dir.replace("_full", f"_decimated_{keep_pct}pct")

    cfg["output_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    if run_ids is None:
        run_ids = _detect_run_ids(input_dir)
        run_ids.sort()

    if max_runs is not None:
        run_ids = run_ids[:max_runs]

    existing_names = []
    run_ids_todo = []
    for rid in run_ids:
        save_name = f"{cfg['prefix']}_{rid}"
        pt_path = os.path.join(output_dir, save_name, "data.pt")
        if os.path.exists(pt_path):
            existing_names.append(save_name)
        else:
            run_ids_todo.append(rid)

    print(f"Processing {len(run_ids)} {dataset_name} runs with {num_workers} workers...")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Skipping {len(existing_names)} already-processed, {len(run_ids_todo)} remaining.")
    if target_reduction > 0:
        keep_pct = int((1.0 - target_reduction) * 100)
        print(f"  Decimation: target_reduction={target_reduction} (keeping ~{keep_pct}%)")

    order_list_new = []

    if num_workers <= 1:
        for rid in tqdm(run_ids_todo, desc=f"Processing {dataset_name}"):
            try:
                result = process_single_vtp(rid, cfg, target_reduction=target_reduction)
                if result is not None:
                    order_list_new.append(result)
            except Exception as e:
                print(f"  [FAIL] run_{rid}: {e}")
                traceback.print_exc()
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_single_vtp, rid, cfg, target_reduction=target_reduction): rid for rid in run_ids_todo
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {dataset_name}"):
                rid = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        order_list_new.append(result)
                except Exception as e:
                    print(f"  [FAIL] run_{rid}: {e}")
                    traceback.print_exc()

    all_names = existing_names + order_list_new
    all_names = sorted(set(all_names), key=lambda x: int(x.rsplit("_", 1)[-1]))

    order_path = os.path.join(output_dir, "order_.txt")
    with open(order_path, "w") as f:
        f.writelines(f"{name}\n" for name in all_names)

    print(f"\nThis run: {len(order_list_new)} processed.")
    print(f"Total in output dir: {len(all_names)} runs.")
    print(f"Order file: {order_path}")

    write_dataset_manifest(
        output_dir=output_dir,
        dataset_name=dataset_name,
        processed_count=len(all_names),
        total_candidates=len(run_ids),
        failed_count=max(0, len(run_ids) - len(all_names)),
        contract_version=PREPROCESS_CONTRACT_VERSION,
        source_root=input_dir,
    )

    return all_names


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Process VTP boundary meshes into GAOT .pt with pressure field.")
    parser.add_argument("dataset", choices=list(DATASETS.keys()), help="Which dataset to process (drivaer or ahmed)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run_ids", nargs="+", type=int, default=None)
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--target-reduction",
        type=float,
        default=0.0,
        help="Mesh decimation target (0.0-1.0, default 0.0 = no decimation). 0.6 = keep 40%% of vertices.",
    )
    args = parser.parse_args()

    process_all(
        dataset_name=args.dataset,
        run_ids=args.run_ids,
        max_runs=args.max_runs,
        num_workers=args.workers,
        output_dir=args.output_dir,
        target_reduction=args.target_reduction,
    )


if __name__ == "__main__":
    main()
