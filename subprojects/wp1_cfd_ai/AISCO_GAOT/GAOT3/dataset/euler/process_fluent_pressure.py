#!/usr/bin/env python3
"""
Process Fluent .cas.h5 / .dat.h5 result files into GAOT-compatible .pt files
with canonical Cp fields and sample-level conditioning.

For each run, extracts:
  - pos:        [N, 3]   vertex coordinates (surface nodes on car_surface)
    - x:          [N, 1]   pressure coefficient at each vertex
  - c:          [N, 3]   outward unit normals
  - face:       [3, F']  triangulated face connectivity (local vertex indices)
  - wall_shear: [N, 3]   wall shear stress at each vertex
    - pressure:   [N, 1]   raw pressure at each vertex
  - force:      [1]      converged drag force (from force report)
  - cd:         [1]      drag coefficient (from force report)
    - v_inf:      [1]      freestream speed for the sample
    - geom_extent:[3]      geometry extents before normalization
    - geom_aspect:[3]      extent ratios before normalization
    - conditioning:[1,7]   [v_inf, extent_xyz, aspect_xyz]
  - filename:   str

Usage:
    python process_fluent_pressure.py                           # all runs
    python process_fluent_pressure.py --run_ids 1 2 3           # specific runs
    python process_fluent_pressure.py --max_runs 5              # first 5 runs
    python process_fluent_pressure.py --workers 4               # parallel workers
"""

import argparse
import glob
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_preprocessing import AIR_DENSITY, V_INF_EULER, compute_geom_conditioning, compute_pressure_coefficient

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
FLUENT_DIR = "/cluster/work/fuge/AISCO/wp1_cfd_ai/FLUENT"
RESULTS_DIR = os.path.join(FLUENT_DIR, "RUN/RESULTS")
FORCE_DIR = os.path.join(FLUENT_DIR, "RUN/EULER_OUTPUT")
OUTPUT_DIR = "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler_cp_full"

SURFACE_ZONE_NAME = "car_surface"
# Fluent wall zone type (from Fluent HDF5 zoneType encoding)
WALL_ZONE_TYPE = 3

NUM_WORKERS = 4  # HDF5 reads + numpy heavy → moderate parallelism
MIN_FORCE_COLS = 3  # minimum columns in a valid force report line


# ---------------------------------------------------------------------------
#  Force report parser (reused from process_my_euler.py)
# ---------------------------------------------------------------------------
def parse_force_report(filepath):
    """
    Parse a Fluent force report .out file.
    Columns: iteration  drag_coefficient  drag_force
    Returns (cd, drag_force) from the last converged iteration.
    """
    last_cd = None
    last_force = None
    with open(filepath) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith(('"', "(")):
                continue
            parts = line.split()
            if len(parts) >= MIN_FORCE_COLS:
                try:
                    last_cd = float(parts[1])
                    last_force = float(parts[2])
                except ValueError:
                    continue
    return last_cd, last_force


# ---------------------------------------------------------------------------
#  Fluent HDF5 zone parser
# ---------------------------------------------------------------------------
class FluentZoneInfo:
    """Parsed face zone topology from a Fluent .cas.h5 file."""

    def __init__(self, cas_path):
        with h5py.File(cas_path, "r") as cas:
            zt = cas["meshes/1/faces/zoneTopology"]
            names_raw = zt["name"][0]
            if isinstance(names_raw, bytes):
                names_raw = names_raw.decode()
            self.names = names_raw.split(";")
            self.zone_ids = zt["id"][:]
            self.zone_types = zt["zoneType"][:]
            self.min_ids = zt["minId"][:]  # 1-based face IDs
            self.max_ids = zt["maxId"][:]

    def find_zone(self, name):
        """Return the topology index for a zone by name."""
        try:
            return self.names.index(name)
        except ValueError as e:
            raise ValueError(f"Zone '{name}' not found.  Available: {self.names}") from e

    def face_range_0based(self, idx):
        """Return (start, end) as 0-based indices into the global face arrays."""
        return int(self.min_ids[idx]) - 1, int(self.max_ids[idx])

    def wall_zone_indices_sorted(self):
        """Return topology indices of wall zones, sorted by ascending minId."""
        walls = [i for i, t in enumerate(self.zone_types) if t == WALL_ZONE_TYPE]
        return sorted(walls, key=lambda i: self.min_ids[i])

    def wall_data_offset(self, target_zone_idx):
        """
        Compute the offset into wall-specific data arrays (e.g. SV_WALL_SHEAR)
        for a given wall zone.

        Wall-specific data is stored for all wall zones concatenated in order
        of ascending global face ID.
        """
        sorted_walls = self.wall_zone_indices_sorted()
        offset = 0
        for wi in sorted_walls:
            if wi == target_zone_idx:
                return offset
            n_faces = int(self.max_ids[wi] - self.min_ids[wi]) + 1
            offset += n_faces
        raise ValueError(f"Zone index {target_zone_idx} is not a wall zone")


# ---------------------------------------------------------------------------
#  Core extraction
# ---------------------------------------------------------------------------
def _extract_face_connectivity(cas_path, zone_start, zone_end):
    """
    Extract face-to-node connectivity for faces in [zone_start, zone_end).
    Indices are 0-based.

    Returns
    -------
    face_nodes : list[np.ndarray]
        Each element is an array of 0-based node indices for one face.
    """
    with h5py.File(cas_path, "r") as cas:
        nnodes = cas["meshes/1/faces/nodes/1/nnodes"][:]
        nodes_flat = cas["meshes/1/faces/nodes/1/nodes"][:]

    # Build cumulative offset into the flat node array
    offsets = np.zeros(len(nnodes) + 1, dtype=np.int64)
    np.cumsum(nnodes, out=offsets[1:])

    face_nodes = []
    for fi in range(zone_start, zone_end):
        s = offsets[fi]
        e = offsets[fi + 1]
        # Fluent node IDs are 1-based → convert to 0-based
        face_nodes.append(nodes_flat[s:e] - 1)

    return face_nodes


def _triangulate_faces(face_nodes, global_to_local):
    """
    Fan-triangulate polygonal faces and remap to local vertex indices.

    Parameters
    ----------
    face_nodes : list[np.ndarray]
        Each element is an array of 0-based global node IDs.
    global_to_local : dict
        Maps global node ID → local vertex index.

    Returns
    -------
    triangles : np.ndarray [F_tri, 3]
        Triangle face list with local vertex indices.
    """
    tris: list[Any] = []
    for nodes in face_nodes:
        local = np.array([global_to_local[n] for n in nodes])
        # Fan triangulation: (v0, v1, v2), (v0, v2, v3), ...
        tris.extend([local[0], local[k], local[k + 1]] for k in range(1, len(local) - 1))
    return np.array(tris, dtype=np.int64)


def _compute_vertex_normals(pos, triangles):
    """
    Compute area-weighted vertex normals from triangulated mesh.

    Parameters
    ----------
    pos : np.ndarray [N, 3]
    triangles : np.ndarray [F, 3]

    Returns
    -------
    normals : np.ndarray [N, 3]   (unit normals)
    """
    v0 = pos[triangles[:, 0]]
    v1 = pos[triangles[:, 1]]
    v2 = pos[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)  # [F, 3], magnitude = 2*area

    vertex_normals = np.zeros_like(pos)
    for j in range(3):
        np.add.at(vertex_normals, triangles[:, j], face_normals)

    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vertex_normals / norms


def _interpolate_face_to_vertex(values_per_face, face_nodes, global_to_local, n_vertices, _pos_np=None, _triangles_np=None):
    """
    Interpolate face-center values to vertices using area-weighted averaging.

    If pos and triangles are provided, uses triangle area weighting.
    Otherwise falls back to simple averaging (each face contributes equally
    to its vertices).

    Parameters
    ----------
    values_per_face : np.ndarray [n_faces] or [n_faces, D]
    face_nodes : list[np.ndarray]
    global_to_local : dict
    n_vertices : int
    pos_np : np.ndarray [N, 3], optional
    triangles_np : np.ndarray [F_tri, 3], optional

    Returns
    -------
    vertex_values : np.ndarray [N] or [N, D]
    """
    is_vector = values_per_face.ndim > 1
    if is_vector:
        n_dims = values_per_face.shape[1]
        accum = np.zeros((n_vertices, n_dims), dtype=np.float64)
    else:
        accum = np.zeros(n_vertices, dtype=np.float64)
    weight = np.zeros(n_vertices, dtype=np.float64)

    for fi, nodes in enumerate(face_nodes):
        local_ids = [global_to_local[n] for n in nodes]
        val = values_per_face[fi]
        # Simple equal-weight contribution to each vertex of the face
        w = 1.0 / len(local_ids)
        for lid in local_ids:
            accum[lid] += val * w
            weight[lid] += w

    # Normalize
    mask = weight > 0
    if is_vector:
        accum[mask] /= weight[mask, None]
    else:
        accum[mask] /= weight[mask]

    return accum.astype(np.float32)


# ---------------------------------------------------------------------------
#  Single-run processor
# ---------------------------------------------------------------------------
def process_single_run(run_id, results_dir=RESULTS_DIR, force_dir=FORCE_DIR, output_dir=OUTPUT_DIR):  # noqa: PLR0915
    """
    Process one Fluent run into a GAOT-compatible .pt file with pressure field.

    Returns the save name on success, None on failure.
    """
    cas_path = os.path.join(results_dir, f"RUN_{run_id:03d}_results.cas.h5")
    dat_path = os.path.join(results_dir, f"RUN_{run_id:03d}_results.dat.h5")
    force_path = os.path.join(force_dir, f"RUN_{run_id:03d}_force_report.out")

    if not os.path.exists(cas_path) or not os.path.exists(dat_path):
        print(f"  [SKIP] RUN_{run_id:03d}: missing .cas.h5 or .dat.h5")
        return None

    # ---- 1. Parse zone topology ----
    zinfo = FluentZoneInfo(cas_path)
    car_idx = zinfo.find_zone(SURFACE_ZONE_NAME)
    zone_start, zone_end = zinfo.face_range_0based(car_idx)
    n_faces = zone_end - zone_start
    car_wall_offset = zinfo.wall_data_offset(car_idx)

    # ---- 2. Extract face-to-node connectivity for car_surface ----
    face_nodes = _extract_face_connectivity(cas_path, zone_start, zone_end)

    # ---- 3. Unique surface nodes → coordinates ----
    unique_global_ids = np.unique(np.concatenate(face_nodes))
    global_to_local = {gid: lid for lid, gid in enumerate(unique_global_ids)}
    n_vertices = len(unique_global_ids)

    with h5py.File(cas_path, "r") as cas:
        all_coords = cas["meshes/1/nodes/coords/1"][:]  # [total_nodes, 3]
    pos_np = all_coords[unique_global_ids].astype(np.float32)  # [N, 3]

    # ---- 4. Triangulate polygon faces ----
    triangles = _triangulate_faces(face_nodes, global_to_local)  # [F_tri, 3]

    # ---- 5. Compute vertex normals ----
    normals_np = _compute_vertex_normals(pos_np, triangles)  # [N, 3]

    # Canonicalize Euler orientation to match the rest of the pipeline.
    pos_np[:, 0] = -pos_np[:, 0]
    normals_np[:, 0] = -normals_np[:, 0]

    # ---- 6. Extract face-center pressure & interpolate to vertices ----
    with h5py.File(dat_path, "r") as dat:
        face_pressure_all = dat["results/1/phase-1/faces/SV_P/1"][:]
        face_pressure = face_pressure_all[zone_start:zone_end]  # [n_faces]

        # Wall shear stress
        wall_shear_all = dat["results/1/phase-1/faces/SV_WALL_SHEAR/1"][:]
        face_shear = wall_shear_all[car_wall_offset : car_wall_offset + n_faces]  # [n_faces, 3]

    vertex_pressure = _interpolate_face_to_vertex(face_pressure, face_nodes, global_to_local, n_vertices)  # [N]

    vertex_shear = _interpolate_face_to_vertex(face_shear, face_nodes, global_to_local, n_vertices)  # [N, 3]

    # ---- 7. Parse force report ----
    force_val = float("nan")
    cd_val = float("nan")
    if os.path.exists(force_path):
        cd_parsed, force_parsed = parse_force_report(force_path)
        if force_parsed is not None:
            force_val = -force_parsed
        if cd_parsed is not None:
            cd_val = cd_parsed

    pressure_tensor = torch.tensor(vertex_pressure, dtype=torch.float32).unsqueeze(-1)
    cp_tensor = compute_pressure_coefficient(pressure_tensor, V_INF_EULER, AIR_DENSITY)
    pos_tensor = torch.tensor(pos_np, dtype=torch.float32)
    geom_extent, geom_aspect, conditioning = compute_geom_conditioning(pos_tensor, V_INF_EULER)

    # ---- 8. Build PyG Data and save ----
    data = Data(
        pos=pos_tensor,  # [N, 3]
        x=cp_tensor,  # [N, 1]
        c=torch.tensor(normals_np, dtype=torch.float32),  # [N, 3]
        face=torch.tensor(triangles.T, dtype=torch.long),  # [3, F_tri]
        wall_shear=torch.tensor(vertex_shear, dtype=torch.float32),  # [N, 3]
        force=torch.tensor([force_val], dtype=torch.float32),  # [1]
        cd=torch.tensor([cd_val], dtype=torch.float32),  # [1]
    )
    data.pressure = pressure_tensor
    data.cp = cp_tensor
    data.v_inf = torch.tensor([V_INF_EULER], dtype=torch.float32)
    data.geom_extent = geom_extent
    data.geom_aspect = geom_aspect
    data.conditioning = conditioning
    data.filename = f"RUN_{run_id:03d}"
    data.n_surface_faces = n_faces

    save_name = f"euler_run_{run_id}"
    save_dir = os.path.join(output_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(data, os.path.join(save_dir, "data.pt"))
    return save_name


# ---------------------------------------------------------------------------
#  Batch processing
# ---------------------------------------------------------------------------
def process_all_runs(
    results_dir=RESULTS_DIR, force_dir=FORCE_DIR, output_dir=OUTPUT_DIR, run_ids=None, max_runs=None, num_workers=NUM_WORKERS
):
    """
    Process all (or selected) Fluent runs into .pt files.

    Parameters
    ----------
    run_ids : list[int], optional
        Specific run numbers to process. If None, auto-detect from files.
    max_runs : int, optional
        Process at most this many runs.
    num_workers : int
        Number of parallel workers.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect available runs
    if run_ids is None:
        cas_files = sorted(glob.glob(os.path.join(results_dir, "RUN_*_results.cas.h5")))
        run_ids = []
        for cf in cas_files:
            m = re.search(r"RUN_(\d+)_results\.cas\.h5", os.path.basename(cf))
            if m:
                run_ids.append(int(m.group(1)))
        run_ids.sort()

    if max_runs is not None:
        run_ids = run_ids[:max_runs]

    print(f"Processing {len(run_ids)} Fluent runs with {num_workers} workers...")
    print(f"  Results: {results_dir}")
    print(f"  Force:   {force_dir}")
    print(f"  Output:  {output_dir}")

    order_list = []

    if num_workers <= 1:
        for rid in tqdm(run_ids, desc="Processing runs"):
            try:
                result = process_single_run(rid, results_dir, force_dir, output_dir)
                if result is not None:
                    order_list.append(result)
            except Exception as e:
                print(f"  [FAIL] RUN_{rid:03d}: {e}")
                traceback.print_exc()
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_run, rid, results_dir, force_dir, output_dir): rid for rid in run_ids}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing runs"):
                rid = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        order_list.append(result)
                except Exception as e:
                    print(f"  [FAIL] RUN_{rid:03d}: {e}")
                    traceback.print_exc()

    # Sort numerically
    order_list.sort(key=lambda x: int(x.split("_")[-1]))

    order_path = os.path.join(output_dir, "order_.txt")
    with open(order_path, "w") as f:
        f.writelines(f"{name}\n" for name in order_list)

    print(f"\nDone! Processed {len(order_list)}/{len(run_ids)} runs.")
    print(f"Order file: {order_path}")
    print(f"Data saved to: {output_dir}")
    return order_list


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Process Fluent .cas.h5/.dat.h5 into GAOT .pt with pressure field.")
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--force_dir", default=FORCE_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--run_ids", nargs="+", type=int, default=None, help="Specific run IDs to process (e.g., 1 2 3)")
    parser.add_argument("--max_runs", type=int, default=None, help="Process at most N runs")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    process_all_runs(
        results_dir=args.results_dir,
        force_dir=args.force_dir,
        output_dir=args.output_dir,
        run_ids=args.run_ids,
        max_runs=args.max_runs,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
