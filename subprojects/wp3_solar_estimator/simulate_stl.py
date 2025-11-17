import os
import time
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import trimesh
from tqdm import tqdm
from joblib import Parallel, delayed


from subprojects.wp3_solar_estimator.pv_index_for_run import pv_perf_index
from subprojects.wp3_solar_estimator.read_mesh import (
    facenormal_angle,
    powerratio_iam,
    shadowing_factor,
    sky_face,
)
from subprojects.wp3_solar_estimator.sun import get_sun_direction

# data-Folder relative to AISCO-code
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
# Folder to safe solardecks
SELECTED_FOLDER = os.path.join(DATA_FOLDER, "selected_solardecks")

# create Folder "selected_solardecks" if not available
SELECTED_FOLDER = unicodedata.normalize("NFC", SELECTED_FOLDER)
os.makedirs(SELECTED_FOLDER, exist_ok=True)


def simulate_stl(path_csv: str, mesh_path: str, max_area=6.0):
    # llinsi
    """
    Run a single STL mesh against a race route and return performance + deck mesh.

    Args:
        path_csv (str): Route definition with `Lat,Long,altitude,time` columns
            (time in local timezone; converted to UTC internally).
        mesh_path (str): Absolute or relative path to the STL mesh to evaluate.
        max_area (float): Maximum deployable area in square metres (default 6 m²).

    Returns:
        tuple[float, trimesh.Trimesh]: Performance index in [0, 1] relative to the
        same area at perfect incidence, and the mesh containing the selected faces.
    """
    # loading the mesh to be simulated
    mesh = trimesh.load_mesh(mesh_path)
    top_mesh = sky_face(mesh)

    # reading the path_csv and converting it to a DataFrame
    # rows: points on the driving path
    # colums: Lat,Long,altitude,time (e.g -12.45,130.83,100,2025-09-15 10:00:00)
    df = pd.read_csv(path_csv)

    # convert to UTC
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
    df_time_acst = df["time"].dt.tz_localize("Australia/Darwin", ambiguous="infer")
    df["time_utc"] = df_time_acst.dt.tz_convert("UTC")

    # creating a Table/Matrix with corresponing performance values
    # rows: Facettes
    # columns: point in time

    # This table contains the weight of each facette of the mesh for every point in time.
    # It could forexamplke be used to identy rows/facettes with the best correlation,
    # (areas that perform and behave equally) this might be interesting to create a layout lateron.
    res = pd.DataFrame()
    for index, row in df.iterrows():
        sun_dir = get_sun_direction(
            row["Lat"], row["Long"], row["altitude"], row["time_utc"]
        )
        angles = facenormal_angle(top_mesh, sun_dir)
        # shadow_mask = shadowing_factor(top_mesh, -1*sun_dir)
        power_weight = powerratio_iam(angles)
        # power_weight[shadow_mask]=0
        res[f"{index}"] = power_weight

    # Computes the performance of the solar deck over the entire run
    # (average over all points in time along the way)
    # returns the performance index (0-1) and the selected solar deck (mesh connected faces)
    performance_index, selected_faces = pv_perf_index(top_mesh, res)

    return performance_index, selected_faces


def process_stl_file(
    stl_folder: str, filename: str, path_csv: str, max_area: float = 6.0
) -> dict:
    """
    Wrapper around `simulate_stl` for a single mesh, including export + timing.

    Args:
        stl_folder (str): Directory containing the STL file.
        filename (str): Name of the STL file relative to `stl_folder`.
        path_csv (str): Route CSV used for all candidates.
        max_area (float): Maximum selectable area in square metres.

    Returns:
        dict: Contains `filename`, `performance_index`, and `duration_s`.
    """
    mesh_path = os.path.join(stl_folder, filename)
    start_time = time.time()

    try:
        performance_index, selected_faces = simulate_stl(
            path_csv, mesh_path, max_area=max_area
        )
        duration = round(time.time() - start_time, 2)

        # Define output filename and path
        base_name = os.path.splitext(filename)[0]  # e.g. car_01
        output_name = f"solardeck_{base_name}.stl"
        output_path = os.path.join(SELECTED_FOLDER, output_name)

        # Export the selected solar deck as STL
        selected_faces.export(output_path)

        return {
            "filename": filename,
            "performance_index": performance_index,
            "duration_s": duration,
        }

    except Exception as e:
        print(f"❌ Error while processing {filename}: {e}")
        duration = round(time.time() - start_time, 2)
        return {
            "filename": filename,
            "performance_index": None,
            "duration_s": duration,
        }


def run_parallel_simulations(
    stl_folder: str,
    path_csv: str,
    output_csv: str = "simulation_results.csv",
    max_area: float = 6.0,
    n_jobs: int = 4,
) -> pd.DataFrame:
    """
    Batch runner that evaluates every STL in a folder using parallel workers.

    Args:
        stl_folder (str): Folder containing the candidate meshes.
        path_csv (str): Route CSV shared by all runs.
        output_csv (str): Where to store the aggregated results table.
        max_area (float): Maximum selectable area in square metres.
        n_jobs (int): Number of workers (`-1` uses all logical cores).

    Returns:
        pd.DataFrame: Sorted table containing filename, performance index, and runtime.
    """
    stl_files = [f for f in os.listdir(stl_folder) if f.lower().endswith(".stl")]
    if not stl_files:
        raise FileNotFoundError(f"No STL files found in folder {stl_folder}.")

    print(f"🛰️  Starting simulation for {len(stl_files)} STL files...")
    print(f"📁 Selected solar decks will be saved in: {SELECTED_FOLDER}\n")

    # Parallelized processing with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_stl_file)(stl_folder, filename, path_csv, max_area)
        for filename in tqdm(stl_files, desc="Simulating STL files")
    )

    # Convert results to DataFrame and export as CSV
    df = pd.DataFrame(results)
    df = df.sort_values(by="filename", ascending=True)
    df_export = df.drop(columns=["output_file"], errors="ignore")
    df_export.to_csv(output_csv, index=False)

    print(f"\n✅ Simulation completed. Results saved to {output_csv}")
    return df_export
