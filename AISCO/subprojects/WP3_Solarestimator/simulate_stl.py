from datetime import datetime

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import pytz
import trimesh
from AISCO.subprojects.WP3_Solarestimator.pv_index_for_run import pv_perf_index
from AISCO.subprojects.WP3_Solarestimator.read_mesh import facenormal_angle, powerratio_iam, shadowing_factor, sky_face
from AISCO.subprojects.WP3_Solarestimator.sun import get_sun_direction

# data-Folder relative to AISCO-code
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
# Folder to safe solardecks
SELECTED_FOLDER = os.path.join(DATA_FOLDER, "selected_solardecks")

# create Folder "selected_solardecks" if not available
os.makedirs(SELECTED_FOLDER, exist_ok=True)


def simulate_stl(path_csv: str, mesh_path: str, max_area=6.0):
#llinsi
    """
    Takes a CSV file that defines the route (with longitude/latitude, altitude above sea level, and local time)
    that the car will take, the file path of the solar car to be simulated (stl file) and the maximum solar area.
    Calculates a performance value (between 0 and 1) and returns the selected solar area (in m^3)
    and the corresponding mesh.

    Args:
        path_mesh: File-Path (str) of the csv, which defines the path the solar car should drive.
        mesh_path: File-Path (str) of the solarcar to be simulated.
        max_area: Maximum area (floar in m^3) that should be considered (per default: 6.0)
    Returns:
        pv_performance: Value representing the ratio of a solar deck to the same area
                        with a perfect angle of solar radiation (90 degrees) -> (float value between 0 and 1)
        selected_solardeck: Mesh (file format Trimmest) representing the selected solar area.
        solar_area: Solar area of the selceted solardeck (just litte faces, no solar cells) in m^3 -> float
        
    """
    #loading the mesh to be simulated
    mesh=trimesh.load_mesh(mesh_path)
    top_mesh = sky_face(mesh)

    # reading the path_csv and converting it to a DataFrame
    # rows: points on the driving path
    # colums: Lat,Long,altitude,time (e.g -12.45,130.83,100,2025-09-15 10:00:00)
    df=pd.read_csv(path_csv)

    # convert to UTC
    df['time'] = pd.to_datetime(
    df['time'],
    format='%Y-%m-%d %H:%M:%S'
    )
    df_time_acst = df['time'].dt.tz_localize('Australia/Darwin', ambiguous='infer')
    df['time_utc'] = df_time_acst.dt.tz_convert('UTC')

    # creating a Table/Matrix with corresponing performance values
    # rows: Facettes
    # columns: point in time

    #This table contains the weight of each facette of the mesh for every point in time.
    # It could forexamplke be used to identy rows/facettes with the best correlation, 
    # (areas that perform and behave equally) this might be interesting to create a layout lateron.
    res = pd.DataFrame()
    for index, row in df.iterrows():

        sun_dir = get_sun_direction(
            row['Lat'],
            row['Long'],
            row['altitude'],
            row['time_utc']
            )
        angles = facenormal_angle(top_mesh,sun_dir)
        #shadow_mask = shadowing_factor(top_mesh, -1*sun_dir)
        power_weight = powerratio_iam(angles)
        #power_weight[shadow_mask]=0
        res[f'{index}']=power_weight

    # Computes the performance of the solar deck over the entire run
    # (average over all points in time along the way)
    # returns the performance index (0-1) and the selected solar deck (mesh connected faces)
    performance_index, selected_faces = pv_perf_index(top_mesh, res)
    
    return performance_index, selected_faces


def process_stl_file(stl_folder: str, filename: str, path_csv: str, max_area: float = 6.0) -> dict:
    """
    Executes the solar deck simulation for a single STL file.

    Loads the STL mesh, runs the simulation using simulate_stl(), measures the computation time,
    and saves the resulting selected solar deck mesh as an STL file inside
    /data/selected_solardecks/.

    Args:
        stl_folder: Folder (str) containing the STL files to be simulated.
        filename: Name (str) of the STL file to be processed.
        path_csv: Path (str) to the route definition CSV file
                  (containing latitude, longitude, altitude, and timestamp).
        max_area: Maximum solar area (float, in m²) to be considered (default: 6.0).

    Returns:
        dict: Dictionary with the following fields:
            - "filename": name of the processed STL file.
            - "performance_index": float representing the PV performance [0–1].
            - "duration_s": runtime in seconds (float).
            - "output_file": relative path (str) to the exported solardeck STL file.
    """
    mesh_path = os.path.join(stl_folder, filename)
    start_time = time.time()

    try:
        performance_index, selected_faces = simulate_stl(path_csv, mesh_path, max_area=max_area)
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
            "output_file": os.path.relpath(output_path, start=os.getcwd())
        }

    except Exception as e:
        print(f"❌ Error while processing {filename}: {e}")
        duration = round(time.time() - start_time, 2)
        return {
            "filename": filename,
            "performance_index": None,
            "duration_s": duration,
            "output_file": None
        }
    

def run_parallel_simulations(
        stl_folder: str, 
        path_csv: str, 
        output_csv: str = "simulation_results.csv", 
        max_area: float = 6.0, 
        n_jobs: int = 4
        ) -> pd.DataFrame:    
    
    """
    Runs the solar deck performance simulation for all STL files in a given folder in parallel.

    Each STL file is processed by process_stl_file(), which executes the simulation,
    measures runtime, and saves the selected solar deck. The aggregated results are stored
    in a DataFrame and exported as a CSV file.

    Args:
        stl_folder: Folder (str) containing all STL files to be simulated.
        path_csv: Path (str) to the route CSV file (latitude, longitude, altitude, time).
        output_csv: Path (str) to the output CSV file where results will be stored.
        max_area: Maximum solar area (float, in m²) to be considered per simulation (default: 6.0).
        n_jobs: Number of CPU cores to use in parallel (-1 uses all available cores).

    Returns:
        pd.DataFrame: DataFrame containing for each STL file:
            - filename
            - performance_index
            - duration_s
            - output_file
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
    df.to_csv(output_csv, index=False)

    print(f"\n✅ Simulation completed. Results saved to {output_csv}")
    return df   

