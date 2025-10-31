import pandas as pd
import numpy as np
import trimesh
import pytz
from read_mesh import facenormal_angle,powerratio_iam,shadowing_factor,sky_face
from sun import get_sun_direction
from datetime import datetime
from pv_index_for_run import pv_perf_index

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
        top_mesh = sky_face(mesh)
        angles = facenormal_angle(top_mesh,sun_dir)
        shadow_mask = shadowing_factor(top_mesh, -1*sun_dir)
        power_weight = powerratio_iam(angles)
        power_weight[shadow_mask]=0
        res[f'{index}']=power_weight

    # Computes the performance of the solar deck over the entire run
    # (average over all points in time along the way)
    # returns the performance index (0-1) and the selected solar deck (mesh connected faces)
    performance_index, selected_faces = pv_perf_index(top_mesh, res)
    
    return performance_index, selected_faces