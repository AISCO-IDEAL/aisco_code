import numpy as np
import pandas as pd
import trimesh
from AISCO.subprojects.WP3_Solarestimator.read_mesh import (
    facenormal_angle,
    powerratio_iam,
    sky_face,
)

from subprojects.wp3_solar_estimator.read_mesh import (
    facenormal_angle,
    sky_face,
    powerratio_iam,
)


def pv_perf_index(
    top_mesh,
    periodic_power_weights,
    max_area=6.0,
):
    # llinsi, #baural
    """
    Computes an performance index for an entire run
    Args:
        top_mesh: mesh of the solar deck
        periodic_power_weights: DataFrame (pd.DataFrame) with power_weights over periodic time snapshots
        max_area: Maximum area that should be considered.
    Returns:
        (performance_index, best_mesh):
        **performance_index:** float in range [0;1] \n
        **best_mesh:** The selection mask of the best perfoming facettes within max_area
    """

    max_area = max_area * 1e6  # Convert m² to mm²
    average_weights = periodic_power_weights.mean(axis=1).to_numpy()

    # get areas and weighted areas
    areas = (
        top_mesh.area_faces
    )  # gives area of each face (triangle, return type: np.array)

    # Sortafter performance weight (best are first)
    sort_weights = np.argsort(average_weights)[::-1]

    # Sort Areas corrsespondingly to performance weights
    sorted_areas = areas[sort_weights]

    # Store the original indices of the faces before sorting
    # (takes the sorted indices and maps them back to original)
    original_face_indices = np.arange(len(areas))[sort_weights]

    #  Cumulative area (rolling sum)
    cumulative_area = np.cumsum(sorted_areas)

    #  Logische Maske erstellen
    # Maske ist TRUE, solange die kumulierte Fläche <= max_area ist
    selection_mask = cumulative_area <= max_area

    # Extract the best face indices based on the selection mask
    best_face_inices = np.where(selection_mask)[0]
    selected_face_indices = original_face_indices[best_face_inices]

    # show the selected faces as a new mesh

    # calculate total performance of selected faces
    # get weights and areas of selected faces
    selected_performance = average_weights[selected_face_indices]
    selected_areas = areas[selected_face_indices]

    # apply weights to areas
    weighted_areas = selected_areas * selected_performance

    # sum up total weighted performance
    total_weighted_area = np.sum(weighted_areas)

    # calculate performance index
    performance_index = total_weighted_area / max_area
    # print( "total area",np.sum(areas), "selected area",np.sum(selected_areas), ""weighted area",total_weighted_area)

    # construct best mesh for  later visualization
    # best_faces = mesh.faces[selected_face_indices]
    best_mesh = trimesh.Trimesh(
        vertices=top_mesh.vertices, faces=top_mesh.faces[selected_face_indices]
    )

    ##In future we might want to pass a Performance object or smth, to get further indices like, atual used area etc.
    return performance_index, best_mesh
