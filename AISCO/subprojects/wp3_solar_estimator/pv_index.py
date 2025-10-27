import numpy as np
import trimesh
from read_mesh import facenormal_angle, powerratio_iam, sky_face


def pv_perf_index(mesh,sun_direction, max_area=6.0,):

    max_area = max_area * 1e6  # Convert m² to mm²
    top_mesh = sky_face(mesh)
    angles = facenormal_angle(top_mesh,sun_direction)
    weights = powerratio_iam(angles,b0=0.15) # default b0 is 0.05


    #get areas and weighted areas
    areas = top_mesh.area_faces

    # Sortafter performance weight (best are first)
    sort_weights = np.argsort(weights)[::-1]

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
    best_face_inices= np.where(selection_mask)[0]
    selected_face_indices = original_face_indices[best_face_inices]



    #show the selected faces as a new mesh



    #calculate total performance of selected faces
    # get weights and areas of selected faces
    selected_performance = weights[selected_face_indices]
    selected_areas = areas[selected_face_indices]

    # apply weights to areas
    weighted_areas = selected_areas * selected_performance

    # sum up total weighted performance
    total_weighted_area = np.sum(weighted_areas)


    #calculate performance index
    performance_index = total_weighted_area / max_area
    #print( "total area",np.sum(areas), "selected area",np.sum(selected_areas), ""weighted area",total_weighted_area)


    #construct best mesh for  later visualization
    best_faces = mesh.faces[selected_face_indices]  # noqa: F841
    best_mesh = trimesh.Trimesh(vertices=top_mesh.vertices, faces=top_mesh.faces[selected_face_indices])

    ##In future we might want to pass a Performance object or smth, to get further indices like, atual used area etc.
    # --- WP4 Feel free to change the output to our needs ---
    return performance_index,best_mesh
