import os
import sys

import numpy as np
import trimesh

# --- Configuration ---
# 1. SET THIS TO THE PATH OF YOUR STL FILE
# FILE_TO_LOAD = r"C:\Users\elias\Documents\AISCO\AISCO-code\AISCO\subprojects\WP1_CFD_AI\crude_estimator\silvretta_crude.stl"
FILE_TO_LOAD = r"C:\Users\elias\Documents\AISCO\AISCO-code\AISCO\subprojects\WP1_CFD_AI\crude_estimator\test_box.stl"

# 2. SET THE NUMBER OF SLICES YOU WANT
N_SLICES = 20

# 3. SET THE MANUAL WEIGHTS
# The length of this list MUST be equal to N_SLICES
MANUAL_WEIGHTS = [
    4.0,  # Slice 1
    3.39,  # Slice 2
    2.78,  # Slice 3
    2.07,  # Slice 4
    1.37,  # Slice 5
    1.2,  # Slice 6
    1.13,  # Slice 7
    1.07,  # Slice 8
    1.0,  # Slice 9
    1.0,  # Slice 10 (Middle)
    1.05,  # Slice 11 (Middle)
    1.1,  # Slice 12
    1.19,  # Slice 13
    1.3,  # Slice 14
    1.61,  # Slice 15
    2.0,  # Slice 16
    2.22,  # Slice 17
    2.5,  # Slice 18
    3.25,  # Slice 19
    4.0,  # Slice 20 (Rear)
]
# ----- Configuration End -----
realistic_factor = 11  # Adjust this factor based on CFD of Silvretta


def approximate_drag_with_vertex_slices(filename, num_slices, manual_weights_list):
    """
    Loads an STL, groups its VERTICES into N X-axis slices,
    and calculates a normalized, weighted sum of their frontal areas
    using a manually provided list of weights.
    """

    # --- 1. Load Mesh ---
    print(f"Loading mesh: {filename}")
    if not os.path.exists(filename):
        print(f"FATAL ERROR: File not found at '{filename}'")
        return

    try:
        mesh = trimesh.load_mesh(filename, process=False)
        vertices = mesh.vertices
        if vertices.shape[0] == 0:
            print("Error: Mesh loaded but contains no vertices.")
            return
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    print(f"Loaded mesh with {len(vertices)} vertices.")

    # --- 2. Slicing Logic (based on vertices) ---
    scene = trimesh.Scene()
    slice_frontal_areas = []  # List to store each slice's area

    min_x, max_x = vertices[:, 0].min(), vertices[:, 0].max()
    split_points = np.linspace(min_x, max_x, num_slices + 1)

    print(
        f"Slicing vertices into {num_slices} parts along X-axis from {min_x:.2f} to {max_x:.2f}"
    )

    # Define alternating colors for visualization
    alternating_colors = [[255, 0, 0], [0, 0, 255]]  # Red, Blue
    n_colors = len(alternating_colors)

    print("\n--- Analyzing Slices ---")

    for i in range(num_slices):
        x_start = split_points[i]
        x_end = split_points[i + 1]

        if i == (num_slices - 1):  # Handle the last slice
            v_mask = (vertices[:, 0] >= x_start) & (vertices[:, 0] <= x_end)
        else:
            v_mask = (vertices[:, 0] >= x_start) & (vertices[:, 0] < x_end)

        slice_vertices = vertices[v_mask]

        if slice_vertices.shape[0] == 0:
            print(f"Slice {i + 1}: Found 0 vertices. Area = 0.0000")
            slice_frontal_areas.append(0.0)
            continue

        # --- 3. Calculate Bounding Box and Area ---
        try:
            min_y, max_y = slice_vertices[:, 1].min(), slice_vertices[:, 1].max()
            min_z, max_z = slice_vertices[:, 2].min(), slice_vertices[:, 2].max()

            L_x = x_end - x_start
            L_y = max_y - min_y
            L_z = max_z - min_z

            if L_y < 1e-6:
                L_y = 0.0
            if L_z < 1e-6:
                L_z = 0.0

            frontal_area = L_y * L_z
            slice_frontal_areas.append(frontal_area)

            print(
                f"Slice {i + 1}: Dims (Lx, Ly, Lz) = [{L_x:6.2f}, {L_y:6.2f}, {L_z:6.2f}], Frontal Area = {frontal_area:8.4f}"
            )

            # --- 4. Build Wireframe Visualization ---
            box_extents = [L_x, L_y, L_z]
            box_center = [
                (x_start + x_end) / 2,
                (min_y + max_y) / 2,
                (min_z + max_z) / 2,
            ]
            box_transform = trimesh.transformations.translation_matrix(box_center)
            box_wireframe = trimesh.path.creation.box_outline(
                extents=box_extents, transform=box_transform
            )
            color_index = i % n_colors
            box_wireframe.colors = [alternating_colors[color_index]] * len(
                box_wireframe.entities
            )
            scene.add_geometry(box_wireframe, geom_name=f"box_wireframe_{i + 1}")

        except Exception as e:
            print(f"Error processing slice {i + 1}: {e}")
            slice_frontal_areas.append(0.0)

    # --- 5. Final Output ---

    # --- CALCULATE YOUR "MANUAL" WEIGHTED LOSS ---

    # Check if the weights list matches N_SLICES
    if len(manual_weights_list) != num_slices:
        print(
            f"FATAL ERROR: MANUAL_WEIGHTS list has {len(manual_weights_list)} items, but N_SLICES is {num_slices}."
        )
        print(
            "Please make sure the number of items in the MANUAL_WEIGHTS list is equal to N_SLICES."
        )
        return

    weights = manual_weights_list  # Use the manual list directly

    final_weighted_loss = 0.0
    print("------------------------")
    print("Calculating Weighted Loss:")
    for i in range(num_slices):
        area = slice_frontal_areas[i]
        weight = weights[i]
        weighted_area = area * weight
        final_weighted_loss += weighted_area
        print(
            f"Slice {i + 1}: Area = {area:8.4f} * Weight = {weight:4.2f} -> {weighted_area:8.4f}"
        )

    print("========================")
    print(f"Final Weighted Loss: {final_weighted_loss:8.4f}")
    print("========================")

    # --- NORMALIZE THE LOSS ---
    A_max = np.max(slice_frontal_areas)

    if A_max > 1e-6 and num_slices > 0:
        # Normalize by N_SLICES, the biggest area, and the realistic factor
        normalized_loss = final_weighted_loss / (A_max * num_slices * realistic_factor)

        print(
            f"Normalization Factors: A_max = {A_max:8.4f}, N_Slices = {num_slices}, Realistic_Factor = {realistic_factor}"
        )
        print(
            f"Normalized Loss (Loss / (A_max * N_Slices * Factor)): {normalized_loss:8.4f}"
        )
        print("========================")
    else:
        print("Cannot normalize: Max area or N_Slices is zero.")

    print("------------------------")

    # Add the original mesh vertices as a point cloud for context
    point_cloud = trimesh.points.PointCloud(vertices)
    point_cloud.colors = [128, 128, 128, 50]  # Semi-transparent gray
    scene.add_geometry(point_cloud, geom_name="point_cloud")

    # --- Export scene ---

    # Create an output filename based on the input STL file
    base_name = os.path.basename(filename)
    file_root = os.path.splitext(base_name)[0]
    output_filename = f"{file_root}_visualization.glb"

    try:
        scene.export(output_filename)
        print(f"\nSuccessfully saved visualization to {output_filename}")
        print(
            f"You can open this .glb file in any 3D viewer to see the {num_slices} boxes."
        )
    except Exception as e:
        print(f"\nError exporting scene: {e}")


if __name__ == "__main__":
    # Pass the manual weights list to the function
    approximate_drag_with_vertex_slices(FILE_TO_LOAD, N_SLICES, MANUAL_WEIGHTS)
