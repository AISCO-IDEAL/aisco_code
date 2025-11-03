import os

import numpy as np
import trimesh

# --- Configuration ---
# 1. SET THIS TO THE PATH OF YOUR STL FILE
FILE_TO_LOAD = r"C:\Users\elias\Documents\AISCO\AISCO-code\AISCO\subprojects\WP1_CFD_AI\crude_estimator\silvretta_big_rot.stl"

# 2. SET THE NUMBER OF SLICES YOU WANT
N_SLICES = 20

# 3. SET THE SLICING AXIS AND DIRECTION
# SLICING_AXIS: 'x', 'y', or 'z'
SLICING_AXIS = "z"
# SLICING_DIRECTION: '+' (for min to max) or '-' (for max to min)
SLICING_DIRECTION = "-"

# 4. SET THE MANUAL WEIGHTS
# The length of this list MUST be equal to N_SLICES
# These weights are defined from MIN to MAX along your chosen axis.
# If you set SLICING_DIRECTION to '-', the list will be reversed automatically.
MANUAL_WEIGHTS = [
    8.0,  # Slice 1 (at min 'x')
    7.39,  # Slice 2
    6.78,  # Slice 3
    2.07,  # Slice 4
    1.2,  # Slice 5
    1.2,  # Slice 6
    1.1,  # Slice 7
    0.75,  # Slice 8
    0.75,  # Slice 9
    0.9,  # Slice 10 (Middle)
    1.05,  # Slice 11 (Middle)
    1.7,  # Slice 12
    2.0,  # Slice 13
    2.3,  # Slice 14
    3.0,  # Slice 15
    3.5,  # Slice 16
    3.7,  # Slice 17
    4.8,  # Slice 18
    6.25,  # Slice 19
    8.0,  # Slice 20 (at max 'x')
]

# 5. SET THE NUMBER OF POINTS TO SAMPLE
N_POINTS_TO_SAMPLE = 50000
# ----- Configuration End -----
realistic_factor = 1  # Adjust this factor based on CFD of Silvretta


def approximate_drag_with_sampled_points(
    filename,
    num_slices,
    manual_weights_list,
    num_points,
    slicing_axis,
    slicing_direction,
):
    """
    Loads an STL, samples a dense point cloud from its surface,
    groups these POINTS into N slices along the chosen axis,
    and calculates a normalized, weighted sum of their frontal areas.
    """

    # --- 1. Set up axis indices based on user config ---
    slicing_axis = slicing_axis.lower()
    if slicing_axis == "x":
        axis_idx = 0  # Slicing axis
        area_idx_1 = 1  # Frontal area plane (Y)
        area_idx_2 = 2  # Frontal area plane (Z)
        axis_name = "X-axis"
    elif slicing_axis == "y":
        axis_idx = 1  # Slicing axis
        area_idx_1 = 0  # Frontal area plane (X)
        area_idx_2 = 2  # Frontal area plane (Z)
        axis_name = "Y-axis"
    elif slicing_axis == "z":
        axis_idx = 2  # Slicing axis
        area_idx_1 = 0  # Frontal area plane (X)
        area_idx_2 = 1  # Frontal area plane (Y)
        axis_name = "Z-axis"
    else:
        print(
            f"FATAL ERROR: Invalid SLICING_AXIS '{slicing_axis}'. Must be 'x', 'y', or 'z'."
        )
        return

    # --- 2. Load Mesh and Sample Point Cloud ---
    print(f"Loading mesh: {filename}")
    if not os.path.exists(filename):
        print(f"FATAL ERROR: File not found at '{filename}'")
        return

    try:
        mesh = trimesh.load_mesh(filename, process=False)
        if mesh.vertices.shape[0] < 4:
            print("Error: Mesh has too few vertices to process.")
            return

        print(f"Original mesh has {len(mesh.vertices)} vertices.")

        print(f"Sampling {num_points} points from the mesh surface...")
        np.random.seed(42)  # for reproducible sampling
        vertices = mesh.sample(num_points)

        if vertices.shape[0] == 0:
            print("Error: Failed to sample points from mesh.")
            return

    except Exception as e:
        print(f"Error loading and sampling mesh: {e}")
        return

    print(f"Using {len(vertices)} sampled points for analysis.")

    # --- 3. Slicing Logic (based on axis and direction) ---
    scene = trimesh.Scene()
    slice_frontal_areas = []  # List to store each slice's area

    # Get min/max of the chosen slicing axis
    min_a, max_a = vertices[:, axis_idx].min(), vertices[:, axis_idx].max()
    # Create split points
    split_points = np.linspace(min_a, max_a, num_slices + 1)

    direction_str = " (min to max)"
    # Reverse the split points if direction is '-'
    if slicing_direction == "-":
        split_points = split_points[::-1]
        direction_str = " (max to min)"

    print(
        f"Slicing points into {num_slices} parts along {axis_name} from {split_points[0]:.2f} to {split_points[-1]:.2f}{direction_str}"
    )

    # Define alternating colors for visualization
    FIRST_SLICE_COLOR = [0, 255, 0, 255]  # Green (for slice i=0)
    alternating_colors = [[255, 0, 0, 255], [0, 0, 255, 255]]  # Red, Blue
    n_colors = len(alternating_colors)

    print("\n--- Analyzing Slices ---")

    for i in range(num_slices):
        a_start = split_points[i]  # Start of slice
        a_end = split_points[i + 1]  # End of slice

        # --- MODIFIED: Handle both directions ---
        if slicing_direction == "+":
            # Slicing min to max (a_start < a_end)
            if i == (num_slices - 1):  # Handle last slice
                v_mask = (vertices[:, axis_idx] >= a_start) & (
                    vertices[:, axis_idx] <= a_end
                )
            else:
                v_mask = (vertices[:, axis_idx] >= a_start) & (
                    vertices[:, axis_idx] < a_end
                )
        # Slicing max to min (a_start > a_end)
        elif i == (num_slices - 1):  # Handle last slice
            v_mask = (vertices[:, axis_idx] <= a_start) & (
                vertices[:, axis_idx] >= a_end
            )
        else:
            v_mask = (vertices[:, axis_idx] <= a_start) & (
                vertices[:, axis_idx] > a_end
            )
        # --- END MODIFIED ---

        slice_points = vertices[v_mask]

        if slice_points.shape[0] == 0:
            print(f"Slice {i + 1}: Found 0 points. Area = 0.0000")
            slice_frontal_areas.append(0.0)
            continue

        # --- 4. Calculate Bounding Box and Area ---
        try:
            # Get bounds for the two area-defining axes
            min_a1, max_a1 = (
                slice_points[:, area_idx_1].min(),
                slice_points[:, area_idx_1].max(),
            )
            min_a2, max_a2 = (
                slice_points[:, area_idx_2].min(),
                slice_points[:, area_idx_2].max(),
            )

            # Calculate lengths
            L_slice_depth = abs(a_start - a_end)  # Depth along slicing axis
            L_area_1 = max_a1 - min_a1  # Length 1 of frontal area
            L_area_2 = max_a2 - min_a2  # Length 2 of frontal area

            if L_area_1 < 1e-6:
                L_area_1 = 0.0
            if L_area_2 < 1e-6:
                L_area_2 = 0.0

            frontal_area = L_area_1 * L_area_2
            slice_frontal_areas.append(frontal_area)

            print(
                f"Slice {i + 1}: Found {slice_points.shape[0]:<5} points. Dims (Depth, Area_1, Area_2) = [{L_slice_depth:6.2f}, {L_area_1:6.2f}, {L_area_2:6.2f}], Frontal Area = {frontal_area:8.4f}"
            )

            # --- 5. Build Wireframe Visualization ---
            # Re-assemble box dims and center in X,Y,Z order for trimesh
            box_extents = [0, 0, 0]
            box_center = [0, 0, 0]

            # Slicing axis dimension
            box_extents[axis_idx] = L_slice_depth
            box_center[axis_idx] = (a_start + a_end) / 2

            # First area dimension
            box_extents[area_idx_1] = L_area_1
            box_center[area_idx_1] = (min_a1 + max_a1) / 2

            # Second area dimension
            box_extents[area_idx_2] = L_area_2
            box_center[area_idx_2] = (min_a2 + max_a2) / 2
            # --- End re-assembly ---

            box_transform = trimesh.transformations.translation_matrix(box_center)
            box_wireframe = trimesh.path.creation.box_outline(
                extents=box_extents, transform=box_transform
            )

            # Color the first box (i=0) green
            if i == 0:
                current_color = FIRST_SLICE_COLOR
            else:
                color_index = (i - 1) % n_colors
                current_color = alternating_colors[color_index]

            box_wireframe.colors = [current_color] * len(box_wireframe.entities)
            scene.add_geometry(box_wireframe, geom_name=f"box_wireframe_{i + 1}")

        except Exception as e:
            print(f"Error processing slice {i + 1}: {e}")
            slice_frontal_areas.append(0.0)

    # --- 6. Final Output ---

    # --- CALCULATE YOUR "MANUAL" WEIGHTED LOSS ---

    if len(manual_weights_list) != num_slices:
        print(
            f"FATAL ERROR: MANUAL_WEIGHTS list has {len(manual_weights_list)} items, but N_SLICES is {num_slices}."
        )
        return

    # Note: manual_weights_list is already reversed if direction='-'
    weights = manual_weights_list

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
    sorted_areas = np.sort(slice_frontal_areas)[::-1]  # Sort descending

    if num_slices < 5:
        print(
            f"Warning: N_SLICES is {num_slices} (less than 5). Using A_max for normalization."
        )
        A_norm = sorted_areas[0] if len(sorted_areas) > 0 else 0.0
        norm_factor_name = "A_max"
    else:
        A_norm = np.mean(sorted_areas[3:6])  # 5th biggest area
        norm_factor_name = "A_5th_biggest"

    if A_norm > 1e-6 and num_slices > 0:
        normalized_loss = (
            final_weighted_loss
            * sorted_areas[0]
            / (A_norm * A_norm * num_slices * realistic_factor)
        )

        print(
            f"Normalization Factors: {norm_factor_name} = {A_norm:8.4f}, N_Slices = {num_slices}, Realistic_Factor = {realistic_factor}"
        )
        print(
            f"Normalized Loss (Loss / ({norm_factor_name} * N_Slices * Factor)): {normalized_loss:8.4f}"
        )
        print("========================")
    else:
        print("Cannot normalize: 5th biggest area or N_Slices is zero.")

    print("------------------------")

    # Add the original mesh vertices as a point cloud for context
    point_cloud = trimesh.points.PointCloud(vertices)
    point_cloud.colors = [128, 128, 128, 50]  # Semi-transparent gray
    scene.add_geometry(point_cloud, geom_name="point_cloud")

    # --- Export scene ---
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
    weights_to_use = MANUAL_WEIGHTS

    # Automatically reverse the weights list if slicing from max to min
    if SLICING_DIRECTION == "+":
        print(f"--- REVERSING SLICE DIRECTION (Max to Min on {SLICING_AXIS}-axis) ---")
        weights_to_use = MANUAL_WEIGHTS[::-1]
    else:
        print(f"--- SLICING (Min to Max on {SLICING_AXIS}-axis) ---")

    # Pass the configured axis, direction, and correct weights list
    approximate_drag_with_sampled_points(
        FILE_TO_LOAD,
        N_SLICES,
        weights_to_use,  # Pass the (potentially) reversed list
        N_POINTS_TO_SAMPLE,
        SLICING_AXIS,  # Pass the new config
        SLICING_DIRECTION,  # Pass the new config
    )
