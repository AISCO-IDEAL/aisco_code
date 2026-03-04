# ruff: noqa: PLR0915
# because we are using this file not anymore anyway
import os

import numpy as np
import trimesh


def calculateCdProxy(
    path_to_file: str,
    create_visualization: bool,
    visualization_path: str,
    slicing_axis: str = "x",
    slicing_direction: str = "-",
    weights: list[float] | None = None,
) -> float | None:
    """
    Calculates a drag coefficient (Cd) proxy for a given STL file.

    This function loads an STL, samples a dense point cloud, slices it
    into sections along a predefined axis, calculates the weighted frontal
    area of each slice, and returns a normalized "drag loss" value.

    The number of slices (n_slices) and sampling density are defined as
    internal constants.

    Args:
        path_to_file: The absolute or relative path to the .stl file.
        create_visualization: If True, a .glb file visualizing the
            slicing boxes will be generated.
        visualization_path: The full path (including filename, e.g.,
            'output/viz.glb') where the visualization file should be saved.
        slicing_axis: The axis to slice along ('x', 'y', or 'z').
            Defaults to 'x'.
        slicing_direction: The direction to slice ('+' for min to max,
            '-' for max to min). Defaults to '-'.
        weights: A list of floats to use as weights for each slice.
            The list length must equal n_slices (20). If None,
            default internal weights are used. Defaults to None.

    Returns:
        The calculated normalized loss (Cd proxy) as a float, or None
        if the calculation fails at any step.
    """

    # --- Internal Configuration ---
    # These parameters are fixed for this function.
    n_slices = 20
    default_weights = [
        8.0,
        7.39,
        6.78,
        2.07,
        1.2,
        1.2,
        1.1,
        0.75,
        0.75,
        0.9,
        1.05,
        1.7,
        2.0,
        2.3,
        3.0,
        3.5,
        3.7,
        4.8,
        6.25,
        8.0,
    ]  # Defined from MIN to MAX axis
    n_points_to_sample = 50000
    realistic_factor = 1.0
    # --- End Internal Configuration ---

    # --- Handle default weights ---
    manual_weights_to_use = weights
    if manual_weights_to_use is None:
        print("--- Using default weights. ---")
        manual_weights_to_use = default_weights
    else:
        print(f"--- Using provided custom weights ({len(manual_weights_to_use)} items). ---")

    # --- Handle weight direction based on config ---
    # The manual_weights_to_use are assumed to be defined from MIN to MAX.
    # If we slice from MAX to MIN ('-'), we must reverse the weights list.
    weights_to_use = manual_weights_to_use
    if slicing_direction == "-":
        print(f"--- Slicing direction is 'Max to Min' ({slicing_axis}-axis). Reversing weights list. ---")
        weights_to_use = manual_weights_to_use[::-1]
    else:
        print(f"--- Slicing direction is 'Min to Max' ({slicing_axis}-axis). Using weights as defined. ---")

    # --- 1. Set up axis indices based on config ---
    slicing_axis_lower = slicing_axis.lower()
    if slicing_axis_lower == "x":
        axis_idx = 0
        area_idx_1 = 1
        area_idx_2 = 2
        axis_name = "X-axis"
    elif slicing_axis_lower == "y":
        axis_idx = 1
        area_idx_1 = 0
        area_idx_2 = 2
        axis_name = "Y-axis"
    elif slicing_axis_lower == "z":
        axis_idx = 2
        area_idx_1 = 0
        area_idx_2 = 1
        axis_name = "Z-axis"
    else:
        print(f"FATAL ERROR: Slicing axis '{slicing_axis}' is invalid. Must be 'x', 'y', or 'z'.")
        return None

    # --- 2. Load Mesh and Sample Point Cloud ---
    print(f"Loading mesh: {path_to_file}")
    if not os.path.exists(path_to_file):
        print(f"FATAL ERROR: File not found at '{path_to_file}'")
        return None

    try:
        mesh = trimesh.load_mesh(path_to_file, process=False)
        if mesh.vertices.shape[0] < 4:
            print("Error: Mesh has too few vertices to process.")
            return None

        print(f"Original mesh has {len(mesh.vertices)} vertices.")

        print(f"Sampling {n_points_to_sample} points from the mesh surface...")
        np.random.seed(42)  # for reproducible sampling
        vertices = mesh.sample(n_points_to_sample)

        if vertices.shape[0] == 0:
            print("Error: Failed to sample points from mesh.")
            return None

    except Exception as e:
        print(f"Error loading and sampling mesh: {e}")
        return None

    print(f"Using {len(vertices)} sampled points for analysis.")

    # --- 3. Slicing Logic ---
    scene = None
    if create_visualization:
        scene = trimesh.Scene()

    slice_frontal_areas = []  # List to store each slice's area

    min_a, max_a = vertices[:, axis_idx].min(), vertices[:, axis_idx].max()
    split_points = np.linspace(min_a, max_a, n_slices + 1)

    direction_str = " (min to max)"
    if slicing_direction == "-":
        split_points = split_points[::-1]
        direction_str = " (max to min)"

    print(
        f"Slicing points into {n_slices} parts along {axis_name} from {split_points[0]:.2f} to {split_points[-1]:.2f}{direction_str}"
    )

    first_slice_color = [0, 255, 0, 255]
    alternating_colors = [[255, 0, 0, 255], [0, 0, 255, 255]]
    n_colors = len(alternating_colors)

    print("\n--- Analyzing Slices ---")
    normalized_loss = None  # Initialize return value

    for i in range(n_slices):
        a_start = split_points[i]
        a_end = split_points[i + 1]

        if slicing_direction == "+":
            if i == (n_slices - 1):
                v_mask = (vertices[:, axis_idx] >= a_start) & (vertices[:, axis_idx] <= a_end)
            else:
                v_mask = (vertices[:, axis_idx] >= a_start) & (vertices[:, axis_idx] < a_end)
        elif i == (n_slices - 1):
            v_mask = (vertices[:, axis_idx] <= a_start) & (vertices[:, axis_idx] >= a_end)
        else:
            v_mask = (vertices[:, axis_idx] <= a_start) & (vertices[:, axis_idx] > a_end)

        slice_points = vertices[v_mask]

        if slice_points.shape[0] == 0:
            print(f"Slice {i + 1}: Found 0 points. Area = 0.0000")
            slice_frontal_areas.append(0.0)
            continue

        # --- 4. Calculate Bounding Box and Area ---
        try:
            min_a1, max_a1 = (
                slice_points[:, area_idx_1].min(),
                slice_points[:, area_idx_1].max(),
            )
            min_a2, max_a2 = (
                slice_points[:, area_idx_2].min(),
                slice_points[:, area_idx_2].max(),
            )

            l_slice_depth = abs(a_start - a_end)
            L_area_1 = max_a1 - min_a1
            l_area_2 = max_a2 - min_a2

            if L_area_1 < 1e-6:
                L_area_1 = 0.0
            if l_area_2 < 1e-6:
                l_area_2 = 0.0

            frontal_area = L_area_1 * l_area_2
            slice_frontal_areas.append(frontal_area)

            print(
                f"Slice {i + 1}: Found {slice_points.shape[0]:<5} points. Dims (Depth, Area_1, Area_2) = [{l_slice_depth:6.2f}, {L_area_1:6.2f}, {l_area_2:6.2f}], Frontal Area = {frontal_area:8.4f}"
            )

            # --- 5. Build Wireframe Visualization (if requested) ---
            if create_visualization and scene is not None:
                box_extents = [0, 0, 0]
                box_center = [0, 0, 0]

                box_extents[axis_idx] = l_slice_depth
                box_center[axis_idx] = (a_start + a_end) / 2
                box_extents[area_idx_1] = L_area_1
                box_center[area_idx_1] = (min_a1 + max_a1) / 2
                box_extents[area_idx_2] = l_area_2
                box_center[area_idx_2] = (min_a2 + max_a2) / 2

                box_transform = trimesh.transformations.translation_matrix(box_center)
                box_wireframe = trimesh.path.creation.box_outline(extents=box_extents, transform=box_transform)

                if i == 0:
                    current_color = first_slice_color
                else:
                    color_index = (i - 1) % n_colors
                    current_color = alternating_colors[color_index]

                box_wireframe.colors = [current_color] * len(box_wireframe.entities)
                scene.add_geometry(box_wireframe, geom_name=f"box_wireframe_{i + 1}")

        except Exception as e:
            print(f"Error processing slice {i + 1}: {e}")
            slice_frontal_areas.append(0.0)

    # --- 6. Final Calculation ---
    if len(weights_to_use) != n_slices:
        print(f"FATAL ERROR: Weights list has {len(weights_to_use)} items, but n_slices is {n_slices}.")
        return None

    final_weighted_loss = 0.0
    print("------------------------")
    print("Calculating Weighted Loss:")
    for i in range(n_slices):
        area = slice_frontal_areas[i]
        weight = weights_to_use[i]
        weighted_area = area * weight
        final_weighted_loss += weighted_area
        print(f"Slice {i + 1}: Area = {area:8.4f} * Weight = {weight:4.2f} -> {weighted_area:8.4f}")

    print("========================")
    print(f"Final Weighted Loss: {final_weighted_loss:8.4f}")
    print("========================")

    # --- NORMALIZE THE LOSS ---
    sorted_areas = np.sort(slice_frontal_areas)[::-1]  # Sort descending

    if n_slices < 5:
        print(f"Warning: n_slices is {n_slices} (less than 5). Using A_max for normalization.")
        a_norm = sorted_areas[0] if len(sorted_areas) > 0 else 0.0
        norm_factor_name = "A_max"
    else:
        a_norm = np.mean(sorted_areas[3:6])  # 5th biggest area (avg of 4th, 5th, 6th)
        norm_factor_name = "A_5th_biggest"

    if a_norm > 1e-6 and n_slices > 0 and len(sorted_areas) > 0:
        normalized_loss = (
            final_weighted_loss
            * sorted_areas[0]  # A_max
            / n_slices
            * realistic_factor
        )

        print(
            f"Normalization Factors: {norm_factor_name} = {a_norm:8.4f}, n_slices = {n_slices}, Realistic_Factor = {realistic_factor}"
        )
        print(f"Normalized Loss (Cd proxy): {normalized_loss:8.4f}")
        print("========================")
    else:
        print("Cannot normalize: 5th biggest area, A_max, or n_slices is zero.")
        normalized_loss = 0.0

    print("------------------------")

    # --- 7. Export Visualization (if requested) ---
    if create_visualization and scene is not None:
        point_cloud = trimesh.points.PointCloud(vertices)
        point_cloud.colors = [128, 128, 128, 50]  # Semi-transparent gray
        scene.add_geometry(point_cloud, geom_name="point_cloud")

        try:
            # Ensure the output directory exists
            vis_dir = os.path.dirname(visualization_path)
            if vis_dir and not os.path.exists(vis_dir):
                os.makedirs(vis_dir, exist_ok=True)
                print(f"Created visualization directory: {vis_dir}")

            scene.export(visualization_path)
            print(f"\nSuccessfully saved visualization to {visualization_path}")
            print(f"You can open this .glb file in any 3D viewer to see the {n_slices} boxes.")
        except Exception as e:
            print(f"\nError exporting scene to {visualization_path}: {e}")
    elif create_visualization:
        print("\nWarning: Visualization was requested but the scene is empty.")
    else:
        print("\nVisualization not requested. Skipping export.")

    return normalized_loss


# --- Example Usage ---
if __name__ == "__main__":
    # 1. SET THE PATH TO YOUR TEST STL FILE
    #    (Please update this path to a valid .stl file on your system)
    TEST_FILE = r"C:\Users\elias\Documents\AISCO\AISCO-code\AISCO\subprojects\WP1_CFD_AI\crude_estimator\silvretta_big_rot.stl"

    # 2. SET VISUALIZATION PREFERENCES
    CREATE_VIZ = True
    VIZ_OUTPUT_PATH = "my_model_visualization.glb"

    print(f"--- Running Drag Approximation for: {TEST_FILE} ---")

    # Check if the test file exists before running
    if os.path.exists(TEST_FILE):
        # --- Standard call using default axis, direction, and weights ---
        print("\n--- RUNNING DEFAULT CALCULATION (X-axis, Max-to-Min) ---")
        cd_value = calculateCdProxy(
            path_to_file=TEST_FILE,
            create_visualization=CREATE_VIZ,
            visualization_path=VIZ_OUTPUT_PATH,
        )

        if cd_value is not None:
            print("\n--- SCRIPT FINISHED (DEFAULT) ---")
            print(f"Approximate Cd Value: {cd_value:.6f}")
        else:
            print("\n--- SCRIPT FAILED (DEFAULT) ---")
            print("Drag calculation returned None.")

    else:
        print("--- TEST SKIPPED ---")
        print(f"Test file not found at: {TEST_FILE}")
        print("Please update the 'TEST_FILE' variable in the '__main__' block to run the example.")
