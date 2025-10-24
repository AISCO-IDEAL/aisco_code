import copy

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def load_mesh(file_path):
    """
    Tries to load a mesh file. Returns None if it fails.
    """
    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
        if mesh.is_empty():
            print(f"Error: Could not read mesh from {file_path}")
            return None
        return mesh
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def show_overlap(file_a, file_b, color_a=(0.8, 0.8, 0.8), color_b=(1.0, 0.1, 0.1)):
    """
    Loads two mesh files and visualizes them overlapping in different colors.

    Args:
        file_a (str): Path to the first mesh file.
        file_b (str): Path to the second mesh file.
        color_a (tuple): RGB color for the first mesh.
        color_b (tuple): RGB color for the second mesh.
    """
    print(f"Loading {file_a} and {file_b} for overlap visualization...")
    mesh_a = load_mesh(file_a)
    mesh_b = load_mesh(file_b)

    if mesh_a is None or mesh_b is None:
        return

    # Create copies for visualization so we don't modify the originals
    mesh_a_viz = copy.deepcopy(mesh_a).paint_uniform_color(color_a)
    mesh_b_viz = copy.deepcopy(mesh_b).paint_uniform_color(color_b)

    print("Showing overlap. Close the window to continue.")
    o3d.visualization.draw_geometries([mesh_a_viz, mesh_b_viz])


def show_heatmap_comparison(source_file, target_file, num_points=5000):
    """
    Computes the distance from the source mesh to the target mesh and
    visualizes the result as a heat map.

    Args:
        source_file (str): Path to the source mesh (the one to be colored).
        target_file (str): Path to the target mesh (the reference).
        num_points (int): Number of points to sample on the source mesh.
    """
    print(f"Loading {source_file} (source) and {target_file} (target) for heatmap...")
    source_mesh = load_mesh(source_file)
    target_mesh = load_mesh(target_file)

    if source_mesh is None or target_mesh is None:
        return

    # --- 1. Sample the source mesh ---
    pcd = source_mesh.sample_points_uniformly(number_of_points=num_points)

    # --- 2. Compute Distances (using the accurate Raycasting method) ---
    print("Computing distances...")
    target_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(target_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(target_mesh_t)

    # Convert points to a Tensor (the robust way)
    points_as_numpy = np.asarray(pcd.points)
    query_points = o3d.core.Tensor(points_as_numpy, dtype=o3d.core.Dtype.Float32)

    distances = scene.compute_distance(query_points).numpy()

    # --- 3. Create a Heat Map ---
    cmap = plt.get_cmap("jet")
    norm_distances = (distances - distances.min()) / (distances.max() - distances.min())
    colors = cmap(norm_distances)[:, :3]  # Get RGB values

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # --- 4. Visualize ---
    print("Showing heatmap. Close the window to continue.")
    o3d.visualization.draw_geometries([pcd])


# --- This block only runs when you execute `python mesh_comparator.py` ---
if __name__ == "__main__":
    print("Running mesh_comparator.py as a standalone script for testing...")

    # Use built-in Open3D data for a simple test
    try:
        bunny_data = o3d.data.BunnyMesh()
        arma_data = o3d.data.ArmadilloMesh()

        file_1 = bunny_data.path
        file_2 = arma_data.path  # These are different sizes, just for a test

        print("\n--- Testing Overlap Function ---")
        show_overlap(file_1, file_2)

        print("\n--- Testing Heatmap Function ---")
        # Show distance from bunny to armadillo
        show_heatmap_comparison(file_1, file_2)

    except Exception as e:
        print(f"Could not load test data. Make sure open3d-data is installed. {e}")
