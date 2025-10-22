import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d  # type: ignore

# --- 1. Load your meshes ---
# (Using the same loading logic as Method 1)
try:
    ply_mesh = o3d.io.read_triangle_mesh("Jass Box - Lid.ply")
    stl_mesh = o3d.io.read_triangle_mesh("Jass Box - Lid.stl")
except Exception as e:
    print(f"Error: {e}. Make sure you have valid mesh files.")
    print("Using test data as a fallback...")
    ply_mesh = o3d.data.BunnyMesh()
    ply_mesh = o3d.io.read_triangle_mesh(ply_mesh.path)
    stl_mesh = o3d.data.ArmadilloMesh()
    stl_mesh = o3d.io.read_triangle_mesh(stl_mesh.path)
    stl_mesh.translate((0.1, 0, 0))

# --- 2. Compute Distances ---
# We will check the distance from the STL mesh to the PLY mesh.
pcd = stl_mesh.sample_points_uniformly(number_of_points=20000)

# Convert mesh to the modern 't.geometry' format for the RaycastingScene
ply_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(ply_mesh)

# Create a scene and add the mesh to it
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(ply_mesh_t)

# Convert to a NumPy array first (this is more robust)
points_as_numpy = np.asarray(pcd.points)

# Now, create the Tensor from the NumPy array
query_points = o3d.core.Tensor(points_as_numpy, dtype=o3d.core.Dtype.Float32)

# Compute the distance from the points to the mesh surface
distances = scene.compute_distance(query_points).numpy()

# --- 3. Create a Heat Map ---
# Use a color map (like 'jet' or 'viridis') to color the points
# 'jet' goes from blue (low) to red (high)
cmap = plt.get_cmap("jet")

# Normalize distances to be between 0 and 1 for the color map
norm_distances = (distances - distances.min()) / (distances.max() - distances.min())
colors = cmap(norm_distances)[:, :3]  # Get RGB values from the colormap

# --- 4. Assign Colors to the Point Cloud ---
pcd.colors = o3d.utility.Vector3dVector(colors)

# --- 5. Visualize the Heat Map ---
# This shows the STL model (as a point cloud) colored by
# its deviation from the PLY model.
print("\nShowing deviation heat map. Close the window to exit.")
o3d.visualization.draw_geometries([pcd])
