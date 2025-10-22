import open3d as o3d

# --- Settings ---
FILE_NAME = r"C:\Users\elias\Documents\AISCO\AISCO-code\subprojects\WP4-Aeroshell\SDF_Prototype\main_models\sdf_sphere_mesh.ply"  # Replace with your .ply file

# --- 1. Load the mesh ---
try:
    mesh = o3d.io.read_triangle_mesh(FILE_NAME)
    if mesh.is_empty():
        raise FileNotFoundError(f"{FILE_NAME} not found or is empty.")
except Exception as e:
    print(f"Warning: {e}. Using test data as a fallback...")
    mesh = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(mesh.path)

# -----------------------------------------------------------------
# Method 1: Draw ONLY the Vertices (as a Point Cloud)
# -----------------------------------------------------------------
print("Showing vertices as a point cloud...")

# Create a new PointCloud object
pcd = o3d.geometry.PointCloud()

# Copy the vertices from the mesh into the point cloud
pcd.points = mesh.vertices

# Copy the colors from the mesh, if they exist
if mesh.has_vertex_colors():
    pcd.colors = mesh.vertex_colors

# Optional: If the mesh has no colors, paint them green
else:
    pcd.paint_uniform_color([0.1, 0.9, 0.1])  # Bright green

# # Visualize the point cloud.
# # Press '+' and '-' to change the point size.
# o3d.visualization.draw_geometries([pcd])


# -----------------------------------------------------------------
# Method 2: Draw the Mesh as a Wireframe
# -----------------------------------------------------------------
print("Showing mesh as a wireframe...")

# Create a LineSet from the mesh to get a wireframe
wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)

# Color the wireframe
wireframe.paint_uniform_color([0.5, 0.5, 0.5])  # Gray

# Visualize the wireframe
o3d.visualization.draw_geometries([pcd])
