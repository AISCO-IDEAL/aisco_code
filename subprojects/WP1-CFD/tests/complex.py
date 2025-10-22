import open3d as o3d

# --- 1. Load the PLY file ---
# This single line reads the file, parses vertices, faces, and vertex colors.
try:
    mesh = o3d.io.read_triangle_mesh("sample.ply")
except Exception as e:
    print(f"Error: {e}. Make sure 'sample.ply' exists.")
    exit()

# --- 2. Inspect the Loaded Mesh Object ---
print("--- Open3D Mesh Object ---")
print(mesh)

# You can see it has all the data we care about:
print(f"Mesh has {len(mesh.vertices)} vertices.")
print(f"Mesh has {len(mesh.triangles)} triangles.")
print(f"Mesh has vertex colors: {mesh.has_vertex_colors()}")

# --- 3. Visualize the Mesh ---
# This opens a fully interactive 3D window.
print("\nOpening interactive 3D visualizer...")
o3d.visualization.draw_geometries([mesh])
