import trimesh

# --- 1. SET YOUR FILE PATH HERE ---
STL_FILE_PATH = r"C:\Users\elias\Documents\AISCO\AISCO-code\subprojects\wp1_cfd_ai\data\crude_estimator\sdf_sphere_mesh.ply"

try:
    # 2. Load the mesh
    mesh = trimesh.load_mesh(STL_FILE_PATH)

    # 3. Get the bounding box
    # 'bounds' is a [2, 3] numpy array:
    # [[min_x, min_y, min_z],
    #  [max_x, max_y, max_z]]
    bounds = mesh.bounds

    # 4. Print the results
    print(f"--- Bounding Box for: {STL_FILE_PATH} ---")

    print(f"  X-Axis: {bounds[0, 0]:.2f} to {bounds[1, 0]:.2f}")
    print(f"  Y-Axis: {bounds[0, 1]:.2f} to {bounds[1, 1]:.2f}")
    print(f"  Z-Axis: {bounds[0, 2]:.2f} to {bounds[1, 2]:.2f}")

    # 5. Print the overall size (extents)
    # 'extents' is the total length, width, and height
    extents = mesh.extents
    print("\n--- Overall Size (Extents) ---")
    print(f"  Length (X): {extents[0]:.2f}")
    print(f"  Width  (Y): {extents[1]:.2f}")
    print(f"  Height (Z): {extents[2]:.2f}")

except Exception as e:
    print(f"Error loading mesh: {e}")
