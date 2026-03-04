import trimesh

# --- Configuration ---
INPUT_FILE = r"C:\Users\elias\Documents\AISCO\AISCO-code\subprojects\wp1_cfd_ai\data\crude_estimator\test_output_diff_mesh.ply"
OUTPUT_FILE = r"C:\Users\elias\Documents\AISCO\AISCO-code\subprojects\wp1_cfd_ai\data\crude_estimator\scaled_up_mesh.ply"
SCALE_FACTOR = 10000.0  # e.g., to go from meters (1.0) to millimeters (1000.0)

# 1. Load the mesh
try:
    mesh = trimesh.load_mesh(INPUT_FILE)
    print(f"Original bounds: {mesh.bounds}")
    print(f"Original extents: {mesh.extents}")

    # 2. Apply the scaling
    # This multiplies all vertex coordinates by the scale factor
    mesh.apply_scale(SCALE_FACTOR)

    # 3. Export the new, scaled mesh
    mesh.export(OUTPUT_FILE)

    print("-" * 30)
    print(f"Successfully saved scaled mesh to: {OUTPUT_FILE}")
    print(f"New bounds: {mesh.bounds}")
    print(f"New extents: {mesh.extents}")

except Exception as e:
    print(f"Error: {e}")
