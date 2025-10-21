import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt

# --- 1. Read the PLY file ---
try:
    plydata = PlyData.read("sample.ply")
    print("PLY file read successfully.")
except FileNotFoundError:
    print(
        "Error: 'sample.ply' not found. Make sure you created it in the same directory."
    )
    exit()

# --- 2. Extract and Inspect the Data ---
# This is the key part for your AI application!
# 'plydata.elements' is a list of elements. We access them by name.

# Get the 'vertex' element
vertices = plydata["vertex"].data
print("\n--- Vertex Data ---")
print(vertices)
print(f"Shape of vertex data: {vertices.shape}")
print(f"Data type (dtype) of vertices: {vertices.dtype}")

# Get the 'face' element
faces = plydata["face"].data
print("\n--- Face Data (Vertex Indices) ---")
print(faces)
print(f"Shape of face data: {faces.shape}")
print(f"Data type (dtype) of faces: {faces.dtype}")

# --- 3. Prepare Data for Plotting ---
# We can access properties by name, just like a dictionary
x = vertices["x"]
y = vertices["y"]
z = vertices["z"]

# Get the colors. Matplotlib needs them as 0-1 floats, not 0-255 integers.
colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
print(f"\nNormalized colors for plotting:\n{colors}")

# --- 4. Visualize the Point Cloud ---
print("\nVisualizing point cloud with matplotlib...")
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the vertices as a scatter plot
ax.scatter(x, y, z, c=colors, s=100)  # 's' is marker size

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("PLY File Vertex Cloud")
plt.show()

print(
    "\nNote: Matplotlib is great for point clouds, but visualizing the full *mesh* (the faces) is complex."
)
print("See Example 2 for a much better mesh visualizer.")
