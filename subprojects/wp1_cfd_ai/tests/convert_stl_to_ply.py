import open3d as o3d  # type : ignore [import-untyped]

# 1. Read the STL file
mesh = o3d.io.read_triangle_mesh("Jass Box - Lid.stl")

# 2. Write the PLY file
o3d.io.write_triangle_mesh("output_model.ply", mesh)

print("Conversion complete!")
