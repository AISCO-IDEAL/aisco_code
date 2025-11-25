import pyvista as pv
import os
from subprojects.wp4_aeroshell_optimizer.src.utils.utils import load_config
from suprojects.wp2_assembler.version1.components.nbz import innerNBZ, outerNBZ

"""
Takes aeroshell and inner- and outer-no-build-zones and returns a gif that circles around the combined assembly 
Args:
    inner_nbz: sq.solids
    outer_nbz: sq.solids
    aeroshell: .ply

Output:
    visualized: .gif, 
"""

username = load_config().get("username")

# Check import paths
aeroshell_path = f"/cluster/home/{username}/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Meshes/mesh1.ply"
if not os.path.exists(aeroshell_path):
    raise ValueError(f"The file {aeroshell_path} does not exist or is not accessible.")

print("type innerNBZ", type(innerNBZ))
# # Load your meshes
# inner_nbz = pv.read(inner_nbz_path)
# aeroshell = pv.read(aeroshell_path)
# outer_nbz = pv.read(inner_nbz_path)

# # Create a plotter and add each mesh with a distinct color
# plotter = pv.Plotter()
# plotter.add_mesh(inner_nbz, style="wireframe", color="green")
# plotter.add_mesh(aeroshell, style="wireframe", color="blue")
# plotter.add_mesh(outer_nbz, style="wireframe", color="red")

# # Set camera and background
# plotter.camera.elevation = 30  # Fixed downward tilt
# plotter.set_background("white", top="white")  # Or use transparent_background=True

# # Orbit and save GIF
# path = plotter.generate_orbital_path(n_points=36)
# plotter.open_gif("output.gif")
# plotter.orbit_on_path(path, write_frames=True)
# plotter.close()
