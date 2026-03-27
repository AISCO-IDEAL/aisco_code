"""Quick PyVista visualization of a BlendedNet VTK file."""

import sys

import pyvista as pv

path = sys.argv[1] if len(sys.argv) > 1 else "/Users/taebersold/Downloads/blended_dataset/train/vtk/case_0000.vtk"

mesh = pv.read(path)

p = pv.Plotter(shape=(1, 2), window_size=(1600, 600))

# Left: Cp field
p.subplot(0, 0)
p.add_mesh(mesh, scalars="cp", cmap="RdBu_r", clim=[-1.5, 1.0], show_edges=False)
p.add_title("Cp")
p.view_xy()

# Right: plain geometry
p.subplot(0, 1)
p.add_mesh(mesh, color="lightgrey", show_edges=True, edge_color="black", opacity=1.0, line_width=0.3)
p.add_title("Mesh")
p.view_xy()

p.link_views()
p.show()
