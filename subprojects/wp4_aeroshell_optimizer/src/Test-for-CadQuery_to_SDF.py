"""
In this file the CadQueryToSDF class from CadQuery_to_SDF.py is tested with a simple shapes.
"""

from typing import cast

import cadquery as cq
import matplotlib.pyplot as plt
import numpy as np
from CadQuery_to_SDF import SDFfromCadQuery

# test with a Cylinder
if False:
    cylinder = cq.Workplane("XY").circle(0.5).extrude(1).objects[0]
    cylinder = cast(cq.Shape, cylinder)
    print(type(cylinder))
    sdf_converter = SDFfromCadQuery(cylinder)

# test with a cube
if False:
    cube = cq.Workplane("XY").box(0.75, 0.75, 0.75).objects[0]
    cube = cast(cq.Shape, cube)
    sdf_converter = SDFfromCadQuery(cube)


sdf_valus = sdf_converter.sdf_at_pointgrid()

# sort them for their z values and plot each layer
pointgrid = sdf_converter.pointgrid()
pointgrid_np = np.array(pointgrid)
sdf_values_np = np.array(sdf_valus)
z_values = np.unique(pointgrid_np[:, 2])
for z in z_values:
    mask = pointgrid_np[:, 2] == z
    x_vals = pointgrid_np[mask][:, 0]
    y_vals = pointgrid_np[mask][:, 1]
    sdf_vals_layer = sdf_values_np[mask]

    # create a imshow plot
    plt.figure()
    plt.tricontourf(x_vals, y_vals, sdf_vals_layer, levels=14, cmap="RdBu_r")
    plt.colorbar(label="SDF Value")
    plt.title(f"SDF values at z={z}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()
