"""
In this file the CadQueryToSDF class from CadQuery_to_SDF.py is tested with a simple shapes.
Run this file by running the two follwoing commands in the terminal:

cd /cluster/home/clhuber/AISCO-code

python -m subprojects.wp4_aeroshell_optimizer.src.test.Test_for_CadQuery_to_SDF
"""
# one point if in the same folder, more if outwards
# file without .py ending
#
# cd /cluster/home/clhuber/AISCO-code
# python -m subprojects.wp4_aeroshell_optimizer.src.test.Test_for_CadQuery_to_SDF
# 

from typing import cast

import cadquery as cq
import matplotlib.pyplot as plt
import numpy as np

from ....wp2_assembler.version1.components.nbz import innerNBZ, outerNBZ
from ..utils.CadQuery_to_SDF import SDFfromCadQuery

sdf_converter: SDFfromCadQuery
# test with a Cylinder
if False:
    cylinder = cq.Workplane("XY").circle(0.5).extrude(1).objects[0]
    cylinder = cast(cq.Shape, cylinder)
    # print(type(cylinder))
    sdf_converter = SDFfromCadQuery(cylinder)

# test with a cube
if False:
    cube = cq.Workplane("XY").box(0.75, 0.75, 0.75).objects[0]
    cube = cast(cq.Shape, cube)
    sdf_converter = SDFfromCadQuery(cube)

# use the innerNBZ from wp2_assembler
if True:
    inner_nbz_shape = innerNBZ(
        random_magnitude_chassis=0.0,
        random_magnitude_seat=0.0,
        random_magnitude_wheels=0.0,
    )
    sdf_converter = SDFfromCadQuery(inner_nbz_shape)

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
    plt.colorbar(label='SDF Value')
    plt.title(f'SDF values at z={z}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    # save the figure at a certain path
    plt.savefig(f'/cluster/home/clhuber/immages/sdf_layer_z_{z:.2f}.png')
    plt.close()
    
    # print the value at the center
    center_mask = (np.isclose(x_vals, 0.0)) & (np.isclose(y_vals, 0.0))
    if np.any(center_mask):
        center_sdf_value = sdf_vals_layer[center_mask][0]
        print(f'SDF value at center (0,0,{z}): {center_sdf_value}')
        print(f'SDF value at center (0,0,{z}): {center_sdf_value}')
