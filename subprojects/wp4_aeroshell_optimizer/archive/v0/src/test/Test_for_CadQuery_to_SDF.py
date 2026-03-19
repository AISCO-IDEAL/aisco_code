"""
In this file the CadQueryToSDF class from CadQuery_to_SDF.py is tested with a simple shapes.
Set testcase to one of the following values to test different shapes:
- 1: cylinder
- 2: cube
- 3: innerNBZ from wp2_assembler version1
- 4: outerNBZ from wp2_assembler version1
"""

testcase = 4

from typing import cast

import cadquery as cq
import matplotlib.pyplot as plt
import numpy as np

from ....wp2_assembler.version1.components.nbz import innerNBZ, outerNBZ
from ..utils.CadQuery_to_SDF import SDFfromCadQuery

username = "clhuber"
h = ""  # string to hold csv header info
sdf_converter: SDFfromCadQuery
# test with a Cylinder
if testcase == 1:
    cylinder = cq.Workplane("XY").circle(500).extrude(200).objects[0]
    cylinder = cast(cq.Shape, cylinder)
    # print(type(cylinder))
    sdf_converter = SDFfromCadQuery(cylinder)

# test with a cube
if testcase == 2:
    cube = cq.Workplane("XY").box(0.75, 0.75, 0.75).objects[0]
    cube = cast(cq.Shape, cube)
    sdf_converter = SDFfromCadQuery(cube)

# use the innerNBZ from wp2_assembler
if testcase == 3:
    inner_nbz_shape = innerNBZ(
        random_magnitude_chassis=0.0,
        random_magnitude_seat=0.0,
        random_magnitude_wheels=0.0,
    )
    h += f"Volume of innerNBZ: {inner_nbz_shape.Volume()} mm^3\n"
    sdf_converter = SDFfromCadQuery(inner_nbz_shape)

# use the outerNBZ from wp2_assembler
if testcase == 4:
    outer_nbz_shape = outerNBZ()
    h += f"Volume of outerNBZ: {outer_nbz_shape.Volume()} mm^3\n"
    sdf_converter = SDFfromCadQuery(outer_nbz_shape)

sdf_valus = sdf_converter.sdf_at_pointgrid()

# sort them for their z values and plot each layer
pointgrid = sdf_converter.pointgrid()
pointgrid_np = np.array(pointgrid)
sdf_values_np = np.array(sdf_valus)
z_values = np.unique(pointgrid_np[:, 2])

# picking certain layers to plot
layers_to_plot = [-1000, -200, 0, 50, 100, 150, 200, 250, 300, 400, 500,1000, 2000, 2900]
layers_to_plot = [z+1 for z in layers_to_plot]
z_values = np.array([z for z in layers_to_plot if z in z_values])

csv_string = h  # initialize csv string with header

for z in z_values:
    mask = pointgrid_np[:, 2] == z
    x_vals = pointgrid_np[mask][:, 0]
    y_vals = pointgrid_np[mask][:, 1]
    sdf_vals_layer = sdf_values_np[mask]

    # create a imshow plot
    plt.figure()
    plt.tricontourf(x_vals, y_vals, sdf_vals_layer, cmap="coolwarm", levels=14)
    plt.colorbar(label='SDF Value')
    plt.title(f'SDF values at z={z}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    # save the figure at a certain path
    plt.savefig(f'/cluster/home/{username}/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/sdf_conversions_tests/sdf_layer_z_{z:.2f}.png')
    plt.close()
    # add centre values to csv string - find closest point to (0,0,z)
    distances = np.sqrt((x_vals - 0)**2 + (y_vals - 0)**2)
    closest_index = np.argmin(distances)
    closest_sdf_value = sdf_vals_layer[closest_index]
    csv_string += f"\n{z},{closest_sdf_value}"

csv_string += f"\n{sdf_converter.volume()} "
# save csv string to a file
with open(f'/cluster/home/{username}/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/sdf_conversions_tests/sdf_centre_values.csv', 'w') as f:
    f.write(csv_string)


