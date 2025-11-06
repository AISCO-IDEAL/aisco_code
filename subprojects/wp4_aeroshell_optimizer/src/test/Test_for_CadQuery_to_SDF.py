"""
In this file the CadQueryToSDF class from CadQuery_to_SDF.py is tested with a simple shapes.
Run this file by running the two follwoing commands in the terminal:

cd /cluster/home/clhuber/AISCO-code

python -m subprojects.wp4_aeroshell_optimizer.src.test.Test_for_CadQuery_to_SDF
"""
from typing import cast

import cadquery as cq
import matplotlib.pyplot as plt
import numpy as np

from ....wp2_assembler.version1.components.nbz import innerNBZ, outerNBZ
from ..utils.CadQuery_to_SDF import SDFfromCadQuery

username = "clhuber"
sdf_converter: SDFfromCadQuery
# test with a Cylinder
if False:
    cylinder = cq.Workplane("XY").circle(500).extrude(200).objects[0]
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

# use the outerNBZ from wp2_assembler
if False:
    outer_nbz_shape = outerNBZ()
    sdf_converter = SDFfromCadQuery(outer_nbz_shape)

sdf_valus = sdf_converter.sdf_at_pointgrid()

# sort them for their z values and plot each layer
pointgrid = sdf_converter.pointgrid()
pointgrid_np = np.array(pointgrid)
sdf_values_np = np.array(sdf_valus)
z_values = np.unique(pointgrid_np[:, 2])

# picking certain layers to plot
layers_to_plot = [-1000, -200, 0, 50, 100, 150, 200, 250, 300, 400, 500,1000, 2000, 2900]
z_values = np.array([z for z in layers_to_plot if z in z_values])

csv_string = f"{type(sdf_converter._shape)}"  # initialize csv string with header

for z in z_values:
    mask = pointgrid_np[:, 2] == z
    x_vals = pointgrid_np[mask][:, 0]
    y_vals = pointgrid_np[mask][:, 1]
    sdf_vals_layer = sdf_values_np[mask]

    # create a imshow plot
    plt.figure()
    plt.tricontourf(x_vals, y_vals, sdf_vals_layer, cmap="coolwarm", levels=[-1000, -500, -400, -300, -200, -100, 0, 100, 400, 500, 1000])
    plt.colorbar(label='SDF Value')
    plt.title(f'SDF values at z={z}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    # save the figure at a certain path
    plt.savefig(f'/cluster/home/{username}/AISCO-code/subprojects/wp4_aeroshell_optimizer/data/sdf_conversions_tests/sdf_layer_z_{z:.2f}.png')
    plt.close()
    # add centre values to csv string - find closest point to (0,0,z)
    distances = np.sqrt((x_vals - 0)**2 + (y_vals - 0)**2)
    closest_index = np.argmin(distances)
    closest_sdf_value = sdf_vals_layer[closest_index]
    csv_string += f"\n{z},{closest_sdf_value}"

# save csv string to a file
with open(f'/cluster/home/{username}/AISCO-code/subprojects/wp4_aeroshell_optimizer/data/sdf_conversions_tests/sdf_centre_values.csv', 'w') as f:
    f.write(csv_string)

# plot the SDV vales at z=200, y=0 and varrying x
for z in range(0, 1650, 50):
    z_fixed = z # 200
    y_fixed = 50
    mask = pointgrid_np[:, 2] == z_fixed
    mask = mask & (pointgrid_np[:, 1] == y_fixed)
    x_vals = pointgrid_np[mask][:, 0]
    sdf_vals_line = sdf_values_np[mask]
    plt.figure()
    plt.plot(x_vals, sdf_vals_line, marker='o')
    plt.title(f'SDF values at z={z_fixed}, y={y_fixed}')
    plt.xlabel('X')
    plt.ylabel('SDF Value')
    plt.grid()
    plt.savefig(f'/cluster/home/{username}/AISCO-code/subprojects/wp4_aeroshell_optimizer/data/sdf_conversions_tests/sdf_line_z_{z_fixed}_y_{y_fixed}.png')
    plt.close()
