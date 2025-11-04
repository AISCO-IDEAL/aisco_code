import cadquery as cq
from cadquery import Assembly, Color
from components import nbz
from ocp_vscode import show


def create_coordinate_system(axis_length=100, axis_radius=2):
    """
    Create a 3D coordinate system with X (red), Y (green), and Z (blue) axes.

    Args:
        axis_length: Length of each axis
        axis_radius: Radius of the axis cylinders

    Returns:
        Assembly with the three axes
    """
    # Create X axis (red)
    x_axis = cq.Workplane("YZ").circle(axis_radius).extrude(axis_length)

    # Create Y axis (green)
    y_axis = cq.Workplane("XZ").circle(axis_radius).extrude(axis_length)

    # Create Z axis (blue)
    z_axis = cq.Workplane("XY").circle(axis_radius).extrude(axis_length)

    # Create assembly
    coord_system = Assembly(x_axis, name="X_axis", color=Color(1, 0, 0, 0.8))  # Red
    coord_system.add(y_axis, name="Y_axis", color=Color(0, 1, 0, 0.8))  # Green
    coord_system.add(z_axis, name="Z_axis", color=Color(0, 0, 1, 0.8))  # Blue

    return coord_system


if __name__ == "__main__":  # only run when executing this file directly
    # Create coordinate system
    coord_system = create_coordinate_system(axis_length=100, axis_radius=2)
    print("Coordinate system created")

    # Create inner NBZ
    inner_nbz = nbz.innerNBZ()
    print("Inner no-build-zone created")
    # create inner NBZ with coordinate system
    inner_nbzWCS = Assembly(inner_nbz, name="innerNBZ", color=Color(0, 0, 1, 0.3))
    inner_nbzWCS.add(coord_system, name="coordinate_systemI")
    print("Inner no-build-zone updated with coordinate system")

    # Create outer NBZ
    outer_nbz = nbz.outerNBZ()
    print("Outer no-build-zone created")
    # create outer NBZ with coordinate system
    outer_nbzWCS = Assembly(outer_nbz, name="outerNBZ", color=Color(1, 0, 0, 0.3))
    outer_nbzWCS.add(coord_system, name="coordinate_systemO")
    print("Outer no-build-zone updated with coordinate system")

    # Create assembly of both NBZs with coordinate system
    assembled_car = Assembly(outer_nbz, name="outerNBZ", color=Color(1, 0, 0, 0.3))
    assembled_car.add(inner_nbzWCS, name="innerNBZ", color=Color(0, 0, 1))
    assembled_car.add(coord_system, name="coordinate_systemA")
    print("Assembly created")

    # Prompt to visualize the shapes
    while True:
        userinput = input("Visualize innerNBZ and/or outerNBZ? (y/n): ")
        if userinput.lower() == "y":
            if show is None:
                print("Visualization not available. ocp_vscode is not installed.")
                break
            userinput2 = input("Visualize innerNBZ, outerNBZ or both? (i/o/b): ")
            if userinput2.lower() == "i":
                show(inner_nbzWCS)
                print(type(inner_nbzWCS))
            elif userinput2.lower() == "o":
                show(outer_nbzWCS)
                print(type(outer_nbzWCS))
            elif userinput2.lower() == "b":
                show(assembled_car)
                print(type(assembled_car))
            else:
                print("Invalid input. Please enter 'i', 'o' or 'b'.")
        elif userinput.lower() == "n":
            print("Visualization stopped.")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
