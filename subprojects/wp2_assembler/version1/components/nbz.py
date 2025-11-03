import cadquery as cq

from .car_assembly import *


def outerBoundingBox(height: float, width: float, depth: float):  # -> TopoDS_Shape
    """
    Generates an outer no-build-zone (outerNBZ) car assembly with fixed dimensions.
    Args:
        height (float): The height of the bounding box in mm.
        width (float): The width of the bounding box in mm.
        depth (float): The depth of the bounding box in mm.
    Returns:
        TopoDS_Shape: The generated outer no-build-zone car assembly.
    """

    # Create a box with the specified dimensions
    bounding_box = (
        cq.Workplane("XY").box(depth, width, height, centered=(True, True, False)).val()
    )
    return bounding_box


# Generate outer no-build-zone (outerNBZ) car assembly with dimensions 5800mm x 2300mm x 1650mm
def outerNBZ():  # -> TopoDS_Shape  # noqa: N802
    """
    Generates an outer no-build-zone (outerNBZ) car assembly with fixed dimensions.
    Returns:
        TopoDS_Shape: The generated outer no-build-zone car assembly.
    """
    box = outerBoundingBox(
        height=1650, width=2300, depth=5800
    )  # do not hardcode dimensions here, create extra input files later
    return box


# Generate inner no-build-zone (innerNBZ) car assembly
def innerNBZ(  # noqa: N802
    random_magnitude_chassis: float = 0.1,
    random_magnitude_seat: float = 0.0,
    random_magnitude_wheels: float = 0.2,
):  # -> TopoDS_Shape
    """
    Generates an inner no-build-zone (innerNBZ) car assembly with randomized parameters.
    Args:
        random_magnitude_chassis (float): Magnitude of randomization for the chassis component.
        random_magnitude_seat (float): Magnitude of randomization for the seat component.
        random_magnitude_wheels (float): Magnitude of randomization for the wheel component.
    Returns:
        TopoDS_Shape: The generated inner no-build-zone car assembly.
    """
    # create car assembly with assembler from freelancers
    assy = CarAssembly().randomized(
        random_magnitude_chassis=random_magnitude_chassis,
        random_magnitude_seat=random_magnitude_seat,
        random_magnitude_wheels=random_magnitude_wheels,
    )

    # convert assembly to compound and then to solid
    compound = assy.model.toCompound()
    solid = cq.Solid(compound.fuse().wrapped)

    return solid
