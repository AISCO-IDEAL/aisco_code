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
