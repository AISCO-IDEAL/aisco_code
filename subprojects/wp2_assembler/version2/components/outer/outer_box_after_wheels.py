import cadquery as cq
from utils.reg_imp import get_regulation


def outerBox_afterWheels():
    """
    Create the outer box shape based on regulations, considering the placement of the wheels.
    Args: dimensions from regulations
    Returns: cq.shape.solid of outer box shape
    """
    length = get_regulation("no_build_zones.outer_prism.length")
    width = get_regulation("no_build_zones.outer_prism.width")
    height = get_regulation("no_build_zones.outer_prism.height")



    return (
        cq.Workplane("XY")
        .box(length, width, height, centered=(True, True, False))
        .val()
    )
