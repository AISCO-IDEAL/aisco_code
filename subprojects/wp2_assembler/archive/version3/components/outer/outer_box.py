from typing import cast

import cadquery as cq

from subprojects.wp2_assembler.version3.utils.reg_imp import get_regulation


def outer_box() -> cq.Solid:
    """
    Create the outer box shape based on regulations.
    Args: dimensions from regulations
    Returns: cq.shape.solid of outer box shape
    """
    length = get_regulation("no_build_zones.outer_prism.length")
    width = get_regulation("no_build_zones.outer_prism.width")
    height = get_regulation("no_build_zones.outer_prism.height")

    return cast(cq.Solid, cq.Workplane("XY").box(length, width, height, centered=(True, True, False)).val())


def outer_box2(length: float, width: float, height: float) -> cq.Solid:
    """
    Create the outer box shape based on given dimensions.
    Args: dimensions from regulations
    Returns: cq.shape.solid of outer box shape
    """

    return cast(cq.Solid, cq.Workplane("XY").box(length, width, height, centered=(True, True, False)).val())
