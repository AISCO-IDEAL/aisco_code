import cadquery as cq
from components.outer.outer_box import outerBox

from .inner.base_triangle import Triangle
from .inner.wheels import Wheels
from .inner.seat import Seat

import random

def inner_nbz2_sym():
    """
    New implementation of innerNBZ for version 2
    Args: constraints from regulations
    Returns: cq.shape.solid of inner NBZ shape
    """

    # draw triangle
    triangle = Triangle().build()
    points = Triangle().get_points()

    # add wheels to the corners of the triangle
    wheels = Wheels()
    wheel = wheels.create_wheel()

    seat_class = Seat()
    seat = seat_class.create_seat()

    assembly = (
        cq.Assembly(triangle, name="triangle")
        # move to 0,0,0
        .add(
            wheel,
            loc=cq.Location(points[0].x, points[0].y, points[0].z),
            name="wheel_1",
        )
        .add(
            wheel,
            loc=cq.Location(points[1].x, points[1].y, points[1].z),
            name="wheel_2",
        )
        .add(
            wheel,
            loc=cq.Location(points[2].x, points[2].y, points[2].z),
            name="wheel_3",
        )
        .add(
            seat,
            loc=cq.Location(random.randint(0, int(points[2].x)),0,-100),
            name="Seat",
        )
    )

    return assembly


def outer_nbz2():
    """
    New implementation of outerNBZ for version 2
    Args: constraints from regulations
    Returns: cq.shape.solid of outer NBZ shape
    """

    box = outerBox()
    return box

#