import cadquery as cq


def innerNBZ2():
    # New implementation of innerNBZ for version 2
    inner_nbz = cq.Workplane("XY").circle(50).extrude(100)
    return inner_nbz


def outerNBZ2():
    # New implementation of outerNBZ for version 2
    outer_nbz = cq.Workplane("XY").circle(70).extrude(120)
    return outer_nbz
