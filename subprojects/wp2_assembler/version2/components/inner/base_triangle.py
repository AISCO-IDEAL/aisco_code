import cadquery as cq
from utils.distributions import truncated_gamma


class Triangle:
    def __init__(self):
        self.a = cq.Vector(0, 0, 0)
        self.b = cq.Vector(
            truncated_gamma("l"),  # get value lengthwise in x-direction
            truncated_gamma("w"),  # get value widthwise in y-direction
            0,  # get value heightwise in z-direction
        )
        self.c = cq.Vector(
            self.b.x,
            -self.b.y,
            0,
        )

    def build(self) -> cq.Workplane:
        triangle = (
            cq.Workplane("XY")
            .moveTo(self.a.x, self.a.y)  # Explicitly start at point a
            .lineTo(self.b.x, self.b.y)  # Draw to point b
            .lineTo(self.c.x, self.c.y)  # Draw to point c
            .close()  # Close back to point a
        )
        return triangle

    def get_points(self) -> tuple[cq.Vector, cq.Vector, cq.Vector]:
        return self.a, self.b, self.c
