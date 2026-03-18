from dataclasses import dataclass, field

import cadquery as cq

from subprojects.wp2_assembler.version4.utils.reg_imp import get_regulation


# --- Wheels class: ---
@dataclass
class Wheels:
    """Wheel specifications for the solar car."""

    # Import from regulations wheel dimensions
    wheel_diameter: float = field(default_factory=lambda: float(get_regulation("wheel.tire_diameter")))
    wheel_width: float = field(default_factory=lambda: get_regulation("wheel.tire_width"))

    def create_wheel(self) -> cq.Workplane:
        """
        Create a wheel object based on the specifications.
        Args:
            None
        Returns:
            A cadquery Workplane object representing the wheel.
        """
        diameter = self.wheel_diameter

        return (
            cq.Workplane("XZ").circle(diameter / 2).extrude(self.wheel_width).translate((0, self.wheel_width / 2, diameter / 2))
        )

    def bounding_box_list(self) -> list[float]:
        """
        Compute the wheel bounding box analytically, without creating CAD geometry.

        Returns:
            [xmin, xmax, ymin, ymax, zmin, zmax]
        """
        radius = self.wheel_diameter / 2
        width = self.wheel_width

        xmin = -radius
        xmax = radius
        ymin = -width / 2
        ymax = width / 2
        zmin = 0.0
        zmax = self.wheel_diameter

        return [xmin, xmax, ymin, ymax, zmin, zmax]


if __name__ == "__main__":
    import time

    start = time.time()
    wheel = Wheels()
    w = wheel.create_wheel()

    bb = w.val().BoundingBox()

    bbox_list = [bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax]
    end = time.time()
    print(bbox_list)
    print(end - start)

    start = time.time()
    wheel = Wheels()
    print(wheel.bounding_box_list())
    print(end - start)

    from ocp_vscode import show

    show(w)
