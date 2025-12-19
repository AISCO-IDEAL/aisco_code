from dataclasses import dataclass, field

import cadquery as cq

from subprojects.wp2_assembler.version3.utils.reg_imp import get_regulation


# --- Wheels class: ---
@dataclass
class Wheels:
    """Wheel specifications for the solar car."""

    # Import from regulations wheel dimensions
    wheel_count: int = field(
        default_factory=lambda: int(get_regulation("wheel.min_wheel_count"))
    )
    wheel_diameter: float = field(
        default_factory=lambda: float(get_regulation("wheel.tire_diameter"))
    )
    wheel_width: float = field(
        default_factory=lambda: get_regulation("wheel.tire_width")
    )
    wheel_margin_axial: float = field(
        default_factory=lambda: get_regulation("wheel.margin_axial")
    )
    wheel_margin_radial: float = field(
        default_factory=lambda: get_regulation("wheel.margin_radial")
    )

    def get_wheel_info(self) -> str:
        """Return a description of the wheel configuration."""
        return f"This vehicle has {self.wheel_count} wheels with diameter {self.wheel_diameter}m and width {self.wheel_width}m."

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
            cq.Workplane("XZ")
            .circle(diameter / 2)
            .extrude(self.wheel_width)
            .translate((0, self.wheel_width / 2, diameter/2))
        )

    def create_wheel_with_margin(self) -> cq.Workplane:
        """
        Create a wheel object including margins based on the specifications.
        Args:
            None
        Returns:
            A cadquery Workplane object representing the wheel with margins.
        """
        total_diameter = self.wheel_diameter + self.wheel_margin_radial
        total_width = self.wheel_width + 2 * self.wheel_margin_axial
        return cq.Workplane("XY").circle(total_diameter).extrude(total_width)

if __name__ == "__main__":
    import time
    start = time.time()
    wheel = Wheels()
    w = wheel.create_wheel()

    bb = w.val().BoundingBox()

    bbox_list = [
        bb.xmin, bb.xmax,
        bb.ymin, bb.ymax,
        bb.zmin, bb.zmax
    ]
    end = time.time()
    print(bbox_list)
    print(end-start)
    from ocp_vscode import show
    show(w)