"""CAD helper for generating a regulation-compliant speed hump."""

import cadquery as cq
from ocp_vscode import show

from subprojects.wp2_assembler.version2.utils.reg_imp import get_regulation


def _fetch_value(key: str) -> float:
    """Return a float regulation value or raise if it is missing."""
    value = get_regulation(key)
    if value == -1:
        raise ValueError(f"Missing regulation entry for '{key}'")
    return float(value)


def _collect_dimensions() -> tuple[float, float, float, float, float]:
    """Gather commonly used regulation dimensions for the speed hump."""
    slope_length = _fetch_value("no_build_zones.wheel_based_rules.speed_hump.outer_lenght")
    flat_length = _fetch_value("no_build_zones.wheel_based_rules.speed_hump.inner_lenght")
    height = _fetch_value("no_build_zones.wheel_based_rules.speed_hump.height")
    total_length = 2 * slope_length + flat_length
    body_width = _fetch_value("no_build_zones.outer_prism.width")
    return slope_length, flat_length, height, total_length, body_width


def create_speed_hump() -> cq.Workplane:
    """
    Build a simple trapezoidal speed hump that follows the regulation profile.

    The profile (viewed from the side) consists of:
        - An upward slope with length `outer_lenght`
        - A flat top with length `inner_lenght`
        - A downward slope mirroring the first

    Returns:
        cadquery Workplane solid ready for placement in the scene origin.
    """
    slope_length, flat_length, height, total_length, body_width = _collect_dimensions()

    profile = (
        cq.Workplane("XZ")
        .moveTo(0, 0)
        .lineTo(slope_length, height)
        .lineTo(slope_length + flat_length, height)
        .lineTo(total_length, 0)
        .close()
    )

    bump = profile.extrude(body_width, both=True)

    return bump


def create_speed_hump_and_floor() -> cq.Workplane:
    """
    Build the speed hump sitting on a supporting box sized from regulation values.

    The box uses double the regulation-based length and width. Its height is twice
    the maximum of the hump length and height, sits fully below z=0, and is
    centered on x and y.
    """
    _, _, height, total_length, body_width = _collect_dimensions()
    bump = create_speed_hump()

    base_length = 3 * _fetch_value("no_build_zones.outer_prism.length") + total_length
    base_width = 2 * body_width
    base_height = 1.5 * max(_fetch_value("no_build_zones.outer_prism.length"), _fetch_value("no_build_zones.outer_prism.height"))

    base = (
        cq.Workplane("XY")
        .box(base_length, base_width, base_height, centered=True)
        .translate((0, 0, -base_height / 2))  # place the box entirely below z=0
    )

    centered_bump = bump.translate((-total_length / 2.0, 0, 0))
    return base.union(centered_bump)


if __name__ == "__main__":
    hump = create_speed_hump_and_floor()

    try:
        show(hump, names=["speed_hump"])
    except Exception as exc:  # pragma: no cover - viewer runtime issue
        print("Visualization failed:", exc)
        print("Speed hump solid:", hump)
