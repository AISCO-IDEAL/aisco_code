"""CAD helper for generating a regulation-compliant speed hump."""

import cadquery as cq
from ocp_vscode import show

from subprojects.wp2_assembler.version2.utils.reg_imp import get_regulation


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

    def _fetch_value(key: str) -> float:
        value = get_regulation(key)
        if value == -1:
            raise ValueError(f"Missing regulation entry for '{key}'")
        return float(value)

    slope_length = _fetch_value("no_build_zones.wheel_based_rules.speed_hump.outer_lenght")
    flat_length = _fetch_value("no_build_zones.wheel_based_rules.speed_hump.inner_lenght")
    height = _fetch_value("no_build_zones.wheel_based_rules.speed_hump.height")

    total_length = 2 * slope_length + flat_length
    body_width = _fetch_value("no_build_zones.outer_prism.width")

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


if __name__ == "__main__":
    hump = create_speed_hump()

    try:
        show(hump, names=["speed_hump"])
    except Exception as exc:  # pragma: no cover - viewer runtime issue
        print("Visualization failed:", exc)
        print("Speed hump solid:", hump)
