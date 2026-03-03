import math
from pathlib import Path

import cadquery as cq
import yaml

from subprojects.wp2_assembler.version4.utils.reg_imp import get_regulation


def outer_nbz(assembly_id: int) -> tuple[cq.Workplane, cq.Workplane]:
    """
    Returns the approach and departure ramps

    :param assembly_id: id of assembly
    :return: tuple with both ramps
    :rtype: tuple with two workplanes
    """

    config_path = Path(__file__).parent.parent.parent / "config" / "assembly_map.yaml"

    with config_path.open("r") as file:
        assembly_map = yaml.safe_load(file) or {}

    assembly_map = assembly_map[assembly_id]

    # collect (y, x) for all wheels
    wheels = [(v["x"], v["y"]) for k, v in assembly_map.items() if isinstance(v, dict) and k.startswith("wheel_")]
    if not wheels:
        raise ValueError("No wheels found in assembly_map")

    print(wheels)
    # backmost = smallest y, frontmost = largest y
    back_x = min(p[0] for p in wheels)
    front_x = max(p[0] for p in wheels)

    print("back_x:", back_x)
    print("front_x:", front_x)

    # percentages
    front_factor = 0.6
    back_factor = 0.6
    max_len = 5800
    max_width = 2300
    max_height = 2000

    # wheel distance
    wheel_distance = abs(front_x - back_x)

    # raw limits
    outer_xmax = front_x + front_factor * wheel_distance
    outer_xmin = back_x - back_factor * wheel_distance

    print(outer_xmin, outer_xmax)
    # cap total lenght to max length defined by regulations

    outer_len = outer_xmax - outer_xmin
    print(outer_len)
    if outer_len > max_len:
        center_x = 0.5 * (outer_xmax + outer_xmin)
        outer_xmin = center_x - max_len / 2
        outer_xmax = center_x + max_len / 2
    outer_len = outer_xmax - outer_xmin

    print(outer_len)

    outer = (
        cq.Workplane("XY")
        .moveTo(outer_xmin, -max_width / 2)
        .lineTo(outer_xmax, -max_width / 2)
        .lineTo(outer_xmax, max_width / 2)
        .lineTo(outer_xmin, max_width / 2)
        .close()
        .extrude(max_height)
        .val()
    )

    approach_angle = math.radians(get_regulation("no_build_zones.wheel_based_rules.approach_angle"))
    departure_angle = math.radians(get_regulation("no_build_zones.wheel_based_rules.departure_angle"))
    wheel_radius = (get_regulation("wheel.margin_radial") + get_regulation("wheel.tire_diameter")) / 2

    A = (front_x + math.sin(approach_angle) * wheel_radius, wheel_radius - math.cos(approach_angle) * wheel_radius)
    B = (outer_xmax, A[1] + (abs(outer_xmax - front_x)) * math.tan(approach_angle))
    C = (outer_xmax, 0)
    D = (front_x, 0)

    Front = (
        cq.Workplane("XZ", origin=(0, max_width / 2, 0))
        .moveTo(*A)  # Start at the contact point from the approach angle and the wheel
        .lineTo(*B)  # Line to the end of the bounding box
        .lineTo(*C)  # Line to the ground
        .lineTo(*D)  # Line to the contact between the ground and the wheel
        .radiusArc(A, -wheel_radius)  # Wheel arc to the start point
        .close()
        .extrude(max_width)
    )

    A = (back_x - math.sin(departure_angle) * wheel_radius, wheel_radius - math.cos(departure_angle) * wheel_radius)
    B = (outer_xmin, A[1] + (abs(outer_xmin - back_x)) * math.tan(departure_angle))
    C = (outer_xmin, 0)
    D = (back_x, 0)

    Rear = (
        cq.Workplane("XZ", origin=(0, max_width / 2, 0))
        .moveTo(*A)  # Start at the contact point from the approach angle and the wheel
        .lineTo(*B)  # Line to the end of the bounding box
        .lineTo(*C)  # Line to the ground
        .lineTo(*D)  # Line to the contact between the ground and the wheel
        .radiusArc(A, wheel_radius)  # Wheel arc to the start point
        .close()
        .extrude(max_width)
    )

    result = outer.cut(Front.val(), Rear.val()).Solids()[0]

    return result


if __name__ == "__main__":
    demo_id = 1
    from ocp_vscode import show

    result = outer_nbz(demo_id)
    show(result)
    print(type(result))
