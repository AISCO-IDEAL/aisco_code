import math
from pathlib import Path

import cadquery as cq
import yaml

from subprojects.wp2_assembler.version4.utils.reg_imp import get_regulation


def outer_nbz(assembly_id: int) -> cq.Solid:
    """
    Returns the outer nbz with the cutted approach and departure ramps

    :param assembly_id: id of assembly
    :return: outer nbz
    :rtype: solid
    """

    config_path = Path(__file__).parent.parent.parent / "config" / "assembly_map.yaml"

    with config_path.open("r") as file:
        assembly_map = yaml.safe_load(file) or {}

    assembly_map = assembly_map[assembly_id]

    # collect (y, x) for all wheels
    wheels = [(v["x"], v["y"]) for k, v in assembly_map.items() if isinstance(v, dict) and k.startswith("wheel_")]
    if not wheels:
        raise ValueError("No wheels found in assembly_map")

    # print(wheels)  # noqa: ERA001
    # backmost = smallest y, frontmost = largest y
    back_x = min(p[0] for p in wheels)
    front_x = max(p[0] for p in wheels)

    # print("back_x:", back_x) # noqa: ERA001
    # print("front_x:", front_x)     # noqa: ERA001

    # # percentages
    # front_factor = 0.6 # noqa: ERA001
    # back_factor = 0.6 # noqa: ERA001
    # max_len = 5800    # noqa: ERA001
    # max_width = 2300  # noqa: ERA001
    # max_height = 2000  # noqa: ERA001

    front_factor = get_regulation("no_build_zones.wheel_based_rules.front_overhang")
    back_factor = get_regulation("no_build_zones.wheel_based_rules.rear_overhang")
    max_len = get_regulation("no_build_zones.outer_prism.length")
    max_width = get_regulation("no_build_zones.outer_prism.width")
    max_height = get_regulation("no_build_zones.outer_prism.height")

    # wheel distance
    wheel_distance = abs(front_x - back_x)

    # raw limits
    outer_xmax = front_x + front_factor * wheel_distance
    outer_xmin = back_x - back_factor * wheel_distance

    # print(outer_xmin, outer_xmax)  # noqa: ERA001 used for debugging
    # cap total lenght to max length defined by regulations

    outer_len = outer_xmax - outer_xmin
    # print(outer_len) # noqa: ERA001 used for debugging
    if outer_len > max_len:
        center_x = 0.5 * (outer_xmax + outer_xmin)
        outer_xmin = center_x - max_len / 2
        outer_xmax = center_x + max_len / 2
    outer_len = outer_xmax - outer_xmin

    # print(outer_len) # noqa: ERA001 used for debugging

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
    wheel_radius = get_regulation("wheel.tire_diameter") / 2

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


def aabb_inside_outer_nbz(  # noqa: PLR0915
    entry: dict[str, object],
    aabb: dict[str, float],
    tol: float = 1e-6,
) -> bool:
    """
    Fast analytical containment check for the seat AABB inside the outer NBZ.

    This reproduces the same geometry logic as outer_nbz(), but without creating
    any CadQuery geometry.

    Args:
        entry: One assembly entry from assembly_map.yaml
        aabb: Dict with xmin, xmax, ymin, ymax, zmin, zmax
        tol: Numerical tolerance

    Returns:
        True if all 8 AABB corners are inside the analytical outer NBZ
    """

    wheels = [float(v["x"]) for k, v in entry.items() if isinstance(v, dict) and k.startswith("wheel_") and "x" in v]
    if not wheels:
        raise ValueError("No wheels found in assembly entry")

    back_x = min(wheels)
    front_x = max(wheels)

    front_factor = float(get_regulation("no_build_zones.wheel_based_rules.front_overhang"))
    back_factor = float(get_regulation("no_build_zones.wheel_based_rules.rear_overhang"))
    max_len = float(get_regulation("no_build_zones.outer_prism.length"))
    max_width = float(get_regulation("no_build_zones.outer_prism.width"))
    max_height = float(get_regulation("no_build_zones.outer_prism.height"))

    approach_angle = math.radians(float(get_regulation("no_build_zones.wheel_based_rules.approach_angle")))
    departure_angle = math.radians(float(get_regulation("no_build_zones.wheel_based_rules.departure_angle")))
    wheel_radius = float(get_regulation("wheel.tire_diameter")) / 2.0

    wheel_distance = abs(front_x - back_x)

    outer_xmax = front_x + front_factor * wheel_distance
    outer_xmin = back_x - back_factor * wheel_distance

    outer_len = outer_xmax - outer_xmin
    if outer_len > max_len:
        center_x = 0.5 * (outer_xmax + outer_xmin)
        outer_xmin = center_x - max_len / 2.0
        outer_xmax = center_x + max_len / 2.0

    a_front_x = front_x + math.sin(approach_angle) * wheel_radius
    a_front_z = wheel_radius - math.cos(approach_angle) * wheel_radius
    b_front_x = outer_xmax
    b_front_z = a_front_z + abs(outer_xmax - front_x) * math.tan(approach_angle)

    a_rear_x = back_x - math.sin(departure_angle) * wheel_radius
    a_rear_z = wheel_radius - math.cos(departure_angle) * wheel_radius
    b_rear_x = outer_xmin
    b_rear_z = a_rear_z + abs(outer_xmin - back_x) * math.tan(departure_angle)

    def line_z(x: float, x1: float, z1: float, x2: float, z2: float) -> float:
        if abs(x2 - x1) < tol:
            return max(z1, z2)
        return z1 + (x - x1) * (z2 - z1) / (x2 - x1)

    def wheel_arc_z(x: float, wheel_center_x: float) -> float:
        dx = x - wheel_center_x
        inside = wheel_radius**2 - dx**2
        if inside < 0:
            inside = 0.0
        return wheel_radius - math.sqrt(inside)

    def required_min_z(x: float) -> float:
        z_req = 0.0

        if front_x - tol <= x <= outer_xmax + tol:
            if x <= a_front_x + tol:
                z_req = max(z_req, wheel_arc_z(x, front_x))
            else:
                z_req = max(z_req, line_z(x, a_front_x, a_front_z, b_front_x, b_front_z))

        if outer_xmin - tol <= x <= back_x + tol:
            if x >= a_rear_x - tol:
                z_req = max(z_req, wheel_arc_z(x, back_x))
            else:
                z_req = max(z_req, line_z(x, b_rear_x, b_rear_z, a_rear_x, a_rear_z))

        return z_req

    def point_inside(x: float, y: float, z: float) -> bool:
        if x < outer_xmin - tol or x > outer_xmax + tol:
            return False
        if y < -max_width / 2.0 - tol or y > max_width / 2.0 + tol:
            return False
        if z < 0.0 - tol or z > max_height + tol:
            return False
        return not z < required_min_z(x) - tol

    corners = [
        (aabb["xmin"], aabb["ymin"], aabb["zmin"]),
        (aabb["xmin"], aabb["ymin"], aabb["zmax"]),
        (aabb["xmin"], aabb["ymax"], aabb["zmin"]),
        (aabb["xmin"], aabb["ymax"], aabb["zmax"]),
        (aabb["xmax"], aabb["ymin"], aabb["zmin"]),
        (aabb["xmax"], aabb["ymin"], aabb["zmax"]),
        (aabb["xmax"], aabb["ymax"], aabb["zmin"]),
        (aabb["xmax"], aabb["ymax"], aabb["zmax"]),
    ]

    return all(point_inside(x, y, z) for x, y, z in corners)


if __name__ == "__main__":
    demo_id = 1
    from ocp_vscode import show

    result = outer_nbz(demo_id)
    show(result)
    print(type(result))
