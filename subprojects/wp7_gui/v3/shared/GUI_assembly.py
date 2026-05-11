import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4]))

import math

import cadquery as cq
import yaml

from subprojects.wp2_assembler.version4.components.inner.seat import Seat
from subprojects.wp2_assembler.version4.components.inner.wheels import Wheels
from subprojects.wp2_assembler.version4.components.outer.outer_nbz import aabb_inside_outer_nbz
from subprojects.wp2_assembler.version4.components.vision import vision_box
from subprojects.wp2_assembler.version4.utils.assemblies_maker import (
    bbox_list_to_margin_dict,
    point_in_polygon,
    save_assembly_parameters,
)
from subprojects.wp2_assembler.version4.utils.reg_imp import get_regulation

_OUTPUT_DIR = Path(__file__).parents[1] / "frontend" / "assembly"
_PARAMS_PATH = _OUTPUT_DIR / "parameters.yaml"
_PLY_OUTPUT_PATH = _OUTPUT_DIR / "assembly.ply"
_CHECKS_OUTPUT_PATH = _OUTPUT_DIR / "checks.txt"

# --- export toggles ---
_EXPORT_SEAT = True
_EXPORT_WHEELS = True
_EXPORT_VISION = False
_EXPORT_OUTER = False


def _write_ply(shape: cq.Shape, path: Path) -> None:
    vertices, triangles = shape.tessellate(0.5)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(triangles)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v.x} {v.y} {v.z}\n")
        for tri in triangles:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def save_ply(inner: cq.Assembly, outer: cq.Solid, vision: cq.Assembly) -> None:
    solids: list[cq.Shape] = []

    solids.extend(inner.toCompound().Solids())
    if _EXPORT_OUTER:
        solids.append(outer)
    if _EXPORT_VISION:
        solids.extend(vision.toCompound().Solids())

    if not solids:
        print("[WARNING] No enabled geometry to export to assembly.ply.")
        return

    _write_ply(cq.Compound.makeCompound(solids), _PLY_OUTPUT_PATH)


def write_params(params: dict) -> None:
    with _PARAMS_PATH.open("w") as f:
        yaml.dump(params, f, default_flow_style=False)


def read_params() -> dict:
    with _PARAMS_PATH.open("r") as f:
        return yaml.safe_load(f) or {}


def _write_checks(messages: list[str]) -> None:
    with _CHECKS_OUTPUT_PATH.open("w") as f:
        if messages:
            f.write("\n".join(messages))
            f.write("\n")
        else:
            f.write("OK: no check warnings.\n")


def _inner_nbz_from_dict(params: dict) -> cq.Assembly:
    wheel = Wheels().create_wheel()
    seat = Seat().create_seat()

    assembly = cq.Assembly(name="inner_nbz_gui")
    assembly.add(
        seat,
        loc=cq.Location(cq.Vector(params["seat"]["x"], params["seat"]["y"], params["seat"]["z"])),
        name="Seat",
    )
    for key, val in params.items():
        if key.startswith("wheel_"):
            assembly.add(
                wheel,
                loc=cq.Location(cq.Vector(val["x"], val["y"], val["z"])),
                name=key,
            )
    return assembly


def _outer_nbz_from_dict(params: dict) -> cq.Solid:
    wheels = [(v["x"], v["y"]) for k, v in params.items() if isinstance(v, dict) and k.startswith("wheel_")]
    if not wheels:
        raise ValueError("No wheels found in params")

    back_x = min(p[0] for p in wheels)
    front_x = max(p[0] for p in wheels)

    front_factor = get_regulation("no_build_zones.wheel_based_rules.front_overhang")
    back_factor = get_regulation("no_build_zones.wheel_based_rules.rear_overhang")
    max_len = get_regulation("no_build_zones.outer_prism.length")
    max_width = get_regulation("no_build_zones.outer_prism.width")
    max_height = get_regulation("no_build_zones.outer_prism.height")

    wheel_distance = abs(front_x - back_x)
    outer_xmax = front_x + front_factor * wheel_distance
    outer_xmin = back_x - back_factor * wheel_distance

    outer_len = outer_xmax - outer_xmin
    if outer_len > max_len:
        center_x = 0.5 * (outer_xmax + outer_xmin)
        outer_xmin = center_x - max_len / 2
        outer_xmax = center_x + max_len / 2

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
    B = (outer_xmax, A[1] + abs(outer_xmax - front_x) * math.tan(approach_angle))
    C = (outer_xmax, 0)
    D = (front_x, 0)
    Front = (
        cq.Workplane("XZ", origin=(0, max_width / 2, 0))
        .moveTo(*A)
        .lineTo(*B)
        .lineTo(*C)
        .lineTo(*D)
        .radiusArc(A, -wheel_radius)
        .close()
        .extrude(max_width)
    )

    A = (back_x - math.sin(departure_angle) * wheel_radius, wheel_radius - math.cos(departure_angle) * wheel_radius)
    B = (outer_xmin, A[1] + abs(outer_xmin - back_x) * math.tan(departure_angle))
    C = (outer_xmin, 0)
    D = (back_x, 0)
    Rear = (
        cq.Workplane("XZ", origin=(0, max_width / 2, 0))
        .moveTo(*A)
        .lineTo(*B)
        .lineTo(*C)
        .lineTo(*D)
        .radiusArc(A, wheel_radius)
        .close()
        .extrude(max_width)
    )

    return outer.cut(Front.val(), Rear.val()).Solids()[0]


def _vision_nbz_from_dict(params: dict) -> cq.Assembly:
    driver_cone = vision_box.vision_driv_nbz()
    assy = cq.Assembly()
    assy.add(
        driver_cone,
        loc=cq.Location(
            cq.Vector(
                params["seat"]["x"],
                params["seat"]["y"],
                params["seat"]["z"] + get_regulation("no_build_zones.vision_box.eye_level_z"),
            )
        ),
        name="driver_cone",
    )
    return assy


def _check_validity(params: dict) -> list[str]:
    messages: list[str] = []

    def warn(message: str) -> None:
        messages.append(message)
        print(message)

    seat = params["seat"]
    wheels = {k: v for k, v in params.items() if k.startswith("wheel_")}

    # Check 1: all wheels at same x — outer NBZ collapses to zero length
    if len(wheels) > 1 and len({v["x"] for v in wheels.values()}) == 1:
        warn("[WARNING] Degenerate layout: all wheels share the same X position — outer NBZ has zero length.")

    # Check 2: bounding box collisions between seat and wheels
    seat_margin = bbox_list_to_margin_dict(Seat().bounding_box_list())
    wheel_margin = bbox_list_to_margin_dict(Wheels().bounding_box_list())
    parts = {"seat": (seat, seat_margin), **{k: (v, wheel_margin) for k, v in wheels.items()}}

    def aabb(pos: dict, margin: dict) -> dict:
        return {
            "xmin": pos["x"] - margin["x_back"],
            "xmax": pos["x"] + margin["x_fwd"],
            "ymin": pos["y"] - margin["y_down"],
            "ymax": pos["y"] + margin["y_up"],
            "zmin": pos["z"] - margin["z_down"],
            "zmax": pos["z"] + margin["z_up"],
        }

    def overlaps(a: dict, b: dict) -> bool:
        return (
            a["xmin"] <= b["xmax"]
            and a["xmax"] >= b["xmin"]
            and a["ymin"] <= b["ymax"]
            and a["ymax"] >= b["ymin"]
            and a["zmin"] <= b["zmax"]
            and a["zmax"] >= b["zmin"]
        )

    names = list(parts.keys())
    aabbs = {n: aabb(*parts[n]) for n in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if overlaps(aabbs[a], aabbs[b]):
                warn(f"[WARNING] Collision: {a} and {b} bounding boxes overlap.")

    # Check 3: seat not enclosed by wheel polygon — vehicle would be unstable
    wheel_xy = [(v["x"], v["y"]) for v in wheels.values()]
    if len(wheel_xy) >= 3 and not point_in_polygon((seat["x"], seat["y"]), wheel_xy):
        warn("[WARNING] Unstable assembly: seat is not enclosed by the wheel polygon.")

    # Check 4: seat bounding box outside outer NBZ
    seat_aabb = aabbs["seat"]
    if not aabb_inside_outer_nbz(params, seat_aabb):
        warn("[WARNING] Out of bounds: seat bounding box extends outside the outer NBZ.")

    wheel_clearance = get_regulation("no_build_zones.outer_prism.wheel_clearance")

    # Check 5: wheel bounding boxes outside outer NBZ after wheel-clearance cut
    for wheel_name in wheels:
        wheel_aabb = dict(aabbs[wheel_name])
        wheel_aabb["zmin"] = max(wheel_aabb["zmin"], wheel_clearance)
        if not aabb_inside_outer_nbz(params, wheel_aabb):
            warn(f"[WARNING] Out of bounds: {wheel_name} bounding box extends outside the outer NBZ after clearance cut.")

    # Check 6: seat bottom below wheel clearance plane
    seat_bottom = seat["z"] - seat_margin["z_down"]
    if seat_bottom < wheel_clearance:
        warn(
            f"[WARNING] Ground violation: seat bottom ({seat_bottom:.1f} mm) is below the wheel clearance plane ({wheel_clearance:.1f} mm)."
        )

    _write_checks(messages)
    return messages


def call_assembly_from_params() -> tuple[cq.Assembly, cq.Solid, cq.Assembly]:
    """
    Reads parameters.yaml and returns the same tuple as call_assembly() in nbz.py:
    (cut_inner_assembly, cut_outer_nbz_solid, vision_assembly)
    """
    params = read_params()
    _check_validity(params)

    ground_clearance = get_regulation("no_build_zones.outer_prism.ground_clearance")
    wheel_clearance = get_regulation("no_build_zones.outer_prism.wheel_clearance")

    def keep_above_z(solid: cq.Solid, z_cut: float) -> cq.Solid | None:
        bbox = solid.BoundingBox()
        dx = max(bbox.xlen, 1.0) + 10000
        dy = max(bbox.ylen, 1.0) + 10000
        dz = max(bbox.zlen, 1.0) + 10000
        upper_box = (
            cq.Workplane("XY")
            .box(dx, dy, dz, centered=(True, True, False))
            .translate(((bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2, z_cut))
            .val()
        )
        result = solid.intersect(upper_box)
        if result is None:
            return None
        try:
            if result.isNull():
                return None
        except AttributeError:
            pass
        return result

    inner_solids = _inner_nbz_from_dict(params).toCompound().Solids()
    if len(inner_solids) < 1:
        raise ValueError("No inner solids found for GUI params assembly.")

    seat_solid = inner_solids[0]
    wheel_solids = inner_solids[1:]

    cut_inner_solids: list[cq.Solid] = [seat_solid]
    for wheel in wheel_solids:
        cut_wheel = keep_above_z(wheel, wheel_clearance)
        if cut_wheel is not None:
            cut_inner_solids.append(cut_wheel)

    cut_outer = keep_above_z(_outer_nbz_from_dict(params), ground_clearance)
    if cut_outer is None:
        print("[WARNING] Outer NBZ was completely removed by the ground clearance cut — layout is degenerate.")
        cut_outer = _outer_nbz_from_dict(params)

    cut_inner_assembly = cq.Assembly()
    if _EXPORT_SEAT and cut_inner_solids:
        cut_inner_assembly.add(cut_inner_solids[0], name="inner_0")
    if _EXPORT_WHEELS:
        for i, s in enumerate(cut_inner_solids[1:], 1):
            cut_inner_assembly.add(s, name=f"inner_{i}")
    cut_inner_assembly.name = "inner_assembly"

    outer_solid = cut_outer.Solids()[0]
    vision = _vision_nbz_from_dict(params) if _EXPORT_VISION else cq.Assembly()

    result = cut_inner_assembly, outer_solid, vision
    save_ply(*result)
    return result


def add_current_gui_assembly_to_map() -> int | None:
    """
    Add the current GUI parameters to assembly_map.yaml only if all checks pass.

    Returns:
        New assembly ID if the layout w
        as added, otherwise None.
    """
    params = read_params()
    messages = _check_validity(params)
    if messages:
        print("[INVALID] Current GUI assembly was not added to assembly_map.yaml.")
        return None

    assembly_id = int(save_assembly_parameters(params))
    print(f"[OK] Added current GUI assembly to assembly_map.yaml as id {assembly_id}.")
    return assembly_id


if __name__ == "__main__":
    from subprojects.wp2_assembler.version4.GUI_helpers import _slider_gui

    _slider_gui.run(read_params, write_params, call_assembly_from_params, add_current_gui_assembly_to_map)
