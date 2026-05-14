import time
from pathlib import Path

import cadquery as cq
import yaml

from subprojects.wp2_assembler.version4.components.inner.seat import Seat
from subprojects.wp2_assembler.version4.components.inner.wheels import Wheels
from subprojects.wp2_assembler.version4.components.outer.outer_nbz import outer_nbz
from subprojects.wp2_assembler.version4.components.vision import tracker_box, vision_box
from subprojects.wp2_assembler.version4.utils.reg_imp import get_regulation


def inner_nbz(assembly_nr: int = 0):
    """
    This function takes the positions from the assembly map, and returns an assembly.
    """
    config_path = Path(__file__).parent.parent / "config" / "assembly_map.yaml"

    # print(config_path)  # noqa: ERA001
    with config_path.open("r") as file:
        assembly_map = yaml.safe_load(file) or {}

    params = assembly_map[assembly_nr]

    wheel = Wheels().create_wheel()
    seat = Seat().create_seat()

    assembly = cq.Assembly(name="inner nbz " + str(assembly_nr))

    assembly.add(
        seat,
        loc=cq.Location(cq.Vector(params["seat"]["x"], params["seat"]["y"], params["seat"]["z"])),
        name="Seat",
    )

    for key in params:
        if key.startswith("wheel_"):
            assembly.add(
                wheel,
                loc=cq.Location(cq.Vector(params[key]["x"], params[key]["y"], params[key]["z"])),
                name=key,
            )

    return assembly


def call_assembly(id_number: int) -> tuple[cq.Assembly, cq.Solid, cq.Assembly]:
    if id_number > max_assembly_id():
        raise ValueError("ValueError in wp2, version4, nbz. id_number of assembly bigger than max.")

    ground_clearance = get_regulation("no_build_zones.outer_prism.ground_clearance")
    wheel_clearance = get_regulation("no_build_zones.outer_prism.wheel_clearance")

    def keep_above_z(solid: cq.Solid, z_cut: float) -> cq.Solid | None:
        """
        Keep only the part of the solid with z >= z_cut.
        Returns None if nothing remains.
        """
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

    inner_solids = inner_nbz(id_number).toCompound().Solids()

    if len(inner_solids) < 1:
        raise ValueError(f"No inner solids found for assembly {id_number}.")

    seat_solid = inner_solids[0]
    wheel_solids = inner_solids[1:]

    seat_bbox = seat_solid.BoundingBox()
    if seat_bbox.zmin < wheel_clearance:
        raise ValueError(
            f"Seat intersects the outer NBZ cut plane. seat zmin = {seat_bbox.zmin:.3f} mm, required >= {wheel_clearance:.3f} mm."
        )

    cut_inner_solids: list[cq.Solid] = [seat_solid]

    for wheel in wheel_solids:
        cut_wheel = keep_above_z(wheel, wheel_clearance)  # the +1 is so that there is space for the mesh to go throgh.
        if cut_wheel is not None:
            cut_inner_solids.append(cut_wheel)

    cut_outer_nbz = keep_above_z(outer_nbz(id_number), ground_clearance)
    if cut_outer_nbz is None:
        raise ValueError(f"Outer NBZ was completely removed for assembly {id_number}.")

    outer_solid = cut_outer_nbz.Solids()[0]

    # Recenter both inner and outer to x=0, y=0.
    # The outer NBZ is built from wheel positions, so its x-center is at the axle
    # midpoint (e.g. +800 mm for a front-axle-at-2900 layout), not at the coordinate
    # origin.  Shifting here aligns the assembly with the SDF training domain, eikonal
    # regularisation domain, and mesh extractor grid (all centred at x=0, y=0).
    outer_bbox = outer_solid.BoundingBox()
    cx = (outer_bbox.xmin + outer_bbox.xmax) / 2.0
    cy = (outer_bbox.ymin + outer_bbox.ymax) / 2.0
    if abs(cx) > 0.001 or abs(cy) > 0.001:
        offset = cq.Vector(-cx, -cy, 0.0)
        outer_solid = outer_solid.translate(offset)
        cut_inner_solids = [s.translate(offset) for s in cut_inner_solids]

    cut_inner_assembly = cq.Assembly()
    for i, s in enumerate(cut_inner_solids):
        cut_inner_assembly.add(s, name=f"inner_{i}")

    cut_inner_assembly.name = "inner_assembly"

    return cut_inner_assembly, outer_solid, vision_nbz(id_number)


def max_assembly_id() -> int:
    assembly_map_path = Path("subprojects/wp2_assembler/version4/config/assembly_map.yaml")

    with assembly_map_path.open("r") as f:
        data = yaml.safe_load(f) or {}

    if not data:
        raise ValueError("assembly_map is empty")

    return max(int(k) for k in data)


def bounding_box_of_assembly(id_number: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Docstring for bounding_box_of_assembly

    :param id_number: id number of the assembly of the wanted bounding box
    :return: tuple with touples with ((x_min, y_min, z_min),(x_max,y_max,z_max))
    :rtype: tuple[tuple[float, float, float], tuple[float, float, float]]
    """
    inner, _, _ = call_assembly(id_number)
    inner_bounding_box = inner.toCompound().BoundingBox()

    return (
        (inner_bounding_box.xmin, inner_bounding_box.ymin, inner_bounding_box.zmin),
        (inner_bounding_box.xmax, inner_bounding_box.ymax, inner_bounding_box.zmax),
    )


def vision_nbz(id_number):
    config_path = Path(__file__).parent.parent / "config" / "assembly_map.yaml"

    # print(config_path)  # noqa: ERA001
    with config_path.open("r") as file:
        assembly_map = yaml.safe_load(file) or {}

    params = assembly_map[id_number]

    driver_cone = vision_box.vision_driv_nbz()
    tracker_cone = tracker_box.tracker_cone_nbz()  # noqa: F841

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
    # assy.add(tracker_cone, loc=cq.Location(cq.Vector(-50, 0, 150)), name="tracker_cone")  # noqa: ERA001

    return assy


# eschaerer: needed for visualiser
def call_assembly_solids(id_number: int) -> tuple[list[cq.Solid], cq.Solid]:
    """
    Call the cadquery Assembly of the inner NBZ with given id-number from congif.assembly_map.yaml

    Args: id-number

    Returns:  tuple with list[inner solids] and outer solid
    """
    inner, outer, __ = call_assembly(id_number)
    return inner.toCompound().Solids(), outer


if __name__ == "__main__":
    from ocp_vscode import show

    start = time.time()
    print(max_assembly_id())
    end = time.time()

    print(end - start)

    start = time.time()

    show(call_assembly(0))

    inner, outer, vision = call_assembly(0)

    print(type(inner), type(outer), type(vision))
    print(type(inner.toCompound().Solids()), type(outer), type(vision.toCompound().Solids()))

    inner, outer = call_assembly_solids(0)

    print(type(inner), type(outer))

    print(max_assembly_id())

    # for i in range(max_assembly_id()):
    #     show(call_assembly(i))        # noqa: ERA001

    # for demo_id in range(100):
    #     show(call_assembly(demo_id))  # noqa: ERA001
    #     print(demo_id)  # noqa: ERA001

    # end = time.time()  # noqa: ERA001
    # print(end - start)  # noqa: ERA001
