import time
from pathlib import Path

import cadquery as cq
import yaml
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.gp import gp_Pnt, gp_Trsf
from OCP.StlAPI import StlAPI_Reader
from OCP.TopoDS import TopoDS_Shape

from subprojects.wp2_assembler.version4.components.inner.seat import Seat
from subprojects.wp2_assembler.version4.components.inner.wheels import Wheels
from subprojects.wp2_assembler.version4.components.outer.outer_nbz import outer_nbz
from subprojects.wp2_assembler.version4.components.vision import tracker_box, vision_box
from subprojects.wp2_assembler.version4.utils.reg_imp import get_regulation

_LOSSES_V0_CONFIG_PATH = Path("subprojects/wp4_aeroshell_optimizer/v2/configs/losses_v0_config.yaml")
_OCCUPANT_CELL_STL_PATH = Path(__file__).parent / "inner" / "occupant_cell.stl"


def _get_seat_type() -> str:
    if not _LOSSES_V0_CONFIG_PATH.exists():
        return "regulations"
    with _LOSSES_V0_CONFIG_PATH.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("bb_config", {}).get("seat_type", "regulations")


_OCCUPANT_CELL_SCALE = 0.8  # uniform scale to roughly match regulation seat dimensions


def _load_occupant_cell(seat_x: float, seat_y: float, seat_z: float) -> cq.Shape:
    raw = TopoDS_Shape()
    StlAPI_Reader().Read(raw, str(_OCCUPANT_CELL_STL_PATH))
    trsf = gp_Trsf()
    trsf.SetScale(gp_Pnt(0, 0, 0), _OCCUPANT_CELL_SCALE)
    s = cq.Shape(BRepBuilderAPI_Transform(raw, trsf, True).Shape())
    bb = s.BoundingBox()
    # Align back-bottom-center of the bounding box with the assembly-map seat position.
    dx = seat_x - bb.xmin
    dy = seat_y - (bb.ymin + bb.ymax) / 2
    dz = seat_z - bb.zmin
    return s.translate(cq.Vector(dx, dy, dz))


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

    assembly_map_path = Path(__file__).parent.parent / "config" / "assembly_map.yaml"
    with assembly_map_path.open("r") as f:
        assembly_map = yaml.safe_load(f) or {}
    seat_params = assembly_map[id_number]["seat"]

    inner_solids = inner_nbz(id_number).toCompound().Solids()

    if len(inner_solids) < 1:
        raise ValueError(f"No inner solids found for assembly {id_number}.")

    seat_solid = (
        _load_occupant_cell(seat_params["x"], seat_params["y"], seat_params["z"])
        if _get_seat_type() == "occupant_cell"
        else inner_solids[0]
    )
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

    # Recenter both inner and outer to x=0, y=0 using the canonical training frame offset.
    # See get_training_frame_offset() for the definition and rationale.
    # Note: z-clipping above does not affect the x/y centroid, so computing the offset
    # from the unclipped outer_nbz (inside get_training_frame_offset) gives identical cx, cy.
    cx, cy, cz = get_training_frame_offset(id_number)
    if abs(cx) > 0.001 or abs(cy) > 0.001 or abs(cz) > 0.001:
        offset = cq.Vector(-cx, -cy, -cz)
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


def get_training_frame_offset(id_number: int) -> tuple[float, float, float]:
    """
    Returns (cx, cy, cz) — the shift that call_assembly() subtracts from all raw assembly
    coordinates to align geometry with the SDF training domain.

    Any loss or visualiser that needs assembly-relative positions (e.g. wheel contact
    points) must subtract this offset from raw assembly_map coordinates before evaluating
    the model. Using raw coordinates without this correction silently places samples in
    the wrong location and the model will produce geometry artefacts at those points.

    Training domain coords:  x_train = x_raw - cx,  y_train = y_raw - cy,  z_train = z_raw - cz

    Currently cz = 0 (ground plane kept at z = 0), but it is included here so that any
    future z-shift from WP2 only requires updating this function.
    """
    raw_bbox = outer_nbz(id_number).BoundingBox()
    cx = (raw_bbox.xmin + raw_bbox.xmax) / 2.0
    cy = (raw_bbox.ymin + raw_bbox.ymax) / 2.0
    cz = 0.0
    return cx, cy, cz


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

    for i in range(max_assembly_id()):
        show(call_assembly(i))

    for demo_id in range(100):
        show(call_assembly(demo_id))
        print(demo_id)

    end = time.time()
    print(end - start)
