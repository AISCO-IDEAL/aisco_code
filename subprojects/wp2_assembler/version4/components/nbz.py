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


def call_assembly(id_number: int) -> tuple[list[cq.Solid], cq.Solid, list[cq.Solid]]:

    if id_number>max_assembly_id():
        raise ValueError("ValueError in wp2, version4, nbz. id_number of assembly bigger than max.")

    return inner_nbz(id_number).toCompound().Solids(), outer_nbz(id_number), vision_nbz(id_number).toCompound().Solids()

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
    inner_bounding_box = cq.Compound.makeCompound(inner).BoundingBox()

    return (
        (inner_bounding_box.xmin,
         inner_bounding_box.ymin,
         inner_bounding_box.zmin
        ),

        (inner_bounding_box.xmax,
         inner_bounding_box.ymax,
         inner_bounding_box.zmax
        )
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

    assy.add(driver_cone, loc=cq.Location(cq.Vector(params['seat']['x'], params['seat']['y'], params['seat']['z'] + get_regulation('no_build_zones.vision_box.eye_level_z'))), name="driver_cone")
   # assy.add(tracker_cone, loc=cq.Location(cq.Vector(-50, 0, 150)), name="tracker_cone")  # noqa: ERA001

    return assy


if __name__ == "__main__":
    from ocp_vscode import show

    start = time.time()
    print(max_assembly_id())
    end = time.time()

    print(end-start)

    start = time.time()

    show(call_assembly(0))

    # for demo_id in range(100):
    #     show(call_assembly(demo_id))  # noqa: ERA001
    #     print(demo_id)  # noqa: ERA001

    # end = time.time()  # noqa: ERA001
    # print(end - start)  # noqa: ERA001
