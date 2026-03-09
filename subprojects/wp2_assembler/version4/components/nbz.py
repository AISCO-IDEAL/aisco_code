import time
from pathlib import Path

import cadquery as cq
import yaml

from subprojects.wp2_assembler.version4.components.inner.seat import Seat
from subprojects.wp2_assembler.version4.components.inner.wheels import Wheels


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
    """
    Call the cadquery Assembly of the inner NBZ with given id-number from congif.assembly_map.yaml
    Args: id-number
    Returns: cadquery Assembly, outer NBZ as cq.Solid, vision NBZ as cadquery Assembly
    """
    config_path = Path(__file__).parent.parent / "config" / "assembly_map.yaml"

    # Read existing data
    with config_path.open("r") as file:
        assembly_map = yaml.safe_load(file) or {}

    # YAML loads numeric keys as integers, not strings
    if id_number not in assembly_map:
        raise ValueError(f"ID {id_number} not found in assembly map.")
    params = assembly_map[id_number]
    seat_pos_z = get_regulation("seat.position.z")

    box_width = get_regulation("no_build_zones.outer_prism.width")
    box_height = get_regulation("no_build_zones.outer_prism.height")

    outer = outer_nbz(params["wheel_base_length"] * 2.2, box_width, box_height)


    box, vision = vision_nbz(params["seat_position"], seat_pos_z, params["wheel_base_length"], get_box=True)
    assembly.add(box)

    outer_with_ramps = outer_nbz_ramps(assembly, outer)

    return inner_nbz(id_number), outer_with_ramps, vision # inner_nbz returns an assembly

if __name__ == "__main__":
    from ocp_vscode import show
    from subprojects.wp2_assembler.version4.components.outer.outer_nbz import outer_nbz
    # for i in range(527):
    #     show(inner_nbz(i))  # noqa: ERA001
    print("showing")
    start = time.time()

    # demo_id = 1
    # show([inner_nbz(random.randint(0, 2337)) for i in range(1)])  # noqa: ERA001
    # show(inner_nbz(demo_id), outer_nbz(demo_id))
    # show(inner_nbz(demo_id), outer_nbz(demo_id))
    # for demo_id in range(100):
    #    show(inner_nbz(demo_id), outer_nbz(demo_id))
    demo_id = 0
    while True:
        show(inner_nbz(demo_id), outer_nbz(demo_id))
        print(demo_id)
        demo_id = demo_id +1

    end = time.time()
    print(end - start)
