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
    
    # print(config_path)
    with config_path.open("r") as file:
        assembly_map = yaml.safe_load(file) or {}

    params = assembly_map[assembly_nr]

    wheel = Wheels().create_wheel()
    seat = Seat().create_seat()

    assembly = cq.Assembly(name="inner nbz " + str(assembly_nr))

    assembly.add(
        seat,
        loc=cq.Location(
            cq.Vector(params["seat"]["x"], params["seat"]["y"], params["seat"]["z"])
        ),
        name="Seat",
    )

    for key in params:
        if key.startswith("wheel_"):
            assembly.add(
            wheel,
            loc=cq.Location(
                cq.Vector(params[key]["x"], params[key]["y"], params[key]["z"])
            ),
            name=key,
            )

    return assembly


if __name__ == "__main__":

    from ocp_vscode import show
    # for i in range(527):
    #     show(inner_nbz(i))
    print("showing")
    start = time.time()
    #show([inner_nbz(random.randint(0, 2337)) for i in range(1)])
    show(inner_nbz(10))
    end = time.time()
    print(end-start)
