import ocp_vscode as ocp
from ocp_vscode.comms import set_port

from subprojects.wp2_assembler.version4.components.nbz import call_assembly


def main_test():
    set_port(49152)
    inner_assembly, __, __ = call_assembly(2)
    # get coordinates of wheel interfaces (assuming at least 3 wheels exist)
    # inner_solids[0] is the seat; inner_solids[1+] are wheel solids
    inner_solids = inner_assembly.toCompound().Solids()
    ocp.show(inner_assembly)
    wheels: list[list] = [[], []]  # list of [solid, coordinates]
    for solid in inner_solids[1:]:
        wheels[0].append(solid)
        # get coordinates of each wheel
        wheels[1].append(solid.BoundingBox().center)

    print(wheels)


if __name__ == "__main__":
    main_test()
