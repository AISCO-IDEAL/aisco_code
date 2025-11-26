import cadquery as cq
from ocp_vscode import show

from subprojects.wp2_assembler.version3.components import nbz
from subprojects.wp2_assembler.version3.components.inner import tracker_box, vision_box
from subprojects.wp2_assembler.version3.utils.reg_imp import get_regulation


def visualize_multiple(lower_id: int, upper_id: int):
    """
    Visualize multiple assemblies by adding them all into a single assembly by expanding in the y-direction
    Args:
        lower_id (int): The starting id-number of the assemblies to visualize.
        upper_id (int): The ending id-number of the assemblies to visualize.
    """
    # check if all assemblies exist
    grouped = cq.Assembly(name="groupedNBZ")
    for count, id_number in enumerate(range(lower_id, upper_id + 1)):
        inner_nbz, outer_nbz = nbz.call_assembly(id_number)
        assembly = cq.Assembly(name=f"NBZ_{id_number}")
        assembly.add(outer_nbz, name="outerNBZ", color=cq.Color(1, 0, 0, 0.3))
        assembly.add(inner_nbz, name="innerNBZ", color=cq.Color(0, 0, 1, 0.3))
        show(assembly)
        grouped.add(
            assembly,
            name=f"assembly_{id_number}",
            # after each 10th assembly start a new row
            loc=cq.Location(
                cq.Vector(
                    0,
                    (count % 10) * get_regulation("no_build_zones.outer_prism.width"),
                    (count // 10) * get_regulation("no_build_zones.outer_prism.height"),
                ),
            ),
        )
        print(f"Added assembly with id-number: {id_number}")
    show(grouped)


if __name__ == "__main__":  # only run when executing this file directly
    while True:
        userinput = input(
            "Visualize inner NBZ, outer NBZ, both, multiple (both inner and outer), vision NBZ, tracker box, generate or stop? (i/o/b/v/t/g/s): "
        )
        if userinput.lower() == "i":
            userinput = input("Which id-number to visualize? ")
            if userinput.isdigit():
                id_number = int(userinput)
                inner_nbz, _ = nbz.call_assembly(id_number)
            else:
                print("Invalid input. Please enter a valid id-number.")
                continue
            show(inner_nbz)
            print(type(inner_nbz))

        elif userinput.lower() == "o":
            userinput = input("Which id-number to visualize? ")
            if userinput.isdigit():
                id_number = int(userinput)
                _, outer_nbz = nbz.call_assembly(id_number)
            else:
                print("Invalid input. Please enter a valid id-number.")
                continue
            show(outer_nbz)
            print(type(outer_nbz))

        elif userinput.lower() == "b":
            userinput = input("Which id-number to visualize? ")
            if userinput.isdigit():
                id_number = int(userinput)
            else:
                print("Invalid input. Please enter a valid id-number.")
                continue
            inner_nbz, outer_nbz = nbz.call_assembly(id_number)
            assembly = cq.Assembly(name=f"NBZ_{id_number}")
            assembly.add(outer_nbz, name="outerNBZ", color=cq.Color(1, 0, 0, 0.3))
            assembly.add(inner_nbz, name="innerNBZ", color=cq.Color(0, 0, 1, 0.3))
            show(assembly)
            print(type(inner_nbz))
            print(type(outer_nbz))
            print(type(assembly))

        elif userinput.lower() == "m":
            userinput = input(
                "Which assemblies are to be visualized? (ranged as 'start-end') "
            )
            userinput_split = userinput.split("-")
            if int(userinput_split[0]) > 0:
                lower_id = int(userinput_split[0])
            else:
                print("Invalid input. Please enter a valid range.")
                continue
            if int(userinput_split[1]) > lower_id:
                upper_id = int(userinput_split[1])
            else:
                print("Invalid input. Please enter a valid range.")
                continue
            visualize_multiple(lower_id, upper_id)

        elif userinput.lower() == "v":
            vision_nbz = vision_box.vision_nbz()
            show(vision_nbz)

        elif userinput.lower() == "t":
            tracker_box = tracker_box.tracker_box()
            show(tracker_box)

        elif userinput.lower() == "g":
            userinput = input("How many assemblies are to be generated?")
            if userinput.isdigit():
                n_assemblies = int(userinput)
                for j in range(n_assemblies):
                    print(f"Generating assembly {j + 1} of {n_assemblies}...")
                    nbz.generate_assembly()
                print(f"{n_assemblies} assemblies generated and saved.")
                # show in 2 second intervals
            else:
                print("Invalid input. Please enter a valid number.")
                continue

        elif userinput.lower() == "s":
            print("Visualization stopped.")
            break
        else:
            print("Invalid input. Please enter 'i', 'o', 'b' or 's'.")
