import cadquery as cq
from ocp_vscode import show

from subprojects.wp2_assembler.version3.components import nbz
from subprojects.wp2_assembler.version3.components.vision import tracker_box, vision_box
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
        inner_nbz, outer_nbz, _ = nbz.call_assembly(id_number)  # kdiehl: ignoring vision NBZ, as its not used later on
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

3
if __name__ == "__main__":  # only run when executing this file directly
    inner_nbz: cq.Assembly
    outer_nbz: cq.Solid
    vision_nbz: cq.Assembly
    while True:
        userinput = input(
            "Visualize inner NBZ, outer NBZ, both, v3, multiple (both inner and outer), vision NBZ, tracker box, generate or stop? (i/o/b/3/v/t/g/s): "
        )
        userinput = userinput.lower().strip()  # makes it case & space insensitive

        if userinput == "s":  # Beenden sofort prüfen
            print("Visualization stopped.")
            break

        if userinput in ["i", "o", "b", "3"]:
            id_str = input("Which id-number to visualize? ")
            if not id_str.isdigit():
                print("Invalid input. Please enter a valid id-number.")
                continue
            id_number = int(id_str)
            inner_nbz, outer_nbz, vision_nbz = nbz.call_assembly(id_number)

            if userinput == "i":
                show(inner_nbz)
                print(type(inner_nbz))
            elif userinput == "o":
                show(outer_nbz)
                print(type(outer_nbz))
            elif userinput == "b":
                assembly = cq.Assembly(name=f"NBZ_{id_number}")
                assembly.add(outer_nbz, name="outerNBZ", color=cq.Color(1, 0, 0, 0.3))
                assembly.add(inner_nbz, name="innerNBZ", color=cq.Color(0, 0, 1, 0.3))
                show(assembly)
                print(type(inner_nbz))
                print(type(outer_nbz))
                print(type(assembly))
            elif userinput == "3":
                assembly = cq.Assembly(name="assembly_v3")
                assembly.add(inner_nbz, name="innerNBZ", color=cq.Color(0, 0, 1, 0.3))
                assembly.add(outer_nbz, name="outerNBZ", color=cq.Color(1, 1, 0, 0.3))
                assembly.add(vision_nbz, name="visionNBZ", color=cq.Color(0, 1, 0, 0.3))
                show(assembly)
        elif userinput == "m":
            rng = input("Range as 'start-end': ")
            try:
                s, e = map(int, rng.split("-"))
                if s > 0 and e > s:
                    visualize_multiple(s, e)
                else:
                    print("Invalid range values.")
            except ValueError:
                print("Format error. Use e.g. '1-10'.")

        elif userinput == "v":
            show(vision_box.vision_driv_nbz())

        elif userinput == "t":
            show(tracker_box.tracker_box_nbz())

        elif userinput == "g":
            number_asm = input("How many assemblies are to be generated?")
            if number_asm.isdigit():
                n_assemblies = int(number_asm)
                for j in range(n_assemblies):
                    print(f"Generating assembly {j + 1} of {n_assemblies}...")
                    nbz.generate_assembly()
                print(f"{n_assemblies} assemblies generated and saved.")
                # show in 2 second intervals
            else:
                print("Invalid input. Please enter a valid number.")
                continue
