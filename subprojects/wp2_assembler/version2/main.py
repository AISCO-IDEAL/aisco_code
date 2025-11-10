import cadquery as cq
from components import nbz
from ocp_vscode import show

if __name__ == "__main__":  # only run when executing this file directly
    while True:
        userinput = input("Visualize innerNBZ (WIP), outerNBZ or stop? (i/o/b/s): ")
        if userinput.lower() == "i":
            inner_nbz = nbz.inner_nbz2_sym()
            show(inner_nbz)
            print(type(inner_nbz))
        elif userinput.lower() == "o":
            outer_nbz = nbz.outer_nbz()
            show(outer_nbz)
            print(type(outer_nbz))
        elif userinput.lower() == "b":
            inner_nbz = nbz.inner_nbz2_sym()
            outer_nbz = nbz.outer_nbz()
            assembly = cq.Assembly(
                outer_nbz, name="outerNBZ", color=cq.Color(1, 0, 0, 0.3)
            )
            assembly.add(inner_nbz, name="innerNBZ", color=cq.Color(0, 0, 1, 0.3))
            show(assembly)
            print(type(assembly))
        elif userinput.lower() == "s":
            print("Visualization stopped.")
            break
        else:
            print("Invalid input. Please enter 'i', 'o', 'b' or 's'.")
