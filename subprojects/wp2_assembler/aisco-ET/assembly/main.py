import cadquery as cq
from cadquery import Assembly, Color
from components import nbz

try:
    from ocp_vscode import show
except ImportError:
    print("Warning: ocp_vscode not available. Visualization will not work.")
    show = None

if __name__ == "__main__":  # only run when executing this file directly
    # Create inner NBZ
    inner_nbz = nbz.innerNBZ()
    print("Inner no-build-zone created")

    # Create outer NBZ
    outer_nbz = nbz.outerNBZ()
    print("Outer no-build-zone created")

    # Create assembly of both NBZs
    assembled_car = Assembly(outer_nbz, name="outerNBZ", color=Color(1, 0, 0, 0.3))
    assembled_car.add(inner_nbz, name="innerNBZ", color=Color(0, 0, 1))
    print("Assembly created")

    # Prompt to visualize the shapes
    while True:
        userinput = input("Visualize innerNBZ and/or outerNBZ? (y/n): ")
        if userinput.lower() == "y":
            if show is None:
                print("Visualization not available. ocp_vscode is not installed.")
                break
            userinput2 = input("Visualize innerNBZ, outerNBZ or both? (i/o/b): ")
            if userinput2.lower() == "i":
                show(inner_nbz)
                print(type(inner_nbz))
            elif userinput2.lower() == "o":
                show(outer_nbz)
                print(type(outer_nbz))
            elif userinput2.lower() == "b":
                show(assembled_car)
                print(type(assembled_car))
            else:
                print("Invalid input. Please enter 'i' or 'o'.")
        elif userinput.lower() == "n":
            print("Visualization stopped.")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
