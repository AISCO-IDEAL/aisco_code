from components import nbz
from ocp_vscode import show
from utils.distributions import visualizeResults
from utils.reg_imp import get_regulation

# testing regulation import
wheel_dim = get_regulation("wheels.tire_radius")
print(wheel_dim)

if __name__ == "__main__":  # only run when executing this file directly
    outer_nbz = nbz.outer_nbz2()

    while True:
        userinput = input("Visualize NBZ or Triangle or stop? (n/t/s): ")
        if userinput.lower() == "n":
            userinput2 = input("Visualize innerNBZ (WIP), outerNBZ or both? (i/o/b): ")
            if userinput2.lower() == "i":
                inner_nbz = nbz.inner_nbz2()
                show(inner_nbz)
                print(type(inner_nbz))
            elif userinput2.lower() == "o":
                show(outer_nbz)
                print(type(outer_nbz))
            else:
                print("Invalid input. Please enter 'i', 'o' or 'b'.")
        elif userinput.lower() == "t":
            userinput3 = input("Visualize symmetric triangle or distribution? (t/d): ")
            if userinput3.lower() == "t":
                triangle = nbz.inner_nbz2_sym()
                show(triangle)
                print(type(triangle))
            elif userinput3.lower() == "d":
                visualizeResults()
            else:
                print("Invalid input. Please enter 't' or 'd'.")
        elif userinput.lower() == "s":
            print("Visualization stopped.")
            break
        else:
            print("Invalid input. Please enter 'n', 't' or 's'.")
