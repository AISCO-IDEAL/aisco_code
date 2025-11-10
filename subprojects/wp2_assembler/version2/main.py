from components import nbz
from ocp_vscode import show

if __name__ == "__main__":  # only run when executing this file directly
    while True:
        userinput = input("Visualize innerNBZ (WIP), outerNBZ or stop? (i/o/s): ")
        if userinput.lower() == "i":
            inner_nbz = nbz.inner_nbz2_sym()
            show(inner_nbz)
            print(type(inner_nbz))
        elif userinput.lower() == "o":
            outer_nbz = nbz.outer_nbz2()
            show(outer_nbz)
            print(type(outer_nbz))
        elif userinput.lower() == "s":
            print("Visualization stopped.")
            break
        else:
            print("Invalid input. Please enter 'i', 'o' or 's'.")
