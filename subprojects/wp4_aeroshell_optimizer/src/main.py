
import train
from sdf import SDFNetwork
from SDF_visualisation import visualize_loaded_model

def main(load_pretrained: bool=False):

    print("Hello world")
    visualize_loaded_model("/cluster/home/aalthaus/AISCO-code/subprojects/wp4_aeroshell_optimizer/src/DummySphere/sdf_sphere_model_2025-10-24.pth")


    return










if __name__ == "__main__":
    main(True)