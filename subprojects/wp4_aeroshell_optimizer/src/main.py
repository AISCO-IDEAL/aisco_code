
import train
from sdf import SDFNetwork
from SDF_visualisation import visualize_loaded_model
import kaolin_test
def main(load_pretrained: bool=False):

    #temporary workspace as long as branch chaos persists
    sphere=visualize_loaded_model("/cluster/home/aalthaus/AISCO-code/subprojects/wp4_aeroshell_optimizer/src/DummySphere/sdf_sphere_model_2025-10-24.pth")
    sphere.to('cuda')
    vertices, faces = kaolin_test.get_diff_mesh(sphere,grid_res=64)
    print("Hello world")
    kaolin_test.flexicubes_to_PLY(vertices,faces,True)

    

    return




if __name__ == "__main__":
    main(True)