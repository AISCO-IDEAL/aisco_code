import kaolin_test
from SDF_visualisation import visualize_loaded_model
from utils import filenamer

# eschaerer, just a proposal for the informative .json
hyperparams = {
    "learning_rate": 0.0001,
    "batch_size": 4096,
    "num_epochs": 1000,
    "model_capacity": 256,
    "num_layers": 8,
    "skip_connections": [4],
    "weight_decay": 0.0,
    "activation": "relu",
    "optimizer": "adam",
    "scheduler_step_size": 200,
    "scheduler_gamma": 0.5,
}

# eschaerer, just a proposal for the informative .json
general_info = {
    "Model": "SDFNetwork",
    "Model_Expl": "an SDF Network with torch Neural Network",
    "Pre": "a sphere as initial shape, datatype: SDF",
    "Post": "a PLY file representing the optimized aeroshell shape",
}
metadata = {
        "Username": "eschaerer",
    }

def main(load_pretrained_model: bool = False):
    
    filename = filenamer(hyperparams, general_info, metadata)  # This is for the JSON metadata
    model_path = "/cluster/home/eschaerer/AISCO-code/subprojects/wp4_aeroshell_optimizer/archive/dummy_models/sdf_sphere_model_2025-10-20.pth"
    sphere = visualize_loaded_model(path=model_path)
    sphere.to('cuda')
    vertices, faces = kaolin_test.get_diff_mesh(sphere, grid_res=64)

    # Use the model's base name for the mesh filename
    mesh_filename = filename.replace('.pth', '_mesh.ply')
    kaolin_test.flexicubes_to_PLY(vertices, faces, save_mesh=True, filename=mesh_filename)

if __name__ == "__main__":
    main(True)