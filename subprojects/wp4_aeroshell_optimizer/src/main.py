from pathlib import Path

import kaolin_test
import torch
import yaml
from sdf import SDFNetwork
from SDF_visualisation import visualize_loaded_model
from train import train_sdf_model
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

def main():
    # 1. Load settings from config:
    with Path.open('AISCO-code/subprojects/wp4_aeroshell_optimizer/src/wp4_config.yaml') as f:
        config = yaml.safe_load(f)


    load_pretrained = config["use_pretrained"]

    if load_pretrained:
        #This loads a model while we redo the training loop and main structure
        
        model=torch.load("AISCO-code/subprojects/wp4_aeroshell_optimizer/src/DummySphere/sdf_sphere_model_2025-10-24.pth")
    else:
        model=SDFNetwork()
    model.to('cuda')

    train_sdf_model(model,config)
    
    





if __name__ == "__main__":
    main()