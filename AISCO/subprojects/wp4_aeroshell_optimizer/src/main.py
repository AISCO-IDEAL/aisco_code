import datetime
import torch
import os
import numpy as np
from sdf import SDFNetwork
from train import train_sdf_model  # Import the training function

username = "aalthaus"  # Replace with your actual username

def main(load_pretrained_model: bool = False):
    # Paths and config
    load_dir = f"/cluster/home/{username}/AISCO-code/subprojects/WP4-Aeroshell/SDF_Prototype/dummy_models"
    model_path = os.path.join(load_dir, 'sdf_sphere_model_2025-10-24.pth')
    save_dir = f"/cluster/home/{username}/AISCO-code/subprojects/WP4-Aeroshell/SDF_Prototype/main_models"
    stl_file_path = f'/cluster/home/{username}/AISCO-code/subprojects/WP2-Assembler/shells/shell0.stl'

    # Config for training
    config = {
        "learning_rate": 1e-4,
        "num_epochs": 50,
        "batch_size": 1024,
    }

    # Initialize or load model
    if load_pretrained_model:
        model = torch.load(model_path)
    else:
        model = SDFNetwork()

    # Train the model
    train_sdf_model(
        model=model,
        config=config,
        point_cloud=None,  # Will be generated in train_sdf_model if stl_file_path is provided
        load_pretrained_model=load_pretrained_model,
        model_path=model_path if load_pretrained_model else None,
        save_dir=save_dir,
        stl_file_path=stl_file_path,
    )

if __name__ == "__main__":
    main(load_pretrained_model=True)
