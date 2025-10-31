from pathlib import Path

import torch
import yaml
from sdf import SDFNetwork
from train import train_sdf_model
from utils import (
    check_cuda_health,
    initialize_cuda,
    load_pretrained_model,
    setup_training_environment,
)

# Hyperparameters - consider moving to config file
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

# General information - consider moving to config file
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
    """Main execution function for WP4 aeroshell optimizer."""

    print("=" * 60)
    print("WP4 Aeroshell Optimizer - Starting")
    print("=" * 60)

    # Perform CUDA health check (optional, remove if too verbose)
    check_cuda_health()

    # Initialize CUDA context early - CRITICAL for avoiding cuBLAS warnings
    device = initialize_cuda()

    # Setup training environment
    setup_training_environment(device)

    # Load configuration
    config_path = (
        Path.home()
        / "AISCO-code"
        / "subprojects"
        / "wp4_aeroshell_optimizer"
        / "src"
        / "wp4_config.yaml"
    )

    try:
        with config_path.open() as f:
            config = yaml.safe_load(f)
        print(f"✓ Configuration loaded from {config_path}")
    except FileNotFoundError:
        print(f"✗ Configuration file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"✗ Error parsing configuration file: {e}")
        return

    # Load or initialize model
    load_pretrained = config.get("use_pretrained", False)

    if load_pretrained:
        # This loads a sphere model while we redo the training loop and main structure
        model = torch.load(
            "AISCO-code/subprojects/wp4_aeroshell_optimizer/src/DummySphere/sdf_sphere_model_2025-10-24.pth"
        )
    else:
        model = SDFNetwork()
    model.to("cuda")

    save_dir = config.get("model_dir", "")
    mesh_dir = config.get("mesh_dir", "")
    train_sdf_model(model, config["training_params"], save_dir, mesh_dir)
    print("WP4 Main.py executed successfully")


if __name__ == "__main__":
    main()
