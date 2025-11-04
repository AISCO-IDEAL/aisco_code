"""
Run this file by running the two follwoing commands in the terminal:

cd /cluster/home/username/AISCO-code/subprojects/wp4_aeroshell_optimizer

python -m src.main
"""

import sys
import traceback
from pathlib import Path

import torch
import yaml


########################################################
# Custom error handler to flush outputs on exceptions
def excepthook(type, value, tb):
    traceback.print_exception(type, value, tb)
    sys.stdout.flush()
    sys.stderr.flush()


sys.excepthook = excepthook
print("DEBUG: Skript gestartet. Versuche Imports...")
########################################################
# WP4 Imports
from .models.sdf import SDFNetwork  # one dot as in subfolder of src and main is in src
from .models.train import train_sdf_model
from .utils.utils import (
    check_cuda_health,
    initialize_cuda,
    load_pretrained_model,
    setup_training_environment,
)

metadata = {
    "Username": "eschaerer",
}


def main():
    print("DEBUG: Starte main()")
    print("DEBUG: Führe CUDA-Health-Check durch...")
    check_cuda_health()
    print("DEBUG: Initialisiere CUDA...")
    device = initialize_cuda()
    print("DEBUG: Lade Konfiguration...")
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

    # Get the directory of the current script
    script_dir = Path(__file__).resolve().parent
    # Define the base directory of the project
    base_dir = script_dir.parent.parent.parent  # Adjust based on your project structure

    # Load configuration
    config_path = (
        base_dir
        / "subprojects"
        / "wp4_aeroshell_optimizer"
        / "src"
        / "config"
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
        model_path = (
            base_dir
            / "subprojects"
            / "wp4_aeroshell_optimizer"
            / "archive"
            / "dummy_models"
            / "sdf_sphere_model_2025-10-24.pth"
        )
        model = torch.load(model_path)
    else:
        model = SDFNetwork()
    model.to("cuda")

    # Define save and mesh directories relative to the project root
    save_dir = (
        base_dir
        / "subprojects"
        / "wp4_aeroshell_optimizer"
        / config.get("model_dir", "")
    )
    mesh_dir = (
        base_dir
        / "subprojects"
        / "wp4_aeroshell_optimizer"
        / config.get("mesh_dir", "")
    )

    # Train the model
    train_sdf_model(model, config["training_params"], save_dir, mesh_dir)
    print("WP4 Main.py executed successfully")


if __name__ == "__main__":
    main()
