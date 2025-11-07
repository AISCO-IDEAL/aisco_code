from pathlib import Path

import torch
import wandb
import yaml

from .models.sdf import SDFNetwork
from .models.train import train_sdf_model
from .utils.utils import (
    check_cuda_health,
    initialize_cuda,
    load_config,
    load_pretrained_model,
    setup_training_environment,
)


def deep_update(base, updates):
    """
    Recursively updates base dict with updates dict.
    Supports dot notation keys: e.g., 'training_params.learning_rate'.
    """
    for key, value in updates.items():
        if "." in key:
            # Split and insert recursively
            parts = key.split(".")
            d = base
            for part in parts[:-1]:
                if part not in d or not isinstance(d[part], dict):
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = value
        elif isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


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
    config = load_config()

    track_run = config["wandb"].get("track_run", False)

    if track_run:
        run = wandb.init(
            # Set the wandb log directory
            dir="wp4_aeroshell_optimizer/data/wandb_logs",
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="aisco",
            # Set the wandb project where this run will be logged.
            project="Aeroshell Optimizer",
            # Track hyperparameters and run metadata.
            config=config,
        )
        config = deep_update(config, dict(wandb.config))

    # Load or initialize model
    load_pretrained = config.get("use_pretrained", False)

    if load_pretrained:
        # This loads a sphere model while we redo the training loop and main structure
        model = torch.load(
            "wp4_aeroshell_optimizer/data/test_models/sdf_sphere_model_2025-11-04.pth"
        )
    else:
        model = SDFNetwork()
    model.to("cuda")

    save_dir = config.get("model_dir", "")
    mesh_dir = config.get("mesh_dir", "")
    train_sdf_model(
        model, config["training_params"], save_dir, mesh_dir, track_run=track_run
    )

    if track_run:
        run.finish()

    print("WP4 Main.py executed successfully")


if __name__ == "__main__":
    main()
