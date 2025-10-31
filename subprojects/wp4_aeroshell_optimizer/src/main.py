from pathlib import Path
import torch
import yaml
from .models.sdf import SDFNetwork
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
        # This loads a sphere model while we redo the training loop and main structure
        model = torch.load(
            "/cluster/home/eschaerer/AISCO-code/subprojects/wp4_aeroshell_optimizer/archive/dummy_models/sdf_sphere_model_2025-10-24.pth"
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
