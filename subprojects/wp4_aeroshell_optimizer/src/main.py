from pathlib import Path
import torch
import yaml
from sdf import SDFNetwork
from train import train_sdf_model
from utils import initialize_cuda, load_pretrained_model, setup_training_environment, check_cuda_health

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
    config_path = Path.home() / "AISCO-code" / "subprojects" / "wp4_aeroshell_optimizer" / "src" / "wp4_config.yaml"
    
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
        model_path = Path.home() / "AISCO-code" / "subprojects" / "wp4_aeroshell_optimizer" / "archive" / "dummy_models" / "sdf_sphere_model_2025-10-24.pth"
        print(f"Loading pretrained model from {model_path}")
        model = load_pretrained_model(model_path, device, SDFNetwork)
    else:
        print("Initializing new SDFNetwork model")
        model = SDFNetwork()
        model = model.to(device)
    
    # Ensure model is in training mode
    model.train()
    
    # Perform a dummy forward pass to fully initialize CUDA context
    # This helps prevent the cuBLAS warning during actual training
    if device == 'cuda':
        print("Warming up CUDA context...")
        with torch.no_grad():
            dummy_input = torch.randn(10, 3, device=device)
            _ = model(dummy_input)
            torch.cuda.synchronize()
        print("✓ CUDA context warmed up")
    
    # Get directories from config
    save_dir = config.get("model_dir", "")
    mesh_dir = config.get("mesh_dir", "")
    
    # Ensure directories exist
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print(f"✓ Model directory: {save_dir}")
    if mesh_dir:
        Path(mesh_dir).mkdir(parents=True, exist_ok=True)
        print(f"✓ Mesh directory: {mesh_dir}")
    
    # Train the model
    print("\n" + "=" * 60)
    print("Starting SDF model training...")
    print("=" * 60 + "\n")
    
    try:
        train_sdf_model(
            model,
            config["training_params"],
            save_dir,
            mesh_dir
        )
        print("\n" + "=" * 60)
        print("✓ WP4 Main.py executed successfully")
        print("=" * 60)
    except KeyError as e:
        print(f"✗ Missing configuration key: {e}")
        print("Please check your wp4_config.yaml file")
    except Exception as e:
        print(f"✗ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()