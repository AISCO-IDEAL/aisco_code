# utils.py

import json
import os
import time
from pathlib import Path
import kaolin_test
import numpy as np
import trimesh
from sdf import SDFNetwork
from skimage import measure
import torch
from typing import Optional


def initialize_cuda() -> str:
    """
    Initialize CUDA context properly and check availability.
    This creates a CUDA context early to avoid cuBLAS warnings.
    
    Returns:
        device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        try:
            # Explicitly set the device and create a tensor to initialize context
            device = 'cuda'
            torch.cuda.set_device(0)  # Set primary GPU
            
            # Create a dummy tensor operation to force CUDA context initialization
            dummy = torch.zeros(1, device=device)
            dummy = dummy + 1  # Simple operation to establish context
            del dummy  # Clean up
            
            # Ensure cuBLAS handle is initialized
            torch.cuda.synchronize()
            
            print(f"CUDA initialized successfully")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
        except Exception as e:
            print(f"CUDA initialization failed: {e}")
            print("Falling back to CPU")
            device = 'cpu'
    else:
        device = 'cpu'
        print("CUDA not available: Using CPU")
    
    return device


def load_pretrained_model(model_path: Path, device: str = 'cuda', model_class=None) -> torch.nn.Module:
    """
    Safely load a pretrained model with proper error handling.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on ('cuda' or 'cpu')
        model_class: Model class to instantiate if loading fails (optional)
    
    Returns:
        Loaded model
    """
    if not model_path.exists():
        print(f"Model file not found at {model_path}")
        if model_class:
            print("Creating new model instance")
            return model_class()
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load to CPU first to avoid CUDA memory issues
        print(f"Loading model from {model_path}...")
        
        # Try loading with weights_only first (safer)
        try:
            state_dict = torch.load(
                model_path,
                map_location='cpu',
                weights_only=True
            )
            if model_class:
                model = model_class()
                model.load_state_dict(state_dict)
            else:
                # If no model class provided, assume state_dict is the full model
                model = state_dict
        except Exception:
            # Fall back to full model loading
            print("Loading full model (weights_only=False)...")
            model = torch.load(
                model_path,
                map_location='cpu',
                weights_only=False
            )
        
        # Move to target device
        model = model.to(device)
        model.eval()  # Set to evaluation mode initially
        
        print(f"Successfully loaded pretrained model")
        return model
        
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        if model_class:
            print("Creating new model instance")
            return model_class()
        raise


def setup_training_environment(device: str):
    """
    Configure PyTorch training environment settings.
    
    Args:
        device: Device string ('cuda' or 'cpu')
    """
    if device == 'cuda':
        # Enable cudnn benchmarking for performance
        torch.backends.cudnn.benchmark = True
        
        # Set deterministic operations if needed (slower but reproducible)
        # torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        print("Training environment configured for CUDA")
    else:
        print("Training environment configured for CPU")


def check_cuda_health():
    """
    Perform a health check on CUDA installation.
    Useful for debugging CUDA-related issues.
    """
    print("\n=== CUDA Health Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # Test basic operations
        try:
            x = torch.randn(100, 100, device='cuda')
            y = torch.randn(100, 100, device='cuda')
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            print("\n✓ Basic CUDA operations working")
        except Exception as e:
            print(f"\n✗ CUDA operation failed: {e}")
    
    print("=========================\n")

def save_sdf2mesh(model, 
    grid_size: int =128,
    threshold: float = 0.0,
    mesh_dir: str = "AISCO-code/subprojects/wp4_aeroshell_optimizer/data",
    filename: str="sdf_mesh.ply",
    device: str='cuda'):

    """
    x = torch.linspace(-8, 8, grid_size, device= device)
    y = torch.linspace(-8, 8, grid_size, device= device)
    z = torch.linspace(-8, 8, grid_size, device= device)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
    #is this line necessary?
    #grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)

    with torch.no_grad():  # Disable gradient computation
        sdf_values = model(grid_points)[0] #.cpu().numpy().reshape((grid_size, grid_size, grid_size))

    verts, faces, normals, _ = measure.marching_cubes(sdf_values, level=threshold)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    """



    verts,faces = kaolin_test.get_diff_mesh(model,grid_res=grid_size)
    
    if mesh_dir:
        export_path= mesh_dir
        export_path=os.path.join(export_path,filename)
        mesh=kaolin_test.flexicubes_to_PLY(verts,faces,filename=filename)
        print(f"Mesh saved to {export_path}.")
        return mesh
    return

def load_model(model_path: str) -> SDFNetwork:
    model = torch.load(model_path)
    return model

def filenamer(hyperparams: dict, general_info: dict, metadata: dict) -> str:
    # Create a filename based on current timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"{metadata['Username']}_{general_info['Model']}_{timestamp}.json"

    # Define the full path to save the JSON file
    save_path = Path(f"/cluster/scratch/{metadata['Username']}/models")
    full_path = save_path / filename

    # Create the JSON structure
    json_data = {
        "Metadata": {
            "Username": metadata['Username'],
            "Timestamp": timestamp,
            "Filename": filename
        },
        "General Information": general_info,
        "Hyperparameters": hyperparams
    }

    # Save the JSON file
    full_path.write_text(json.dumps(json_data, indent=4))

    return filename
