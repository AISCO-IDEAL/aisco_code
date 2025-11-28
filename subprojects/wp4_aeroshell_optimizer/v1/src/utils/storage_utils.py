import os
import json
from pathlib import Path
import hashlib
import datetime
import torch
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.kaolin_conversions import get_diff_mesh, marching_tetrahedra, kaolin_to_PLY
from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel

def allround_save_data(model: BaseModel, # Model to save data from
                       wandb_run : str = None, # Name of the wandb run to log to (e.g. "beautiful-banana-164")
                       save_mesh: bool = True, # Whether to save the mesh data
                       save_model: bool = True, # Whether to save the model data,
                       epoch: int = None, # Current epoch number
                       storage_path: str = "cluster/work/fuge/AISCO/wp4_aeroshell_optimize/bulk_storage",
                       verbose: bool = True,
                       hash_length: int = 12,
                       ) -> None:
    """
    Save model data to a specified file path in JSON format.

    Args:
        model (BaseModel): The model instance containing the data to save.
        wandb_run (str, optional): The name of the wandb run to log to. If None, a hash based on the current time will be used.
        save_mesh (bool, optional): Whether to save the mesh data. Defaults to True.
        save_model (bool, optional): Whether to save the model data. Defaults to True.
        epoch (int, optional): The current epoch number. If None, 'no_epoch' will be used in the filename.
        storage_path (str, optional): The base directory where data will be stored. Defaults to a predefined path.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        hash_length (int, optional): Length of the hash string used for directory naming. Defaults to 12.
        
    """
    # Get Dictonary containing all relevant model information needed to recreate the model
    model_data = model.get_model_info_dict() 
    if model_data is None or file_path is None:
        raise ValueError("Both model_data and file_path must be provided.")

    
    
    #Hash the model_data dictionary
    canonical_bytes = json.dumps(model_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    hash_value = hashlib.sha1(canonical_bytes).hexdigest()[:hash_length]


    #Baed on hash value see if directory exists, if not create it
    dir_path = Path(storage_path) / hash_value
    os.makedirs(dir_path, exist_ok=True)
    #If not present, instantly store the model_data in a json file for readability
    if not (dir_path / "Architecture.json").exists():
        file_path = dir_path / "Architecture.json"
        with open(file_path, 'w') as f:
            json.dump(model_data, f, indent=4)
        if verbose:
            print(f"Info about Model Architecture saved to {file_path}, created directory accordingly")

    # Make a folder for meshes and models if they don't exist
    mesh_dir = dir_path / "meshes"
    model_dir = dir_path / "models"
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # If there is no wandb_name, differentiate by using the time, but hashed to avoid conflicts if runs end at the same time
    if wandb_run is None:
        current_time = datetime.datetime.now().isoformat()
        canonical_time_bytes = current_time.encode("utf-8")
        wandb_hash = hashlib.sha1(canonical_time_bytes).hexdigest()[:10] # arbitrarily fix time hash length to 10
        wandb_run = wandb_hash

    if epoch is None:
        epoch_str = "no_epoch"
    else:
        epoch_str = f"epoch{epoch:04d}"

    if save_mesh:
        # get a model using kaolin conversions
        model=model.to('cpu')
        verts,faces = get_diff_mesh(model,grid_res=64, device='cpu')
        kaolin_to_PLY(verts, faces, filename=f"{wandb_run}_{epoch_str}.ply",mesh_dir=mesh_dir)
        if (mesh_dir / f"{wandb_run}_{epoch_str}.ply").exists():
            print(f"Warning: Overwriting existing mesh at {mesh_dir / f'{wandb_run}_{epoch_str}.ply'}")
        if verbose:
            print(f"Mesh saved to {mesh_dir / f'{wandb_run}_{epoch_str}.ply'}")

    if save_model:
        model_file_path = model_dir / f"{wandb_run}_{epoch_str}.pth"
        #Check if you would overwrite an existing model
        if model_file_path.exists():
            print(f"Warning: Overwriting existing model at {model_file_path}")
        
        torch.save(model.state_dict(), model_file_path)
        if verbose:
            print(f"Model state_dict saved to {model_file_path}")

    