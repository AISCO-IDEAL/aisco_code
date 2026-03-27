# ruff: noqa: PLR0912 # reduce the if else mess, once everything works again!
import copy
import datetime
import glob
import json
import os
from pathlib import Path

import torch
import trimesh

from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.hashing import dict_hashing, str_hashing
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.kaolin_conversions import (
    get_diff_mesh,
    kaolin_to_PLY,
)


def allround_save_data(  # noqa: PLR0915
    model: BaseModel,  # Model to save data from
    wandb_run: str | None = None,  # Name of the wandb run to log to (e.g. "beautiful-banana-164")
    save_mesh: bool = True,  # Whether to save the mesh data
    save_model: bool = True,  # Whether to save the model data,
    epoch: int | None = None,  # Current epoch number
    storage_path: str = "cluster/work/fuge/AISCO/wp4_aeroshell_optimizer/bulk_storage",
    verbose: bool = True,
    pretraining: bool = False,
) -> None:
    """
    Save model data to a specified file path in JSON format, with options to save mesh and model state dictionary.

    Args:
        model (BaseModel): The model instance containing the data to save.
        wandb_run (str, optional): The name of the wandb run to log to. If None, a hash based on the current time will be used.
        save_mesh (bool, optional): Whether to save the mesh data. Defaults to True.
        save_model (bool, optional): Whether to save the model data. Defaults to True.
        epoch (int, optional): The current epoch number. If None, 'no_epoch' will be used in the filename.
        storage_path (str, optional): The base directory where data will be stored. Defaults to a predefined path.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    For questions regarding this function, talk to #aalthaus
    """
    # Get general config:
    general_config = load_config(v="v1", cfg_name="general_config.yaml")
    # Get Dictonary containing all relevant model information needed to recreate the model
    model_data = model.get_model_info_dict()
    if model_data is None or storage_path is None:
        raise ValueError("Both model_data and file_path must be provided.")

    # Hash the model_data dictionary
    model_hash_value = dict_hashing(model_data)
    # Baed on hash value see if directory exists, if not create it
    run_locally = general_config["general_info"].get("run_locally", "False") == "True"
    if run_locally:
        storage_path = general_config["storage_paths"]["local_storage_dir"]
    else:
        storage_path = general_config["storage_paths"]["storage_dir"]
    dir_path = Path(storage_path) / model_hash_value
    os.makedirs(dir_path, exist_ok=True)
    # If not present, instantly store the model_data in a json file for readability
    if not (dir_path / f"Architecture_{model_hash_value}.json").exists():
        file_path = dir_path / f"Architecture_{model_hash_value}.json"
        with open(file_path, "w") as f:
            json.dump(model_data, f, indent=4)
        if verbose:
            print(f"Info about Model Architecture saved to {file_path}, created directory accordingly")

    # before first usage, adapt wandb_run so that the numbers at the back go in the front,
    # for easier file sorting in editor (change from e.g. beautiful-banana-164 to 164-beautiful-banana)
    tmp = wandb_run
    wandb_run = reformat_wandb_name(tmp) if tmp is not None else None
    # Make a folder for meshes and models if they don't exist
    # If we're pretraining, we want to store everything in the same folder
    if not pretraining:
        if wandb_run is None:
            mesh_dir = dir_path / "meshes"
            model_dir = dir_path / "models"
        else:
            # create subfolder for each wandb run
            mesh_dir = dir_path / wandb_run / "meshes"
            model_dir = dir_path / wandb_run / "models"
    else:
        mesh_dir = dir_path
        model_dir = dir_path
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # if in pretraining, we want model to have the name of its hash value only
    if pretraining:
        epoch_str = "pretrained"
        wandb_run = model_hash_value
    else:
        # Otherwise, proceed as usual and set wandb_run and epoch_str accordingly
        # If there is no wandb_name, differentiate by using the time, but hashed to avoid conflicts if runs end at the same time
        if wandb_run is None:
            current_time = datetime.datetime.now().isoformat()
            canonical_time_bytes = str(current_time.encode("utf-8"))
            wandb_run = str_hashing(canonical_time_bytes)

        epoch_str = "no_epoch" if epoch is None else f"epoch{epoch:04d}"

    #####################
    ### save the mesh ###
    #####################
    if save_mesh:
        # get a model using kaolin conversions
        # Use a copy to avoid disrupting training
        temp_model = copy.deepcopy(model).cpu()
        verts, faces = get_diff_mesh(temp_model, grid_res=64, device="cpu")
        if (mesh_dir / f"{wandb_run}_{epoch_str}.ply").exists():
            print(f"Warning: Overwriting existing mesh at {mesh_dir / f'{wandb_run}_{epoch_str}.ply'}")

        kaolin_to_PLY(verts, faces, filename=f"{wandb_run}_{epoch_str}.ply", mesh_dir=mesh_dir, verbose=verbose)

    ######################
    ### save the model ###
    ######################
    if save_model:
        model_file_path = model_dir / f"{wandb_run}_{epoch_str}.pth"
        # Check if you would overwrite an existing model
        if model_file_path.exists():
            print(f"Warning: Overwriting existing model at {model_file_path}")

        torch.save(model.state_dict(), model_file_path)
        if verbose:
            print(f"Model state_dict saved to {model_file_path}")


def fetch_pretrained_model(
    model: BaseModel,
    pretrained_path: str = "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/pretrained_models",
    verbose: bool = True,
) -> None | Path:
    """
    Check whether a pretrained model exists for the given model architecture in the pretrained_path.
    Returns the path to the pretrained model if found, else None.

    """
    print(f"[DEBUG storage_utils] requested pretrained_path: {pretrained_path}")
    if pretrained_path is None:
        print("[DEBUG storage_utils] path is None")
    else:
        print(f"[DEBUG storage_utils] os.path.exists: {os.path.exists(pretrained_path)}")
        print(f"[DEBUG storage_utils] os.path.isfile: {os.path.isfile(pretrained_path)}")
        print(f"[DEBUG storage_utils] os.path.isdir: {os.path.isdir(pretrained_path)}")

    # If exact file provided
    if pretrained_path and os.path.isfile(pretrained_path):
        print(f"[DEBUG storage_utils] exact file found: {pretrained_path}")
        return Path(pretrained_path)

    # If a directory provided, search for .pth recursively
    if pretrained_path and os.path.isdir(pretrained_path):
        candidates = glob.glob(os.path.join(pretrained_path, "**", "*.pth"), recursive=True)
        print(f"[DEBUG storage_utils] candidates found in dir: {candidates}")
        if candidates:
            chosen = sorted(candidates)[0]
            print(f"[DEBUG storage_utils] chosen candidate: {chosen}")

    if pretrained_path is None:
        print("No valid pretrained path provided, skipping pretrained model fetch.")
        return None

    model_hash = dict_hashing(model.get_model_info_dict())
    found = False
    file = None
    # Check whether there is a directory containing the hash in its name and whether theres a corresponding .pth file (also with hash in name)
    for d in Path(pretrained_path).glob(f"*{model_hash}*"):
        if d.is_dir():
            for f in d.glob(f"*{model_hash}*.pth"):
                file = f
                found = True
                break
        if found:
            break

    if verbose and found:
        print(f"Pretrained model for {model_hash} found: {file}")
    elif verbose and not found:
        print(f"No pretrained model found for {model_hash}.")

    if not found:
        return None
    else:
        return file


# Not usesd as of yet. Maybe once we create different pretrained shapes
def save_pretrained_model(
    model: BaseModel,
    file_path: str = "cluster/work/fuge/AISCO/wp4_aeroshell_optimizer/pretrained_models",
    verbose: bool = True,
    shape: str = "sphere_1m",
) -> None:
    """
    Save the state dictionary of a pretrained model to a specified file path.

    Args:
        model (BaseModel): The model instance containing the pretrained weights.
        file_path (str): The file path where the state dictionary will be saved.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    """
    if model is None or file_path is None:
        raise ValueError("Both model and file_path must be provided.")

    model_name = f"{shape}_" + dict_hashing(model.get_model_info_dict()) + ".pth"
    torch.save(model.state_dict(), Path(file_path) / model_name)
    if verbose:
        print(f"Pretrained model state_dict saved to {Path(file_path) / model_name}")


def reformat_wandb_name(wandb_name: str) -> str:
    """
    Reformats a wandb run name from 'name-number' to 'number-name' for easier sorting.

    Args:
        wandb_name (str | None): The original wandb run name.
    """
    wandb_name = (
        f"{wandb_name.rsplit('-', maxsplit=1)[-1]}_" + "-".join(wandb_name.split("-")[:-1])
        if wandb_name is not None and "-" in wandb_name
        else wandb_name
    )
    return wandb_name


def _load_mesh(path: str) -> trimesh.Trimesh:
    if not path:
        raise ValueError("PLY path must be a non-empty string.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"PLY file does not exist: {path}")

    loaded = trimesh.load(path, file_type="ply", process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = tuple(loaded.geometry.values())
        if len(geoms) == 0:
            raise ValueError(f"PLY scene has no geometry: {path}")
        mesh: trimesh.Trimesh = trimesh.util.concatenate(geoms)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported PLY payload at {path}: {type(loaded)}")

    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) > 0:
        trimesh.repair.fix_normals(mesh, multibody=False)
    else:
        raise ValueError(f"Mesh has no faces: {path}")

    return mesh


def fetch_bounding_box_ply(inner_ply_location: str = "", outer_ply_location: str = ""):
    """Load and return inner/outer bounding-box meshes from PLY files."""
    inner_mesh = _load_mesh(inner_ply_location)
    outer_mesh = _load_mesh(outer_ply_location)
    return inner_mesh, outer_mesh
