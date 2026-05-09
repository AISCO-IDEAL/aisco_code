# ruff: noqa: PLR0912 # reduce the if else mess, once everything works again!
import copy
import datetime
import json
import os
from dataclasses import replace
from pathlib import Path

import torch
import trimesh

from subprojects.wp2_assembler.version4.utils.reg_imp import get_outer_prism_bounds_mm
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat, SupportsModelContract
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.device_detection import is_local_machine
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.hashing import dict_hashing, str_hashing
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.kaolin_conversions import kaolin_to_PLY
from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor, load_mesh_extraction_config


def allround_save_data(  # noqa: PLR0915
    model: SupportsModelContract,  # Model to save data from
    wandb_run: str | None = None,  # Name of the wandb run to log to (e.g. "beautiful-banana-164")
    save_mesh: bool = True,  # Whether to save the mesh data
    save_model: bool = True,  # Whether to save the model data,
    epoch: int | None = None,  # Current epoch number
    storage_path: str = "",
    resolution: int = 64,  # resolution for get_diff_mesh, only used if save_mesh is True and marching_cubes_usage is not active
    verbose: bool = True,
    pretraining: bool = False,
    device: str | None = None,
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
    # Get Dictonary containing all relevant model information needed to recreate the model
    model_data = model.get_model_info_dict()
    if model_data is None or storage_path is None:
        raise ValueError("Both model_data and file_path must be provided.")

    # Hash the model_data dictionary
    model_hash_value = dict_hashing(model_data)
    # If storage_path is provided in the function arguments, use it
    # Otherwise fallback to environment-aware path selection
    if not storage_path:
        if is_local_machine():
            fallback_config = load_config(v="v2", cfg_name="general_config.yaml")
            storage_path = fallback_config["storage_paths"].get("beast_storage_dir") or fallback_config["storage_paths"].get(
                "local_storage_dir", "./local_storage_dir"
            )
        else:
            fallback_config = load_config(v="v2", cfg_name="general_config.yaml")
            storage_path = fallback_config["storage_paths"].get(
                "storage_dir", "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/bulk_storage"
            )

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
        mesh_config = load_mesh_extraction_config()
        if resolution is not None:
            mesh_config = replace(mesh_config, base_resolution=resolution)
        if device is not None:
            mesh_config = replace(mesh_config, device=device)

        mesh_device = str(mesh_config.device)
        if mesh_device.startswith("cuda") and not torch.cuda.is_available():
            if verbose:
                print(f"Configured mesh save device '{mesh_device}' unavailable. Falling back to CPU.")
            mesh_device = "cpu"
            mesh_config = replace(mesh_config, device=mesh_device)

        temp_model = copy.deepcopy(model).to(mesh_device)
        extractor = MeshSDFExtractor(config=mesh_config)
        _prism = get_outer_prism_bounds_mm()
        extractor.set_grid(
            bbox=[[_prism["x"][0], _prism["x"][1]], [_prism["y"][0], _prism["y"][1]], [_prism["z"][0], _prism["z"][1]]],
            resolution=mesh_config.base_resolution,
        )
        verts, faces = extractor.extract_mesh(temp_model)

        filename = f"{wandb_run}_{epoch_str}_adaptive_mesh_res_{mesh_config.base_resolution}.ply"
        if (mesh_dir / filename).exists():
            print(f"Warning: Overwriting existing mesh at {mesh_dir / filename}")
        kaolin_to_PLY(
            verts,
            faces,
            filename=filename,
            mesh_dir=mesh_dir,
            verbose=verbose,
        )

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


def save_population_meshes(
    model: SupportsLatentConcat,
    population_z: torch.Tensor,
    storage_path: str,
    wandb_run: str | None,
    epoch: int | None,
    resolution: int = 128,
    verbose: bool = True,
    device: str | None = None,
) -> None:
    """Save one PLY mesh per latent slot after population training.

    Iterates over all rows in ``population_z``, sets the model's active latent
    vector, extracts the mesh via :class:`MeshSDFExtractor`, and writes a PLY
    file to the run's mesh directory.  The model is *not* mutated: each
    extraction uses ``copy.deepcopy(model)``.

    Args:
        model: Trained model implementing ``SupportsLatentConcat``
            (which also satisfies ``SupportsModelContract``).
        population_z: Float tensor of shape ``(N, z_dim)`` containing the
            final latent vectors for all N population slots.
        storage_path: Root storage directory (same value passed to
            ``allround_save_data``).
        wandb_run: W&B run name used to resolve the mesh subdirectory.
        epoch: Final epoch number, embedded in each filename.
        resolution: Marching-cubes grid resolution (default 128).
        verbose: Whether to print per-slot progress.
        device: Torch device string for mesh extraction (e.g. ``"cuda"``).

    Returns:
        None
    """
    n_slots = population_z.shape[0]
    if n_slots == 0:
        return

    model_hash_value = dict_hashing(model.get_model_info_dict())
    wname = reformat_wandb_name(wandb_run) if wandb_run is not None and "-" in wandb_run else wandb_run or model_hash_value
    epoch_str = "no_epoch" if epoch is None else f"epoch{epoch:04d}"

    mesh_dir = Path(storage_path) / model_hash_value / wname / "meshes"
    os.makedirs(mesh_dir, exist_ok=True)

    mesh_config = load_mesh_extraction_config()
    mesh_config = replace(mesh_config, base_resolution=resolution)
    if device is not None:
        mesh_config = replace(mesh_config, device=device)

    mesh_device = str(mesh_config.device)
    if mesh_device.startswith("cuda") and not torch.cuda.is_available():
        if verbose:
            print(f"[population_meshes] Configured device '{mesh_device}' unavailable. Falling back to CPU.")
        mesh_device = "cpu"
        mesh_config = replace(mesh_config, device=mesh_device)

    extractor = MeshSDFExtractor(config=mesh_config)
    _prism = get_outer_prism_bounds_mm()
    extractor.set_grid(
        bbox=[[_prism["x"][0], _prism["x"][1]], [_prism["y"][0], _prism["y"][1]], [_prism["z"][0], _prism["z"][1]]],
        resolution=mesh_config.base_resolution,
    )

    for i in range(n_slots):
        z_vec = population_z[i]
        slot_model = copy.deepcopy(model)
        slot_model.set_latent_vector(z_vec.clone())
        verts, faces = extractor.extract_mesh(slot_model)

        z0 = z_vec[0].item()
        z1 = z_vec[1].item() if z_vec.numel() > 1 else 0.0
        filename = f"slot_{i}_z{z0:.2f}_{z1:.2f}_{epoch_str}.ply"

        kaolin_to_PLY(verts, faces, filename=filename, mesh_dir=mesh_dir, verbose=verbose)
        if verbose:
            print(f"[population_meshes] Slot {i} mesh saved: {mesh_dir / filename}")


def fetch_pretrained_model(
    model: SupportsModelContract,
    pretrained_path: str | None = "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/pretrained_models",
    verbose: bool = True,
) -> None | Path:
    """
    Check whether a pretrained model exists for the given model architecture in the pretrained_path.
    Returns the path to the pretrained model if found, else None.
    """
    if pretrained_path is None:
        if verbose:
            print("No valid pretrained path provided, skipping pretrained model fetch.")
        return None

    # If exact file provided
    if os.path.isfile(pretrained_path):
        return Path(pretrained_path)

    # First, hash the model to look for specific architecture
    model_hash = dict_hashing(model.get_model_info_dict())

    # If a directory provided, look for hash specific file or any .pth
    if os.path.isdir(pretrained_path):
        # 1. Search for directory containing hash and file with hash
        for d in Path(pretrained_path).glob(f"*{model_hash}*"):
            if d.is_dir():
                for f in d.glob(f"*{model_hash}*.pth"):
                    if verbose:
                        print(f"Pretrained model for {model_hash} found by hash: {f}")
                    return f

    if verbose:
        print(f"No pretrained model found for {model_hash} in {pretrained_path}.")
    return None


# Not usesd as of yet. Maybe once we create different pretrained shapes
def save_pretrained_model(
    model: SupportsModelContract,
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
