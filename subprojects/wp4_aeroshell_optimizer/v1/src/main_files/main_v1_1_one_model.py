import os
from pathlib import Path

import torch
import wandb

from subprojects.wp4_aeroshell_optimizer.v1.src.main_files.pretraining_v1_1 import pretraining
from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.dynamic_loading import dynamic_type_loading
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.hashing import dict_hashing
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.losses_utils import deep_update
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.storage_utils import (
    allround_save_data,
    fetch_pretrained_model,
    reformat_wandb_name,
)
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visual_wrapper import Wrapper

# we want to dynamically run an entire optimization process for the WIRE model.
# we need the model, config and (losses)
# we need paths to store model (aka the sdf) in the end

# use the train config file


def main(verbose: bool = True) -> None:
    # Get general config

    general_config = load_config(v="v1", cfg_name="general_config.yaml")

    # Initialize WandB early to capture sweep parameters
    wandb_config = None # Store wandb config
    if general_config["wandb_general"].get("use_wandb", False):
        wandb.init(
            config=general_config,
        )
        wandb_config = wandb.config

    model_config = load_config(v="v1", cfg_name=general_config["active_configs"]["model_params"])
    
    # Update model_config with wandb sweep parameters if available
    if wandb_config:
        for k, v in wandb_config.items():
            if k.startswith("model_params."):
                key = k.split(".")[1]
                model_config[key] = v
            # Also allow direct keys if they match model_config keys
            elif k in model_config:
                model_config[k] = v

    if verbose:
        print("Successfully loaded general and model config")

    # Get locations of model definition
    model_name = general_config["model_def"]["model_name"]
    model_definition = general_config["model_def"]["model_file_path"]

    # Import model and get <class 'type'>
    ModelClass = dynamic_type_loading(file_path=model_definition, class_name=model_name)  # noqa: N806

    # Initialise model with config,important that all keys match to a keyword argument in the model __init__()
    model = ModelClass(**model_config)
    if verbose:
        print(f"Successfully initialised model: {model_name}")

    # Move model to device
    device = general_config["general_info"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Check if there is a pretrained model already
    # Note that for now this only includes pretrained spheres
    
    # If using wandb sweep, force pretraining if requested via config
    force_pretraining = False
    if wandb_config and wandb_config.get("force_pretraining"):
        force_pretraining = True
    
    # Also Check if a specific pretraining shape is requested, and make part of cache key or similar if we were caching
    
    pretrained_model = fetch_pretrained_model(
        model, pretrained_path=general_config["storage_paths"].get("pretrained_dir", None), verbose=verbose
    )
    
    if pretrained_model is not None and not force_pretraining:
        model.load_state_dict(torch.load(pretrained_model, map_location=device))
        if verbose:
            print(f"Loaded pretrained model from {pretrained_model}")
    else:
        if verbose:
            print("No pretrained model found (or forced pretraining), proceeding by pretraining")
        
        # Determine override config for pretraining
        pretraining_override = {}
        if wandb_config:
             # Pass any relevant pretraining params from wandb config
             for k, v in wandb_config.items():
                 if k.startswith("pretraining.") or k == "pretraining_shape":
                      pretraining_override[k] = v

        model = pretraining(model=model, override_config=pretraining_override)
        # Save pretrained model and mesh for sanity check
        allround_save_data(
            model=model,
            storage_path=general_config["storage_paths"].get(
                "pretrained_dir", "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/pretrained_models"
            ),
            save_model=True,
            save_mesh=True,
            verbose=verbose,
            pretraining=True,
        )

    # wandb init
    if general_config["wandb_general"].get("use_wandb", False):
        run = wandb.init(
            # Set the wandb log directory
            dir="wp4_aeroshell_optimizer/data/wandb_logs",
            # Track hyperparameters and run metadata.
            config=general_config,
        )
        config = deep_update(general_config, dict(wandb.config))
        run.config.update(config, allow_val_change=True)

    model_hash = dict_hashing(model.get_model_info_dict())
    wandb_name = wandb.run.name if wandb.run else None
    rendering_dir = None
    if wandb_name is not None:
        #set rendering dir inside wandb log dir
        rendering_dir = Path(general_config["storage_paths"]["storage_dir"]) / model_hash / reformat_wandb_name(wandb_name)
        os.makedirs(rendering_dir, exist_ok=True)
    # Initialize shared visual wrapper (reads rendering modes from config)
    wrapper = Wrapper(verbose=verbose,output_dir=rendering_dir)


    # Start optimisation
    training_config = load_config(v="v1", cfg_name=general_config["active_configs"].get("training", None))
    
    # Update training_config with wandb sweep parameters
    if wandb_config:
        for k, v in wandb_config.items():
            # Support dot notation updates e.g. "training_params.learning_rate"
            if "." in k:
                parts = k.split(".")
                if parts[0] in training_config and isinstance(training_config[parts[0]], dict):
                     training_config[parts[0]][parts[1]] = v
            elif k in training_config:
                training_config[k] = v
            # If flattening is used in sweep config, try to match keys in training_params
            elif "training_params" in training_config and k in training_config["training_params"]:
                training_config["training_params"][k] = v

    # Check if model still is on device after pretraining
    model = model.to(device)



    model.optimise(training_config=training_config, renderer_wrapper=wrapper)
    if verbose:
        print("Model optimisation completed.")

    # Save final model and mesh

    allround_save_data(
        model=model,
        storage_path=general_config["general_info"].get(
            "storage_path", "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/bulk_storage"
        ),
        save_model=True,
        save_mesh=True,
        verbose=verbose,
        pretraining=False,
        wandb_run=wandb_name,
    )
    # Log neural renderer video to wandb if applicable
    if general_config["wandb_general"]["use_wandb"] and general_config["wandb_general"]["use_neural_renderer"]:
        # Finalize all rendering modes and save results
        try:
            wrapper.finalize()
        except Exception:
            print("Wrapper finalize failed:")

    print("Completed job successfully.")


if __name__ == "__main__":
    main()

    print("Completed job successfully.")
