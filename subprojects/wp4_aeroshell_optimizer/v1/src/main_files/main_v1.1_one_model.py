from pathlib import Path

import torch
import wandb

from subprojects.wp4_aeroshell_optimizer.v1.src.main_files.pretraining_v1_1 import pretraining
from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.dynamic_loading import dynamic_type_loading
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.hashing import dict_hashing
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.storage_utils import allround_save_data, fetch_pretrained_model

# we want to dynamically run an entire optimization process for the WIRE model.
# we need the model, config and (losses)
# we need paths to store model (aka the sdf) in the end

# use the train config file




def main(verbose: bool = True) -> None:
    # Get general config

    general_config = load_config(v="v1", cfg_name="general_config.yaml")
    model_config = load_config(v="v1", cfg_name=general_config["active_configs"]["model_params"])
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

    pretrained_model = fetch_pretrained_model(
        model, pretrained_path=general_config["storage_paths"].get("pretrained_dir", None), verbose=verbose
    )
    if pretrained_model is not None:
        model.load_state_dict(torch.load(pretrained_model, map_location=device))
        if verbose:
            print(f"Loaded pretrained model from {pretrained_model}")
    else:
        if verbose:
            print("No pretrained model found, proceeding by pretraining")
        model = pretraining(model=model)
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

    # Start optimisation
    training_config = load_config(v="v1", cfg_name=general_config["active_configs"].get("training", None))

    # Check if model still is on device after pretraining
    model = model.to(device)

    model.optimise(training_config=training_config)
    if verbose:
        print("Model optimisation completed.")

    # Save final model and mesh
    if general_config["wandb_general"].get("use_wandb", False):
        wandb_name = wandb.run.name
    else:
        wandb_name = None
    allround_save_data(
        model=model,
        storage_path=general_config["general_info"].get(
            "storage_path", "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/bulk_storage"
        ),
        save_model=True,
        save_mesh=True,
        verbose=verbose,
        pretraining=False,
        wandb_run=wandb_name
    )
    if general_config["wandb_general"]["use_wandb"] and general_config["wandb_general"]["use_neural_renderer"]:
        if len(model.visualiser.stored_frames) == 0:
            print("Warning: No frames stored in neural renderer for logging.")

        if general_config.get("neural_renderer", {})["gif"].get("make_gif", False):
            fps = general_config.get("neural_renderer", {}).get("gif", {}).get("fps", 10)

            if general_config.get("neural_renderer", {}).get("save_locally", False):
                model.visualiser.frames_to_gif(
                    file_path=Path(f"subprojects/wp4_aeroshell_optimizer/data/rendered_run_{wandb.run.name}.gif"),
                    fps=fps,
                    return_uint8=True,
                )

            if general_config.get("neural_renderer", {}).get("upload", True):
                # wandb.Video expects float32 frames in [0, 1] range
                frames_float = model.visualiser.frames_to_gif(
                    file_path=None,
                    fps=fps,
                    return_uint8=False,
                )
                wandb.log({"rendered_video": wandb.Video(frames_float, fps=fps, format="gif")})

    print("Completed job successfully.")


if __name__ == "__main__":
    main()
