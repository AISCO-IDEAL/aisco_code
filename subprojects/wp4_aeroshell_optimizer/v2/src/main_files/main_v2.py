"""
This should be the final main file we use for the Project. It should be modular and easily extendible
"""

from typing import Any

import torch

import wandb
from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_v1_1 import pretraining
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat, SupportsOptimise
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import (
    apply_sweep_overrides,
    get_master_config,
    load_config,
    register_all_configs,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.debug_manager import DebugManager
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.dynamic_loading import smart_model_init
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.pyvista_headless import (
    configure_headless_pyvista,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.storage_utils import (
    allround_save_data,
    fetch_pretrained_model,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.wandb_utils import patch_wandb_log_for_monotonic_steps

configure_headless_pyvista()


def main() -> None:  # noqa: PLR0912, PLR0915
    print("Starting Main file V2...")

    # 1. Load and register configs
    general_config: dict[str, Any] = load_config(v="v2", cfg_name="general_config.yaml")
    verbose: bool = general_config.get("general_info", {}).get("verbose", True)
    device: str = general_config.get("general_info", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Init Elias' debug manager NOT WORKING PROPERLY ATM, PLEASE CHECK
    debug_cfg = load_config(v="v2", cfg_name="debug.yaml")
    debug_manager = DebugManager(config=debug_cfg)
    if verbose:
        print(debug_manager.summary())
    # 2. Handle wandb init and potential sweep/logging setup.

    use_wandb: bool = general_config["wandb_general"].get("use_wandb", False)
    if use_wandb:
        try:
            wandb.init(
                entity=general_config["wandb_general"]["entity_name"],
                project=general_config["wandb_general"]["project_name"],
                config=general_config,
            )
            # keep this for now, not sure if this is any longer required
            patch_wandb_log_for_monotonic_steps()
            # Separate axes keep training and rendering logs chronological
            # without forcing everything onto W&B's global step.
            wandb.define_metric("training_epoch")
            wandb.define_metric("render_epoch")
            # Default: bind all metrics to training_epoch so scalar curves stay
            # aligned even when new keys are added without an explicit mapping.
            wandb.define_metric("*", step_metric="training_epoch")
            # Override media/render namespaces to use their own render_epoch axis.
            wandb.define_metric("renders/*", step_metric="render_epoch")
            wandb.define_metric("animations/*", step_metric="render_epoch")
            wandb.define_metric("images/*", step_metric="render_epoch")
        except Exception as e:
            print(f"Error initializing Weights & Biases: {e}")
            print("Proceeding without W&B logging.")
            use_wandb = False  # Disable W&B logging if initialization fails
            general_config["wandb_general"]["use_wandb"] = False  # Override the config to reflect that W&B is not being used

    ### Sweeping setup

    # TODO: Remove hardcoded seperate configs in the function
    register_all_configs(general_config, v="v2")
    if wandb.run is not None:
        apply_sweep_overrides(dict(wandb.config))
        # Update wandb dashboard with the full patched master config
        wandb.config.update(get_master_config(), allow_val_change=True)

    if verbose:
        print("Successfully loaded and registered all configs")

    # 3. Load model and handle pretraining
    model_name: str = general_config["model_def"]["model_name"]
    model, parent_class_name, model_class_name, settings = smart_model_init(model_name)

    # 3a. Find out what parentclass the model has and thus infer what kind of treatment it needs
    uses_diversity: bool = settings["uses_diversity"]  # whether model uses diversity latent variable
    has_optimise: bool = settings[
        "has_optimise"
    ]  # whether model is old and thus relies on .optimise() member function instead of a separate trainer class
    if has_optimise and not isinstance(model, SupportsOptimise):
        raise TypeError(
            "Model is marked with has_optimise=True but does not implement optimise(). "
            "Check models.json and the model implementation."
        )

    # 3b. Handle pretraining if needed.
    pretraining_config_name = general_config["active_configs"].get("pretraining", "pretraining_v0_config.yaml")
    pretraining_config = load_config(v="v2", cfg_name=pretraining_config_name)
    force_pretraining = pretraining_config["pretraining"].get("force_pretraining", False)
    run_locally = general_config["general_info"].get("run_locally", "False") == "True"

    ### get paths
    if run_locally:
        pretrained_path = pretraining_config["pretraining"].get("local_pretraining_model_path", "")
        if not pretrained_path:  # check if empty path given in pretraining config, if so fallback to general config path
            pretrained_path = general_config["storage_paths"].get("local_storage_dir", "./local_storage_dir")
    else:
        pretrained_path = pretraining_config["pretraining"].get("pretraining_model_path", "")
        if not pretrained_path:  # check if empty path given in pretraining config, if so fallback to general config path
            pretrained_path = general_config["storage_paths"].get("storage_dir", "/cluster/work/fuge/AISCO/temp_")

    if verbose:
        print(f"Pretrained path (from pretraining cfg): {pretrained_path}")

    ### load pretrained model if exists
    pretrained_model_path = fetch_pretrained_model(model, pretrained_path=pretrained_path, verbose=verbose)

    ### pretraining if model doesn't exist
    if pretrained_model_path is not None and not force_pretraining:
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        if verbose:
            print(f"Loaded pretrained model of type {model.__class__.__name__} from {pretrained_model_path}")
    else:
        if verbose and not force_pretraining:
            print("No pretrained model found, proceeding by pretraining")
        if verbose and force_pretraining:
            print("Force pretraining enabled, proceeding by pretraining")
            if pretrained_model_path:
                print(
                    f"Note: A pretrained model was found at {pretrained_model_path} but will be ignored due to force_pretraining=True"
                )

        print(f"Starting pretraining for model {model.__class__.__name__}")
        if uses_diversity:
            if not isinstance(model, SupportsLatentConcat):
                raise TypeError(
                    "Model must support latent-vector methods to use diversity training. "
                    "Check uses_diversity in models.json and the model implementation."
                )

            active_latent = model.get_latent_vector()
            if active_latent is None or active_latent.numel() == 0:
                raise ValueError(
                    "Latent vector is not initialized. Ensure the model initializes a non-empty latent vector before pretraining."
                )

            # Reset active latent vector to zeros before pretraining.
            model.set_latent_vector(torch.zeros_like(active_latent))

        model = pretraining(model=model, debug_manager=debug_manager)

        allround_save_data(
            model=model,
            storage_path=pretrained_path,
            save_model=True,
            save_mesh=True,
            epoch=None,
            verbose=verbose,
            pretraining=True,
        )
    if verbose:
        print(f"Pretraining complete for model: {model_class_name} with Parent class: {parent_class_name}")
        print(f"Detailed settings: {[f'{key}: {value}' for key, value in settings.items()]}")
    # 4. Initialise whatever we need to run the main training loop, e.g. trainer, loss function, etc.

    # 5. Run main training loop

    # 6. Saving of files and postprocessing

    print("Completed job successfully.")


if __name__ == "__main__":
    main()
