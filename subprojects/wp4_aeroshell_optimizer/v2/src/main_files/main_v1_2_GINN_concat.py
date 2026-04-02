"""
Our first attempt at reproducing the GINN paper for our purposes
Attempts to do this by concatenating a latent vector to the input coordinates,
which is then used to compute a diversity measure between the networks in the latent space

"""
import os
from pathlib import Path  # noqa: F401

import torch

import wandb
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.total_loss import TotalLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_v1_1 import pretraining
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import (
    apply_sweep_overrides,
    get_master_config,
    load_config,
    register_all_configs,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.dynamic_loading import dynamic_type_loading
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.hashing import dict_hashing
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.storage_utils import (
    allround_save_data,
    fetch_pretrained_model,
    reformat_wandb_name,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.visual_wrapper import VisualisationManager


def main(verbose: bool = True) -> None:  # noqa: PLR0912, PLR0915
    # ------------------------------------------------------------------
    # 1. Load general config from YAML
    # ------------------------------------------------------------------
    general_config = load_config(v="v2", cfg_name="general_config.yaml")
    # ------------------------------------------------------------------
    # 2. Initialise WandB (picks up sweep agent params into wandb.config)
    # ------------------------------------------------------------------
    use_wandb = general_config["wandb_general"].get("use_wandb", False)
    if use_wandb:
        wandb.init(
            entity=general_config["wandb_general"]["entity_name"],
            project=general_config["wandb_general"]["project_name"],
            config=general_config,
        )
    # ------------------------------------------------------------------
    # 3. Register all sub-configs & apply sweep overrides (if any)
    #
    #    After this block every subsequent load_config() call returns the
    #    sweep-patched version — no ad-hoc override blocks needed.
    register_all_configs(general_config, v="v2")

    if wandb.run is not None:
        apply_sweep_overrides(dict(wandb.config))
        # Update wandb dashboard with the full patched master config
        wandb.config.update(get_master_config(), allow_val_change=True)

    if verbose:
        print("Successfully loaded and registered all configs")

    # ------------------------------------------------------------------
    # 4. Load model config (now sweep-aware) and instantiate model
    # ------------------------------------------------------------------
    model_config = load_config(v="v2", cfg_name=general_config["active_configs"]["model_params"])

    model_name = general_config["model_def"]["model_name"]
    model_definition = general_config["model_def"]["model_file_path"]

    ModelClass = dynamic_type_loading(file_path=model_definition, class_name=model_name)
    model = ModelClass(**model_config)
    if verbose:
        print(f"Successfully initialised model: {model_name}")

    device: str = general_config["general_info"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    # ------------------------------------------------------------------
    # 5. Pretraining (or load pretrained model)
    # ------------------------------------------------------------------
    ### load configs
    pretraining_config_name = general_config["active_configs"].get("pretraining", "pretraining_v0_config.yaml")
    pretraining_config = load_config(v="v2", cfg_name=pretraining_config_name)
    force_pretraining = pretraining_config["pretraining"].get("force_pretraining", False)
    run_locally = general_config["general_info"].get("run_locally", "False") == "True"

    ### get paths
    if run_locally:
        pretrained_path = pretraining_config["pretraining"].get("local_pretraining_model_path", "")
    else:
        pretrained_path = pretraining_config["pretraining"].get("pretraining_model_path", "")

    print(f"Pretrained path: {pretrained_path}")

    # Sicherheitscheck: Nur ausführen, wenn pretrained_path ein gültiger Wert ist
    if pretrained_path:
        pretrained_model = fetch_pretrained_model(model, pretrained_path=pretrained_path, verbose=verbose)
    else:
        print("Skipping pretrained model fetch: No path provided in config.")
        pretrained_model = None

    # Dein Debug-Print kann danach kommen
    print(f"[DEBUG main] pretraining_model_path from config: {pretrained_path}")

    ### load pretrained model if exists
    ### # aalthaus, why the hell does the fetch function get called a second time here?
    pretrained_model = fetch_pretrained_model(model, pretrained_path=pretrained_path, verbose=verbose)
    print(f"[DEBUG main] os.path.exists(path): {os.path.exists(pretrained_path)}")
    print(f"[DEBUG main] os.path.isfile(path): {os.path.isfile(pretrained_path)}")
    print(f"[DEBUG main] os.path.isdir(path): {os.path.isdir(pretrained_path)}")
    ### pretraining if model doesn't exist
    if pretrained_model is not None and not force_pretraining:
        model.load_state_dict(torch.load(pretrained_model, map_location=device))
        if verbose:
            print(f"Loaded pretrained model of type {model.__class__.__name__} from {pretrained_model}")
    else:
        if verbose:
            print("No pretrained model found (or forced pretraining), proceeding by pretraining")

        # Determine override config for pretraining
        # Replace lines 94-98 with this:
        wandb_config = dict(wandb.config) if wandb.run is not None else {}
        pretraining_override = (
            {k: v for k, v in wandb_config.items() if k.startswith("pretraining.") or k == "pretraining_shape"}
            if wandb_config
            else {}
        )
        print(f"Starting pretraining for model {model.__class__.__name__}")
        model = pretraining(model=model, override_config=pretraining_override)

        allround_save_data(
            model=model,
            storage_path=pretrained_path,
            save_model=True,
            save_mesh=True,
            epoch=None,
            verbose=verbose,
            pretraining=True,
        )

    # Model Loading complete
    if verbose:
        print("Pretraining and model loading complete")

    # Initialize TotalLoss to comput all the performance/constraint losses, for now without visualization manager
    total_loss: TotalLoss = TotalLoss(dev=device, renderer_manager=None)
    training_config = load_config(v="v2", cfg_name=general_config["active_configs"]["training"])


    ### LLM please replace this part withe the diversity trainer
    optimiser = torch.optim.Adam(model.parameters(), lr=training_config["training"]["learning_rate"])
    for epoch in range(training_config["training"]["num_epochs"]):
        # Set to training mode, for safety at start of each epoch
        model.train()
        optimiser.zero_grad()
        .
        .
        .
