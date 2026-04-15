"""
This should be the final main file we use for the Project. It should be modular and easily extendible
"""

import os
from pathlib import Path
from typing import Any

import torch

import wandb
from subprojects.wp3_solar_estimator.mesh_extractors.non_diff_algos.marching_cubes import MeshExtractor
from subprojects.wp4_aeroshell_optimizer.v2.src.GINN.diversity_manager import (
    DiversityTrainer,
    default_field_diversity_loss,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.alm import ALMManager
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.total_loss_v2 import TotalLossV2
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
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.hashing import dict_hashing
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.pyvista_headless import (
    configure_headless_pyvista,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.storage_utils import (
    allround_save_data,
    fetch_pretrained_model,
    reformat_wandb_name,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.wandb_utils import patch_wandb_log_for_monotonic_steps
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.GINN_visuals.ginn_visuals_utils import (
    cleanup_latent_interpolator,
    run_latent_interpolation,
    setup_latent_interpolator,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.visual_wrapper import VisualisationManager

configure_headless_pyvista()


def main() -> None:  # noqa: PLR0912, PLR0915
    print("Starting Main file V2...")

    # 1. Load and register configs
    general_config: dict[str, Any] = load_config(v="v2", cfg_name="general_config.yaml")
    verbose: bool = general_config.get("general_info", {}).get("verbose", True)
    device: str = general_config.get("general_info", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print(f"Configured device '{device}' is unavailable. Falling back to CPU.")
        device = "cpu"

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

    # Keep model parameters/buffers on the runtime device expected by losses and data tensors.
    model = model.to(device)

    if verbose:
        print(f"Pretraining complete for model: {model_class_name} with Parent class: {parent_class_name}")
        print(f"Detailed settings: {[f'{key}: {value}' for key, value in settings.items()]}")

    # 4. Initialise whatever we need to run the main training loop, e.g. trainer, loss function, etc.

    training_config = load_config(v="v2", cfg_name=general_config["active_configs"].get("training", None))

    # Shared rendering-dir logic (used by both legacy and diversity paths)
    model_hash = dict_hashing(model.get_model_info_dict())
    wandb_name = wandb.run.name if wandb.run is not None else None
    rendering_dir: Path | None = None
    if wandb_name is not None:
        if run_locally:
            rendering_dir = (
                Path(general_config["storage_paths"]["local_storage_dir"]) / model_hash / reformat_wandb_name(wandb_name)
            )
        else:
            rendering_dir = Path(general_config["storage_paths"]["storage_dir"]) / model_hash / reformat_wandb_name(wandb_name)
        os.makedirs(rendering_dir, exist_ok=True)

    # 5. Run main training loop

    if has_optimise:
        assert isinstance(model, SupportsOptimise), (
            "Model is marked with has_optimise=True but does not adhere to SupportsOptimise contract."
        )

        renderer_manager = VisualisationManager(verbose=verbose, output_dir=rendering_dir)
        print("Model has legacy optimise() method, assuming single batch optimisation is desired. \n Starting training...")
        model.optimise(
            training_config=training_config,
            renderer_manager=renderer_manager,
            wandb_name=wandb_name,
            general_config=general_config,
            debug_manager=debug_manager,
        )

        # ------------------------------------------------------------------
        # 9. Finalize renderers
        # ------------------------------------------------------------------
        if use_wandb and general_config["wandb_general"].get("use_neural_renderer", False):
            try:
                renderer_manager.finalize()
            except Exception as e:
                print(f"Manager finalize failed: {e}")

    elif uses_diversity:
        # New path: TotalLossV2 + optional ALM + DiversityTrainer.
        # Covers both single-model (population_size=1, diversity_loss_fn=None) and
        # population-based GINN training through the same trainer.
        if not isinstance(model, SupportsLatentConcat):
            raise TypeError(
                "Non-legacy models must implement SupportsLatentConcat (set_latent_vector / "
                "get_latent_vector) to use the TotalLossV2 + DiversityTrainer pipeline."
            )

        losses_config = load_config(v="v2", cfg_name=general_config["active_configs"]["losses"])
        wandb_cfg: dict[str, Any] = general_config.get("wandb_general", {})

        total_loss = TotalLossV2(device=device)
        use_alm: bool = losses_config["general_info"].get("ALM") == "True"
        alm_manager: ALMManager | None = (
            ALMManager(total_loss.constraint_names, losses_config["general_info"], torch.device(device)) if use_alm else None
        )

        # Mesh extractor + optional renderer (only used if rendering_freq is set)
        renderer_cfg = load_config(v="v1", cfg_name="neural_renderer_v0.yaml")
        mesh_extraction_config: dict[str, Any] = renderer_cfg.get("mesh_extraction", {})
        mesh_extractor = MeshExtractor(
            algorithm=mesh_extraction_config.get("algorithm", "mc"),
            device=mesh_extraction_config.get("device", str(device)),
            verbose=mesh_extraction_config.get("verbose", False),
        )
        mesh_extractor.set_grid(
            bbox=mesh_extraction_config.get("bbox", [[-5000, 5000], [-5000, 5000], [-5000, 5000]]),
            resolution=mesh_extraction_config.get("resolution", 128),
        )
        renderer_manager = VisualisationManager(verbose=verbose, output_dir=rendering_dir)

        # GINN / diversity config — absent or uses_diversity=False collapses to single-model mode
        ginn_config: dict[str, Any] = {}
        try:
            ginn_config = load_config(v="v2", cfg_name="GINN.yaml")
        except Exception as e:
            if uses_diversity:
                raise
            print(f"GINN config not found (ok for single-model training): {e}")

        training_params: dict[str, Any] = training_config["training_params"]
        learning_rate: float = training_params.get("learning_rate", 1e-3)
        num_epochs: int = training_params.get("num_epochs", 1000)

        if uses_diversity:  # double checks for diversity again, but serves as potential control split once we develop further
            population_size = ginn_config.get("population_size", 4)
            z_dim = model.get_model_info_dict()["z_dim"]
            diversity_loss_fn = default_field_diversity_loss  # TODO: make this configurable and add more options
        else:
            population_size = 1
            latent_vec = model.get_latent_vector()
            z_dim = int(latent_vec.numel()) if latent_vec is not None else 0
            diversity_loss_fn = None

        trainer = DiversityTrainer(
            model=model,
            total_loss=total_loss,
            alm_manager=alm_manager,
            population_size=population_size,
            z_dim=z_dim,
            device=device,
            model_lr=learning_rate,
            diversity_loss_fn=diversity_loss_fn,
            diversity_weight=ginn_config.get("diversity_weight", 0.1),
            diversity_num_points=ginn_config.get("diversity_num_points", 2048),
            diversity_bbox_size=ginn_config.get("diversity_bbox_size", 10.0),
            diversity_norm_order=ginn_config.get("diversity_norm_order", 2),
            diversity_neighbor_agg_fn=ginn_config.get("diversity_neighbor_agg_fn", "min"),
            force_z0_to_origin=ginn_config.get("force_z0_to_origin", True),
            z_resample_cadence=ginn_config.get("z_resample_cadence", 0),
            mesh_extractor=mesh_extractor,
            mesh_extraction_config=mesh_extraction_config,
            renderer_manager=renderer_manager,
            rendering_freq=wandb_cfg.get("rendering_frequency"),
            use_neural_renderer=wandb_cfg.get("use_neural_renderer", False),
            use_wandb=use_wandb,
            log_dir=rendering_dir,
        )

        interpolation_state = setup_latent_interpolator(
            model=model,
            device=device,
            general_config=general_config,
            run_locally=run_locally,
            rendering_dir=rendering_dir,
            verbose=verbose,
        )
        run_latent_interpolation(state=interpolation_state, epoch=0, stage="initial", verbose=verbose)

        if verbose:
            mode = f"diversity (N={population_size})" if uses_diversity else "single-model"
            print(f"Starting {mode} training loop for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            model.train()
            metrics = trainer.step(epoch=epoch)
            run_latent_interpolation(state=interpolation_state, epoch=epoch, stage="periodic", verbose=verbose)
            if verbose and epoch % 50 == 0:
                print(
                    f"Epoch {epoch}/{num_epochs} | per_vector={metrics['per_vector_loss']:.4f} | "
                    f"diversity={metrics['diversity_loss']:.4f} | total={metrics['total_loss']:.4f}"
                )

        run_latent_interpolation(state=interpolation_state, epoch=num_epochs, stage="final", verbose=verbose)
        cleanup_latent_interpolator(interpolation_state)

        if use_wandb and wandb_cfg.get("use_neural_renderer", False):
            try:
                renderer_manager.finalize()
            except Exception as e:
                print(f"Manager finalize failed: {e}")

    # 6. Saving of files and postprocessing

    # Reset latent vector to zeros before saving final model.
    if uses_diversity and isinstance(model, SupportsLatentConcat):
        model.set_latent_vector(torch.zeros_like(model.get_latent_vector()))

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
        epoch=training_config["training_params"].get("num_epochs", 1000),
        resolution=64,
        marching_cubes_usage=True,
    )
    print("Completed job successfully.")


if __name__ == "__main__":
    main()
