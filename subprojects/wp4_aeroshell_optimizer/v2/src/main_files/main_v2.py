"""
This should be the final main file we use for the Project. It should be modular and easily extendible
"""

import argparse
import os
from pathlib import Path
from typing import Any

import torch

import wandb
from subprojects.wp4_aeroshell_optimizer.v2.scripts.postprocess_gradient_conflict import build_conflict_gif
from subprojects.wp4_aeroshell_optimizer.v2.src.GINN.diversity_manager import DiversityTrainer
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.alm import ALMManager
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.diversity_loss_manager import DiversityLossManager
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.total_loss_v2 import TotalLossV2
from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.main_v2_utils import (
    _upload_mesh_gradient_gifs,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_latent import (
    _compute_latent_pretrain_hash,
    pretraining_latent,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_v1_1 import (
    _compute_solitary_pretrain_hash,
    _resolve_solitary_ckpt_path,
    pretraining,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat, SupportsOptimise
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import (
    NullVisManager,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import (
    CentralVisManagerV2,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import (
    apply_sweep_overrides,
    get_master_config,
    load_config,
    register_all_configs,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config as _load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.debug_manager import DebugManager
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.device_detection import is_cluster
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.dynamic_loading import smart_model_init
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.gradient_conflict_tracker import GradientConflictTracker
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.hashing import dict_hashing
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.log_file_renamer import (
    rename_log_file_with_wandb_run_name,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.pyvista_headless import (
    configure_headless_pyvista,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.storage_utils import (
    allround_save_data,
    reformat_wandb_name,
    save_population_meshes,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.wandb_utils import patch_wandb_log_for_monotonic_steps
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.visual_wrapper import VisualisationManager

configure_headless_pyvista()


def main(_gui_mode: bool = False, _cfg_folder: str | None = None) -> None:  # noqa: PLR0912, PLR0915

    print("Starting Main file V2...")

    # 1. Load and register configs
    general_config: dict[str, Any] = load_config(v="v2", cfg_name="general_config.yaml")
    # Auto-detect environment: override run_locally if not on a known cluster
    if not is_cluster():
        general_config["general_info"]["run_locally"] = "True"
    verbose: bool = general_config.get("general_info", {}).get("verbose", True)
    device: str = general_config.get("general_info", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print(f"Configured device '{device}' is unavailable. Falling back to CPU.")
        device = "cpu"
    latent_space: bool = bool(general_config["general_info"].get("latent_space", False))
    run_locally: bool = general_config["general_info"].get("run_locally", "False") == "True"

    # Init Elias' debug manager NOT WORKING PROPERLY ATM, PLEASE CHECK
    debug_cfg = load_config(v="v2", cfg_name="debug.yaml")
    debug_manager = DebugManager(config=debug_cfg)
    if verbose:
        print(debug_manager.summary())

    # Setup profiler config
    profiler_cfg = general_config.get("profiler", {})
    profile_mode = profiler_cfg.get("profile_mode", False)
    prof_wait = profiler_cfg.get("wait", 5)
    prof_warmup = profiler_cfg.get("warmup", 2)
    prof_active = profiler_cfg.get("active", 10)
    prof_repeat = profiler_cfg.get("repeat", 1)
    prof_top_entries = profiler_cfg.get("top_entries", 30)
    if profile_mode and verbose:
        print(
            f"Performance profiler enabled with schedule: wait={prof_wait}, warmup={prof_warmup}, active={prof_active}, repeat={prof_repeat}, top_entries={prof_top_entries}"
        )
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

            # Rename log file IMMEDIATELY after W&B init to include run name
            print("[CHECKPOINT] About to call rename_log_file_with_wandb_run_name()", flush=True)
            rename_log_file_with_wandb_run_name()
            print("[CHECKPOINT] rename_log_file_with_wandb_run_name() completed", flush=True)

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

    # Re-read flags that may have been overridden by the sweep so the rest of
    # main() respects the overridden values (e.g. latent_space, run_locally).
    _raw_latent_space = general_config["general_info"].get("latent_space", False)
    if isinstance(_raw_latent_space, str):
        latent_space = _raw_latent_space.strip().lower() in {"true", "1", "yes", "on"}
    else:
        latent_space = bool(_raw_latent_space)
    _raw_run_locally = general_config["general_info"].get("run_locally", "False")
    run_locally = str(_raw_run_locally).strip().lower() in {"true", "1", "yes", "on"}

    if verbose:
        print("Successfully loaded and registered all configs")

    # 3. Load model and handle pretraining
    model_name: str
    if latent_space:
        model_name = general_config["model_def"]["concat_model_name"]
        print(f"Using latent space model: {model_name}")
    else:
        model_name = general_config["model_def"]["model_name"]
        print(f"Using standard model: {model_name}")
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

    # Shared rendering-dir logic (used by both legacy and diversity paths)
    model_hash = dict_hashing(model.get_model_info_dict())
    wandb_name = wandb.run.name if wandb.run is not None else None
    rendering_dir: Path | None = None
    if wandb_name is not None:
        if run_locally:
            rendering_dir = (
                Path(
                    general_config["storage_paths"].get("beast_storage_dir")
                    or general_config["storage_paths"].get("local_storage_dir", "./local_storage_dir")
                )
                / model_hash
                / reformat_wandb_name(wandb_name)
            )
        else:
            rendering_dir = Path(general_config["storage_paths"]["storage_dir"]) / model_hash / reformat_wandb_name(wandb_name)
        os.makedirs(rendering_dir, exist_ok=True)

    # Construct vis manager once per run.
    # vis_manager: "v2" | "null" (default: "disabled" raises ValueError)
    # "v2"  → CentralVisManagerV2 (async PLY writes, ProcessPool GIFs)
    # "null" → NullVisManager (no-op; is_enabled() always False — dry-run)
    _vis_mode = general_config.get("vis_manager", "disabled")
    _vis_manager: CentralVisManagerV2 | NullVisManager | None = None

    if _vis_mode == "v2" and rendering_dir is None:
        # If CVM requested but wandb is off, create fallback vis dir.
        _fallback_dir = (
            Path(general_config["storage_paths"].get("local_storage_dir", "./local_storage_dir")) / model_hash / "vis_fallback"
        )
        rendering_dir = _fallback_dir

    if rendering_dir is not None and _vis_mode == "v2":
        _run_id = wandb_name or "unknown"
        _vis_manager = CentralVisManagerV2(run_dir=rendering_dir, run_id=_run_id)
        print(f"[VisManager:v2] enabled → {rendering_dir}")
    elif _vis_mode == "null":
        _vis_manager = NullVisManager()
        print("[VisManager:null] dry-run mode — all writes are no-ops")
    else:
        print(f"[VisManager] disabled (vis_manager={_vis_mode!r})")
        raise ValueError(f"Invalid vis_manager mode: {_vis_mode}. Expected 'v2', 'legacy', 'null', you can't disable it entirely")
    assert _vis_manager is not None, (
        "Vis manager should be initialized at this point. Check the logic for vis_manager initialization."
    )
    # 3b. Handle pretraining.
    # latent_space=true  → pretraining_latent.py (handles fast-resume internally; checkpoint in latent_pretraining dir)
    # latent_space=false → standard pretraining_v1_1.py (checkpoint in pretrained_path)
    Z_pretrained: torch.Tensor | None = None
    latent_config: dict[str, Any] = {}

    if latent_space:
        if not isinstance(model, SupportsLatentConcat):
            raise TypeError(
                f"latent_space=true requires a SupportsLatentConcat model, got {type(model).__name__}. "
                "Set model_name to a concat model (e.g. ConcatWIREV2) in general_config.yaml."
            )
        latent_cfg_name: str = general_config["active_configs"].get("latent_pretraining", "pretraining_latent_config.yaml")
        latent_config = _load_config(v="v2", cfg_name=latent_cfg_name)
        # Vis dir: derive from canonical pretraining run folder (never /tmp).
        # Computed before calling pretraining_latent so GIF generation has the path.
        pretrained_hash: str = _compute_latent_pretrain_hash(latent_config, model)
        model, Z_pretrained = pretraining_latent(
            model,
            latent_config,
            device,
            use_wandb=use_wandb,
            vis_manager=_vis_manager,
        )
        if verbose:
            print(f"[latent_space] Latent pretraining complete. Population size: {Z_pretrained.shape[0]}")

    else:
        # Standard pretraining: hash-based canonical lookup (Plan 031 / M4).
        pretraining_config_name = general_config["active_configs"].get("pretraining", "pretraining_v0_config.yaml")
        pretraining_config = load_config(v="v2", cfg_name=pretraining_config_name)
        force_pretraining = pretraining_config["pretraining"].get("force_pretraining", False)

        # Canonical base dir for solitary pretraining (environment-aware)
        if run_locally:
            _local_root = general_config["storage_paths"].get("beast_storage_dir") or general_config["storage_paths"].get(
                "local_storage_dir", "./local_storage_dir"
            )
            solitary_base_dir = Path(_local_root) / "pretraining" / "solitary_pretraining"
        else:
            solitary_base_dir = Path("/cluster/work/fuge/AISCO/pretraining/solitary_pretraining")

        pretrained_hash = _compute_solitary_pretrain_hash(pretraining_config, model)
        pretrained_model_path: Path = _resolve_solitary_ckpt_path(pretraining_config, model, solitary_base_dir)

        if pretrained_model_path.exists() and not force_pretraining:
            ckpt = torch.load(pretrained_model_path, map_location=device, weights_only=True)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
            if verbose:
                print(f"[pretraining] LOAD  path={pretrained_model_path}  hash={pretrained_hash}  status=fast_resume")
        else:
            if verbose and not force_pretraining:
                print(f"[pretraining] MISS  path={pretrained_model_path}  hash={pretrained_hash}  action=train")
            if verbose and force_pretraining:
                print(f"[pretraining] Force pretraining enabled. hash={pretrained_hash}")

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

            model = pretraining(
                model=model,
                debug_manager=debug_manager,
                pretraining_config_name=pretraining_config_name,
                canonical_ckpt_path=pretrained_model_path,
            )
    model = model.to(device)

    if verbose:
        print(f"Pretraining complete for model: {model_class_name} with Parent class: {parent_class_name}")
        print(f"Detailed settings: {[f'{key}: {value}' for key, value in settings.items()]}")

    # 4. Initialise whatever we need to run the main training loop, e.g. trainer, loss function, etc.

    training_config = load_config(v="v2", cfg_name=general_config["active_configs"].get("training", None))
    _mode_cfg: dict = training_config.get("mode", {})
    diagnostic_mode: bool = _mode_cfg.get("training_mode", "diagnostic") == "diagnostic"
    print_timing: bool = _mode_cfg.get("print_timing", False)

    # Gradient conflict tracking setup
    conflict_cadence: int = training_config["training_params"].get("gradient_conflict_cadence", 0)
    conflict_tracker: GradientConflictTracker | None = None
    conflict_output_dir: Path | None = None
    if conflict_cadence > 0:
        conflict_output_dir_str = training_config["training_params"].get("gradient_conflict_output_dir", "")
        if conflict_output_dir_str:
            conflict_output_dir = Path(conflict_output_dir_str)
        elif wandb_name is not None:
            if run_locally:
                base_dir = Path(
                    general_config["storage_paths"].get("beast_storage_dir")
                    or general_config["storage_paths"].get("local_storage_dir", "./local_storage_dir")
                )
            else:
                base_dir = Path(general_config["storage_paths"]["storage_dir"])
            conflict_output_dir = base_dir / model_hash / reformat_wandb_name(wandb_name) / "gradient_conflict"
        else:
            conflict_output_dir = Path("/tmp/gradient_conflict")
        conflict_output_dir.mkdir(parents=True, exist_ok=True)
        conflict_tracker = GradientConflictTracker(conflict_output_dir)
        if verbose:
            print(f"Gradient conflict tracking enabled (cadence={conflict_cadence}, output={conflict_output_dir}")

    # 5. Run main training loop

    if has_optimise:
        assert isinstance(model, SupportsOptimise), (
            "Model is marked with has_optimise=True but does not adhere to SupportsOptimise contract."
        )

        renderer_manager = VisualisationManager(verbose=verbose, output_dir=rendering_dir)
        print("Model has legacy optimise() method, assuming single batch optimisation is desired. \n Starting training...")

        prof = None
        if profile_mode:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=prof_wait, warmup=prof_warmup, active=prof_active, repeat=prof_repeat),
                on_trace_ready=None,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            prof.start()

        model.optimise(
            training_config=training_config,
            renderer_manager=renderer_manager,
            wandb_name=wandb_name,
            general_config=general_config,
            debug_manager=debug_manager,
            conflict_tracker=conflict_tracker,
            conflict_cadence=conflict_cadence,
            vis_manager=_vis_manager,
            prof=prof,
        )

        # ------------------------------------------------------------------
        # 9. Finalize renderers
        # ------------------------------------------------------------------
        # CentralVisManager finalize: generate convergence GIFs for all visual types.
        if _vis_manager is not None:
            try:
                _vis_manager.finalize()
            except Exception as e:
                print(f"CentralVisManager finalize failed: {e}")
            _upload_mesh_gradient_gifs(_vis_manager, use_wandb)

        if use_wandb and general_config["wandb_general"].get("use_neural_renderer", False):
            try:
                renderer_manager.finalize()
            except Exception as e:
                print(f"Manager finalize failed: {e}")

    elif uses_diversity:
        # New path: TotalLossV2 + optional ALM + optional DiversityLossManager + DiversityTrainer.
        if not isinstance(model, SupportsLatentConcat):
            raise TypeError(
                "Non-legacy models must implement SupportsLatentConcat (set_latent_vector / "
                "get_latent_vector) to use the TotalLossV2 + DiversityTrainer pipeline."
            )

        losses_config = load_config(v="v2", cfg_name=general_config["active_configs"]["losses"])
        wandb_cfg: dict[str, Any] = general_config.get("wandb_general", {})

        total_loss = TotalLossV2(device=device, vis_manager=_vis_manager)
        diversity_loss_manager = DiversityLossManager(losses_config=losses_config, device=device)

        # ALM setup
        use_alm: bool = losses_config["general_info"].get("ALM") == "True"
        alm_manager: ALMManager | None = None
        if use_alm:
            # get all constraint categories that are defined in the loss config, regardless of whether they are active or not. This is needed for the ALM manager to properly map the initial multipliers to the active constraints based on their categories, and also to know which constraints exist, even if not all are active.
            configured_constraint_categories: list[str] = [
                category
                for category, block in losses_config.get("losses", {}).items()
                if isinstance(block, dict) and block.get("_role") == "constraint"
            ]
            alm_manager = ALMManager(
                total_loss.constraint_names,
                losses_config["general_info"],
                torch.device(device),
                active_constraint_categories=total_loss.constraint_category_order,
                configured_constraint_categories=configured_constraint_categories,
            )

        renderer_manager = VisualisationManager(verbose=verbose, output_dir=rendering_dir)

        # GINN / diversity config — absent or uses_diversity=False collapses to single-model mode
        ginn_config: dict[str, Any] = {}
        try:
            ginn_config = load_config(v="v2", cfg_name="training_config.yaml")
        except Exception as e:
            if uses_diversity:
                raise
            print(f"GINN config not found (ok for single-model training): {e}")

        training_params: dict[str, Any] = training_config["training_params"]
        learning_rate: float = float(training_params.get("learning_rate", 1e-3))
        num_epochs: int = int(training_params.get("num_epochs", 1000))

        if uses_diversity:  # double checks for diversity again, but serves as potential control split once we develop further
            if latent_space and Z_pretrained is not None:
                # Use actual checkpoint N — correct even when use_specific_pretraining overrides latent_population length.
                population_size = Z_pretrained.shape[0]
            elif latent_space:
                population_size = len(latent_config["latent_population"])
            else:
                population_size = ginn_config.get("population_size", 4)
            z_dim = model.get_model_info_dict()["z_dim"]
        else:
            population_size = 1
            latent_vec = model.get_latent_vector()
            z_dim = int(latent_vec.numel()) if latent_vec is not None else 0

        trainer = DiversityTrainer(
            model=model,
            total_loss=total_loss,
            alm_manager=alm_manager,
            population_size=population_size,
            z_dim=z_dim,
            device=device,
            model_lr=learning_rate,
            diversity_manager=diversity_loss_manager,
            diversity_num_points=ginn_config.get("diversity_num_points", 2048),
            diversity_norm_order=ginn_config.get("diversity_norm_order", 2),
            diversity_neighbor_agg_fn=ginn_config.get("diversity_neighbor_agg_fn", "min"),
            force_z0_to_origin=ginn_config.get("force_z0_to_origin", True),
            z_resample_cadence=ginn_config.get("z_resample_cadence", 0),
            z_resample_mode=ginn_config.get("z_resample_mode", "on_circle"),
            initial_Z=Z_pretrained,
            Z_pretrained=Z_pretrained,
            z_resample_start_epoch=ginn_config.get("z_resample_start_epoch", 0),
            renderer_manager=renderer_manager,
            rendering_freq=wandb_cfg.get("rendering_frequency"),
            diversity_bbox_scale_factor=ginn_config.get("bbox_scale_factor", 1.0),
            use_neural_renderer=wandb_cfg.get("use_neural_renderer", False),
            use_wandb=use_wandb,
            log_dir=rendering_dir,
            conflict_tracker=conflict_tracker,
            conflict_cadence=conflict_cadence,
            diagnostic_mode=diagnostic_mode,
            print_timing=print_timing,
            vis_manager=_vis_manager,
            vis_cadence_optimisation=int(
                (_vis_manager.latent_optimisation_cfg if hasattr(_vis_manager, "latent_optimisation_cfg") else {}).get(
                    "vis_cadence_optimisation", 0
                )
            ),
            vis_strategy=str(
                (_vis_manager.latent_optimisation_cfg if hasattr(_vis_manager, "latent_optimisation_cfg") else {}).get(
                    "vis_strategy", "dynamic"
                )
            ),
            vis_all_losses=bool(
                (_vis_manager.latent_optimisation_cfg if hasattr(_vis_manager, "latent_optimisation_cfg") else {}).get(
                    "vis_all_losses", False
                )
            ),
        )

        if verbose:
            mode = f"diversity (N={population_size})" if uses_diversity else "single-model"
            print(f"Starting {mode} training loop for {num_epochs} epochs...")

        prof = None
        if profile_mode:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=prof_wait, warmup=prof_warmup, active=prof_active, repeat=prof_repeat),
                on_trace_ready=None,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            prof.start()

        enforce_cfg = losses_config.get("general_info", {}).get("enforce_projection", {})
        activation_ratio = enforce_cfg.get("activation_epoch_ratio", 0.9)
        activation_epoch = int(num_epochs * activation_ratio)

        for epoch in range(num_epochs):
            if epoch == activation_epoch and hasattr(model, "enforce_layer") and not model.enforce_layer.is_active:
                if verbose:
                    print(f"Activating ENFORCE projection layer at epoch {epoch} / {num_epochs}")
                model.enforce_layer.is_active = True

            model.train()
            metrics = trainer.step(epoch=epoch)

            mesh_save_interval = training_config["training_params"].get("mesh_save_interval", 50)
            if (epoch + 1) % mesh_save_interval == 0:
                allround_save_data(
                    model=model,
                    storage_path=(
                        general_config["storage_paths"].get("beast_storage_dir")
                        or general_config["storage_paths"].get("local_storage_dir", "./local_storage_dir")
                    )
                    if run_locally
                    else general_config["storage_paths"]["storage_dir"],
                    save_model=True,
                    save_mesh=True,
                    epoch=epoch + 1,
                    verbose=general_config["general_info"].get("verbose", False),
                    pretraining=False,
                    wandb_run=wandb_name,
                    device=str(device),
                    z_pretrained=Z_pretrained if latent_space else None,
                )

            if verbose and epoch % 50 == 0:  # arbitray number, adjust as needed
                print(
                    f"Epoch {epoch}/{num_epochs} | per_vector={metrics['per_vector_loss']:.4f} | "
                    f"diversity={metrics['diversity_loss']:.4f} | total={metrics['total_loss']:.4f}"
                )
            if prof is not None:
                prof.step()

        # CentralVisManager finalize: generate convergence GIFs for all visual types.
        if _vis_manager is not None:
            try:
                _vis_manager.finalize()
            except Exception as e:
                print(f"CentralVisManager finalize failed: {e}")
            _upload_mesh_gradient_gifs(_vis_manager, use_wandb)

        if use_wandb and wandb_cfg.get("use_neural_renderer", False):
            try:
                renderer_manager.finalize()
            except Exception as e:
                print(f"Manager finalize failed: {e}")

    # 6. Saving of files and postprocessing

    # Post-process gradient conflict data into GIF if tracking was active
    if conflict_tracker is not None and conflict_output_dir is not None:
        try:
            gif_path = conflict_output_dir / "conflict_evolution.gif"
            build_conflict_gif(conflict_output_dir, gif_path, fps=5)
            if verbose:
                print(f"Gradient conflict GIF saved to {gif_path}")
        except Exception as e:
            print(f"Warning: Failed to post-process gradient conflict data: {e}")

    # Optional: save one PLY mesh per latent slot for population validation.
    # Gated by training_config.yaml: save_population_meshes (default false to avoid overhead).
    if uses_diversity and isinstance(model, SupportsLatentConcat) and ginn_config.get("save_population_meshes", False):
        final_Z = trainer.get_population()
        save_population_meshes(
            model=model,
            population_z=final_Z,
            storage_path=(
                general_config["storage_paths"].get("beast_storage_dir")
                or general_config["storage_paths"].get("local_storage_dir", "./local_storage_dir")
            )
            if run_locally
            else general_config["storage_paths"].get(
                "storage_dir", "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/bulk_storage"
            ),
            wandb_run=wandb_name,
            epoch=training_config["training_params"].get("num_epochs", 420420),
            resolution=128,
            verbose=verbose,
            device=str(device),
        )

    # Reset latent vector to zeros before saving final model.
    if uses_diversity and isinstance(model, SupportsLatentConcat):
        model.set_latent_vector(torch.zeros_like(model.get_latent_vector()))

    allround_save_data(
        model=model,
        storage_path=(
            general_config["storage_paths"].get("beast_storage_dir")
            or general_config["storage_paths"].get("local_storage_dir", "./local_storage_dir")
        )
        if run_locally
        else general_config["storage_paths"].get("storage_dir", "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/bulk_storage"),
        save_model=True,
        save_mesh=True,
        verbose=verbose,
        pretraining=False,
        wandb_run=wandb_name,
        epoch=training_config["training_params"].get(
            "num_epochs", 420420
        ),  # Use 420420 to avoid default value that is misleading
        resolution=128,
        device=str(device),
        z_pretrained=Z_pretrained if latent_space else None,
    )

    # Stop profiler after ALL training, rendering, and saving is complete
    if prof is not None:
        prof.stop()

        profiler_cfg = general_config.get("profiler", {})
        if profiler_cfg.get("export_trace", False):
            trace_path = rendering_dir / "trace.json" if rendering_dir else Path("trace.json")
            prof.export_chrome_trace(str(trace_path))
            if verbose:
                print(f"Exported profiler trace to {trace_path}")

        print("\n" + "=" * 80)
        print("PROFILER RESULTS: CUDA TIME (GPU Operations)")
        print("=" * 80)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=prof_top_entries))
        print("\n" + "=" * 80)
        print("PROFILER RESULTS: CPU TIME (Host Operations)")
        print("=" * 80)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=prof_top_entries))
        print("=" * 80 + "\n")

    print("Completed job successfully.")


if __name__ == "__main__":
    # CLI argument parsing (for GUI config override infrastructure)
    parser = argparse.ArgumentParser(description="AISCO WP4 Aeroshell Optimizer")
    parser.add_argument("--uses-gui", action="store_true", help="Running from GUI")
    parser.add_argument("--cfg-folder", type=str, default="", help="Path to config override folder")
    args, _ = parser.parse_known_args()
    print(f"CLI arguments received: {args}, type: {type(args)}")
    print(f"Extras: {_}, type: {type(_)}")
    # Arguments are available but not used yet - caller will handle them
    print(f"uses_gui: {args.uses_gui}, cfg_folder: {args.cfg_folder}")
    # NOTE: When launched by `wandb agent`, the agent appends sweep hyperparameter
    # overrides as `--key=value` CLI args. These are consumed via `wandb.config`
    # in `apply_sweep_overrides`, not via argparse, so we tolerate `--...`-style
    # extras here. Only non-flag extras indicate a real caller bug.
    _bad_extras = [a for a in _ if not a.startswith("--")]
    assert _bad_extras == [], (
        f"Unexpected non-flag CLI arguments received: {_bad_extras}. Check the caller or argument parsing logic."
    )
    # --cfg-folder is only required when running through the GUI override flow.
    if args.uses_gui:
        assert args.cfg_folder != "", '--uses-gui requires "--cfg-folder PATH" to be set'
    assert " " not in args.cfg_folder, f"Config folder path cannot contain spaces. Received: '{args.cfg_folder}'"

    main(_gui_mode=args.uses_gui, _cfg_folder=args.cfg_folder)
