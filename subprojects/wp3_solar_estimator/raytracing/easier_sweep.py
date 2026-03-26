


from __future__ import annotations

import os
from typing import Any

import torch

import wandb
from subprojects.wp3_solar_estimator.v7.solar_loss_v7 import SolarLossEstimator
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import (
    apply_sweep_overrides,
    get_master_config,
    load_config,
    register_all_configs,
)
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.dynamic_loading import dynamic_type_loading
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.storage_utils import (
    allround_save_data,
    fetch_pretrained_model,
)


def _init_wandb(general_config: dict[str, Any]) -> None:
    """Initialize wandb if enabled in config."""
    use_wandb = bool(general_config.get("wandb_general", {}).get("use_wandb", False))
    if use_wandb:
        wandb.init(
            entity=general_config["wandb_general"]["entity_name"],
            project=general_config["wandb_general"]["project_name"],
            config=general_config,
        )


def _register_and_apply_sweep_overrides(general_config: dict[str, Any]) -> None:
    """Register all configs, then apply wandb sweep overrides when present."""
    register_all_configs(general_config, v="v1")
    if wandb.run is not None:
        apply_sweep_overrides(dict(wandb.config))
        wandb.config.update(get_master_config(), allow_val_change=True)


def _instantiate_model(general_config: dict[str, Any]) -> torch.nn.Module:
    """Instantiate the configured model class with sweep-aware config."""
    model_cfg_name = general_config["active_configs"]["model_params"]
    model_config = load_config(v="v1", cfg_name=model_cfg_name)

    model_name = general_config["model_def"]["model_name"]
    model_definition = general_config["model_def"]["model_file_path"]
    model_class = dynamic_type_loading(file_path=model_definition, class_name=model_name)

    model = model_class(**model_config)
    device = general_config["general_info"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def _load_pretrained_or_raise(model: torch.nn.Module, general_config: dict[str, Any]) -> None:
    """Load matching pretrained weights into model, fail fast if unavailable."""
    pre_cfg_name = general_config["active_configs"].get("pretraining", "pretraining_v0_config.yaml")
    pre_cfg = load_config(v="v1", cfg_name=pre_cfg_name)

    run_locally = general_config["general_info"].get("run_locally", "False") == "True"
    if run_locally:
        pretrained_path = pre_cfg["pretraining"].get("local_pretraining_model_path", None)
    else:
        pretrained_path = pre_cfg["pretraining"].get("pretraining_model_path", None)

    pretrained_model = fetch_pretrained_model(model, pretrained_path=pretrained_path, verbose=True)
    if pretrained_model is None:
        raise FileNotFoundError(
            "No pretrained model found for this architecture. "
            "This one-shot sweep runner requires an existing pretrained checkpoint."
        )

    device = general_config["general_info"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(pretrained_model, map_location=device))
    print(f"Loaded pretrained model from {pretrained_model}")


def _run_one_shot_solar(model: torch.nn.Module, general_config: dict[str, Any]) -> tuple[torch.Tensor, SolarLossEstimator]:
    """Run exactly one solar loss evaluation using SolarLossConfigV7."""
    solar_cfg_path = general_config["active_configs"]["solar"]
    solar_cfg = load_config(v="v1", cfg_name=solar_cfg_path)["SolarLossConfigV7"]
    solar_estimator = SolarLossEstimator(config=solar_cfg)

    model.eval()
    loss = solar_estimator.solar_loss(model=model, epoch_name=0)
    return loss, solar_estimator


def _finalize_rendering_if_enabled(solar_estimator: SolarLossEstimator) -> None:
    """Finalize rendering outputs if the estimator created a visualization manager."""
    if not getattr(solar_estimator, "rendering", False):
        return

    manager = getattr(solar_estimator, "visualisation_manager", None)
    if manager is not None:
        try:
            manager.finalize(custom_name="one_shot")
        except Exception as exc:
            print(f"Warning: rendering finalize failed: {exc}")


def _save_artifacts(model: torch.nn.Module, general_config: dict[str, Any]) -> None:
    """Save model and mesh using the same helper used by the full pipeline."""
    training_cfg_name = general_config["active_configs"].get("training", "training_v0_config.yaml")
    training_config = load_config(v="v1", cfg_name=training_cfg_name)
    epoch = int(training_config["training_params"].get("num_epochs", 1000))

    wandb_name = wandb.run.name if wandb.run is not None else None
    allround_save_data(
        model=model,
        storage_path=general_config["general_info"].get(
            "storage_path", "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/bulk_storage"
        ),
        save_model=True,
        save_mesh=True,
        verbose=bool(general_config["general_info"].get("verbose", True)),
        pretraining=False,
        wandb_run=wandb_name,
        epoch=epoch,
    )


def main() -> None:
    general_config = load_config(v="v1", cfg_name="general_config.yaml")

    _init_wandb(general_config)
    _register_and_apply_sweep_overrides(general_config)

    # Reload after overrides so we use the patched config for everything below.
    general_config = load_config(v="v1", cfg_name="general_config.yaml")

    model = _instantiate_model(general_config)
    _load_pretrained_or_raise(model, general_config)

    loss, solar_estimator = _run_one_shot_solar(model, general_config)
    print(f"One-shot solar loss: {float(loss.detach().item()):.8f}")

    if wandb.run is not None:
        wandb.log({"one_shot/solar_loss": float(loss.detach().item())}, step=0)

    _finalize_rendering_if_enabled(solar_estimator)
    _save_artifacts(model, general_config)

    if wandb.run is not None:
        wandb.finish()

    print("Completed one-shot pretrained solar evaluation.")


if __name__ == "__main__":
    # Preserve current working-directory behavior used by existing launch scripts.
    os.chdir(os.getcwd())
    main()