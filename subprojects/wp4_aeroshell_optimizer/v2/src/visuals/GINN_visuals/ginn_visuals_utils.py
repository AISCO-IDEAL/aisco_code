from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import wandb
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.storage_utils import reformat_wandb_name

if TYPE_CHECKING:
    from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat
    from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.archive.interpolator import Interpolator


InterpolationStage = Literal["initial", "periodic", "final"]


@dataclass
class LatentInterpolationState:
    interpolator: Interpolator | None
    config: dict[str, Any]
    save_dir: Path | None


def setup_latent_interpolator(
    model: SupportsLatentConcat,
    device: str,
    general_config: dict[str, Any],
    run_locally: bool,
    rendering_dir: Path | None,
    verbose: bool,
) -> LatentInterpolationState:
    """Create interpolation state from config with safe fallback behavior."""
    cfg_name = general_config.get("active_configs", {}).get("interpolator", "postprocessing_interpolator_config.yaml")

    try:
        loaded_cfg = load_config(v="v2", cfg_name=cfg_name)
        interpolator_config = dict(loaded_cfg.get("interpolator", {}))
    except Exception as exc:
        print(f"Warning: Failed to load interpolator config '{cfg_name}': {exc}")
        return LatentInterpolationState(interpolator=None, config={}, save_dir=None)

    explicit_save_dir = interpolator_config.get("save_dir")
    if explicit_save_dir:
        save_dir = Path(explicit_save_dir)
    elif rendering_dir is not None:
        save_dir = rendering_dir / "interpolation_grids"
    else:
        storage_paths = general_config.get("storage_paths", {})
        base_dir = storage_paths.get("local_storage_dir", "./local_storage_dir")
        if not run_locally:
            base_dir = storage_paths.get("storage_dir", "/cluster/work/fuge/AISCO/temp_")
        run_name = wandb.run.name if wandb.run is not None else "local_run"
        save_dir = Path(base_dir) / "interpolation_grids" / reformat_wandb_name(str(run_name))

    interpolator_config["save_dir"] = str(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.archive.interpolator import (  # noqa: PLC0415
            Interpolator,
        )

        interpolator = Interpolator(model=model, config=interpolator_config, device=device)
        if verbose:
            print(f"Interpolator initialized (backend={interpolator_config.get('render_backend', 'pyvista')})")
        return LatentInterpolationState(interpolator=interpolator, config=interpolator_config, save_dir=save_dir)
    except Exception as exc:
        print(f"Warning: Interpolator initialization failed: {exc}")
        return LatentInterpolationState(interpolator=None, config=interpolator_config, save_dir=save_dir)


def run_latent_interpolation(
    state: LatentInterpolationState,
    epoch: int,
    stage: InterpolationStage,
    verbose: bool,
) -> None:
    """Render/save/upload latent interpolation grids for initial, periodic, or final stages."""
    if state.interpolator is None:
        return

    if stage == "periodic":
        if not state.config.get("render_during_training", False):
            return
        render_frequency = int(state.config.get("render_frequency", 10))
        if render_frequency <= 0 or epoch <= 0 or epoch % render_frequency != 0:
            return

    try:
        state.interpolator.generate_interpolation_grid(epoch=epoch)

        if state.save_dir is not None:
            filename = "interpolation_grid_final.png" if stage == "final" else f"interpolation_grid_epoch{epoch}.png"
            save_path = state.save_dir / filename
            state.interpolator.save_interpolation_grid(filepath=save_path, clear_cache=False)
            if verbose:
                print(f"Saved latent interpolation grid to {save_path}")

        if state.config.get("upload_to_wandb", True) and wandb.run is not None:
            key = "renders/latent_interpolation_grid_final" if stage == "final" else "renders/latent_interpolation_grid"
            state.interpolator.upload_to_wandb(key=key, clear_cache=False, step=epoch)

        state.interpolator.clear_image_cache()
    except Exception as exc:
        print(f"Warning: Latent interpolation failed at stage='{stage}' epoch={epoch}: {exc}")


def cleanup_latent_interpolator(state: LatentInterpolationState) -> None:
    """Release interpolation resources."""
    if state.interpolator is None:
        return
    try:
        state.interpolator.cleanup()
    except Exception as exc:
        print(f"Warning: Failed to clean up interpolator: {exc}")
