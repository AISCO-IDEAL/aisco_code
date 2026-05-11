from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from subprojects.wp4_aeroshell_optimizer.v2.logging_lib.image_uploads import upload_neural_renderer_gif
from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_latent import pretraining_latent
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config as _load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.device_detection import is_local_machine

def _resolve_wandb_name(name_cfg: Any, run_locally: bool) -> str | None:
    """Resolve wandb run name from config and runtime environment."""
    if name_cfg is None:
        return None

    raw = str(name_cfg).strip()
    if raw == "":
        return None

    normalized = raw.lower()
    if normalized == "local_run_id":
        return f"local-{os.getpid()}"

    if normalized == "euler_run_id":
        return f"euler-{os.environ.get('SLURM_JOB_ID', 'no-slurm-id')}"

    if normalized == "auto_run_id":
        if run_locally:
            return f"local-{os.getpid()}"
        return f"euler-{os.environ.get('SLURM_JOB_ID', 'no-slurm-id')}"

    return raw

def _upload_mesh_gradient_gifs(vis_manager: Any, use_wandb: bool) -> None:
    """Upload finalize-generated mesh-gradient GIFs to wandb."""
    if not use_wandb or vis_manager is None:
        return
    run_dir = getattr(vis_manager, "run_dir", None)
    if run_dir is None:
        return
    grad_root = run_dir / "mesh_gradients"
    if not grad_root.is_dir():
        return
    for gif in sorted(grad_root.rglob("animation.gif")):
        loss_key = gif.parent.name
        try:
            upload_neural_renderer_gif(str(gif), key=f"mesh_gradients/{loss_key}", step=0)
        except Exception as exc:
            print(f"mesh_gradients GIF upload failed for {loss_key}: {exc}")


def _maybe_run_latent_pretraining(
    model: Any,
    general_config: dict[str, Any],
    device: str,
    *,
    use_wandb: bool = False,
) -> tuple[Any, torch.Tensor | None]:
    """Run latent-population pretraining if configured and model supports it."""
    latent_cfg_name: str = general_config.get("active_configs", {}).get("latent_pretraining", "")
    if not latent_cfg_name:
        return model, None

    if not isinstance(model, SupportsLatentConcat):
        print(
            f"WARNING: latent_pretraining='{latent_cfg_name}' is configured but model "
            f"{type(model).__name__} does not implement SupportsLatentConcat. "
            "Skipping latent pretraining."
        )
        return model, None

    root = str(Path.home() / "beast_logs") if is_local_machine() else "/cluster/work/fuge/AISCO/latent_pretraining"
    raw = json.dumps(_load_config(v="v2", cfg_name=latent_cfg_name))
    latent_config = json.loads(raw.replace("${LATENT_DATA_ROOT}", root))

    model, Z_pretrained = pretraining_latent(model, latent_config, device, use_wandb=use_wandb, vis_sampling=False)
    return model, Z_pretrained
