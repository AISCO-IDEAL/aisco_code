from __future__ import annotations

import contextlib
import csv
import fcntl
import json
import os
import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from subprojects.wp4_aeroshell_optimizer.v2.logging_lib.image_uploads import upload_neural_renderer_gif
from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_latent import pretraining_latent
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config as _load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.device_detection import is_local_machine

_WP4_V2_ROOT = Path(__file__).resolve().parents[2]
_RUN_NAME_POOL_PATH = _WP4_V2_ROOT / "configs" / "german_run_names_10000.txt"
_USED_RUN_NAME_CSV_PATH = _WP4_V2_ROOT / "configs" / "used_german_run_names.csv"
_RUN_NAME_LOCK_PATH = _WP4_V2_ROOT / "configs" / ".german_run_names.lock"


def _consume_unique_german_name() -> str | None:
    """Consume one unique run-name token from the 10k pool.

    Returns a token in the form ``adjective-noun``. The selected token is
    removed from the source pool and appended to a used-names CSV so it cannot
    be reused.
    """
    if not _RUN_NAME_POOL_PATH.exists():
        raise RuntimeError(f"German run-name pool not found: {_RUN_NAME_POOL_PATH}")

    lock_file = _RUN_NAME_LOCK_PATH
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    with lock_file.open("a+", encoding="utf-8") as lock_handle:
        with contextlib.suppress(Exception):  # Best effort lock: if fcntl is unavailable, continue without it.
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)

        raw_lines = _RUN_NAME_POOL_PATH.read_text(encoding="utf-8").splitlines()
        available = [line.strip() for line in raw_lines if line.strip()]
        if not available:
            raise RuntimeError("German run-name pool is exhausted. Refill german_run_names_10000.txt before starting a new run.")

        idx = random.randrange(len(available))
        selected = available.pop(idx)

        tmp_fd, tmp_path = tempfile.mkstemp(prefix="german_run_names_", suffix=".tmp", dir=str(_RUN_NAME_POOL_PATH.parent))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="\n") as tmp_fh:
                if available:
                    tmp_fh.write("\n".join(available) + "\n")
            Path(tmp_path).replace(_RUN_NAME_POOL_PATH)
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink(missing_ok=True)

        _USED_RUN_NAME_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        csv_exists = _USED_RUN_NAME_CSV_PATH.exists()
        with _USED_RUN_NAME_CSV_PATH.open("a", encoding="utf-8", newline="") as used_fh:
            writer = csv.writer(used_fh)
            if not csv_exists:
                writer.writerow(["timestamp_utc", "name"])
            writer.writerow([datetime.now(UTC).isoformat(), selected])

        return selected


def _resolve_wandb_name(name_cfg: Any, run_locally: bool) -> str | None:  # noqa: PLR0911
    """Resolve wandb run name from config and runtime environment.

    Returns None to let wandb auto-generate a name, which is then appended with
    the SLURM job ID if on Euler.
    """
    if name_cfg is None:
        return None

    raw = str(name_cfg).strip()
    if raw == "":
        return None

    normalized = raw.lower()
    if normalized == "local_run_id":
        return f"local-{os.getpid()}"

    if normalized == "euler_run_id":
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        return f"euler-{slurm_job_id}" if slurm_job_id else None

    if normalized == "auto_run_id":
        if run_locally:
            return f"local-{os.getpid()}"
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        return f"euler-{slurm_job_id}" if slurm_job_id else None

    if normalized in {"german_run_id", "german_name", "german_run_name"}:
        return _consume_unique_german_name()

    if not run_locally:
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id and not raw.endswith(f"_{slurm_job_id}"):
            return f"{raw}_{slurm_job_id}"

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

    model, Z_pretrained = pretraining_latent(model, latent_config, device, use_wandb=use_wandb)
    return model, Z_pretrained
