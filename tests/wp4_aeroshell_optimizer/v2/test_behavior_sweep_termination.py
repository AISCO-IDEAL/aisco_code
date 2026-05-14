"""Validation tests for behavior sweep launch readiness.

These tests are intentionally static and fast:
1) confirm both new sweep files exist and contain termination-oriented settings,
2) confirm both launch scripts request exactly one GPU and 6h wall-time,
3) confirm long-horizon behavior setup (CSE fixed on; high epoch budgets in general sweep),
4) confirm latent sweep avoids re-sweeping general knobs.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path("/cluster/home/eschaerer/AISCO_code")
_SWEEPS_DIR = _ROOT / "subprojects/wp4_aeroshell_optimizer/v2/configs/sweeps"
_SLURM_DIR = _ROOT / "subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts"

_SWEEP_GENERAL = _SWEEPS_DIR / "behavior_general_loss_schedule_sweep.yaml"
_SWEEP_LATENT = _SWEEPS_DIR / "behavior_latent_resample_sweep.yaml"

_LAUNCH_GENERAL = _SLURM_DIR / "launch_behavior_general_loss_sweep.sh"
_LAUNCH_LATENT = _SLURM_DIR / "launch_behavior_latent_resample_sweep.sh"


def _extract_epoch_values(sweep_text: str) -> list[int]:
    match = re.search(
        r"training_config\.training_params\.num_epochs:\s*\n\s*values:\s*\[([^\]]+)\]",
        sweep_text,
        flags=re.MULTILINE,
    )
    assert match is not None, "Missing training_config.training_params.num_epochs values list"
    values_raw = match.group(1)
    values = [int(part.strip()) for part in values_raw.split(",") if part.strip()]
    assert values, "Epoch values list must not be empty"
    return values


def _assert_common_termination_fields_from_text(sweep_path: Path) -> str:
    assert sweep_path.exists(), f"Missing expected sweep file: {sweep_path}"
    text = sweep_path.read_text()

    assert "program: subprojects/wp4_aeroshell_optimizer/v2/src/main_files/main_v2.py" in text
    assert re.search(r"^method:\s*(bayes|random|grid)\s*$", text, flags=re.MULTILINE)

    assert "early_terminate:" in text, "Sweep must define early_terminate"
    assert re.search(r"^\s*type:\s*hyperband\s*$", text, flags=re.MULTILINE), "Use hyperband early termination"

    min_iter_match = re.search(r"^\s*min_iter:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    assert min_iter_match is not None, "early_terminate.min_iter missing"
    min_iter = int(min_iter_match.group(1))
    assert min_iter >= 1, "early_terminate.min_iter must be positive"
    return text


def _assert_launcher_limits(script_path: Path) -> None:
    assert script_path.exists(), f"Missing expected launcher script: {script_path}"
    text = script_path.read_text()

    assert "#SBATCH --time=06:00:00" in text, f"Launcher must enforce 6h wall time: {script_path}"

    gpu_match = re.search(r"^#SBATCH --gpus=([^\n]+)$", text, flags=re.MULTILINE)
    assert gpu_match is not None, f"Launcher must define '#SBATCH --gpus=...': {script_path}"
    gpu_value = gpu_match.group(1).strip()
    assert gpu_value.endswith(":1"), f"Launcher must request exactly one GPU, got '{gpu_value}' in {script_path}"

    assert "CUDA_VISIBLE_DEVICES=0 wandb agent" in text, (
        f"Launcher must start exactly one agent on one visible device: {script_path}"
    )


def test_behavior_general_sweep_is_termination_bounded() -> None:
    text = _assert_common_termination_fields_from_text(_SWEEP_GENERAL)
    assert 'value: "True"' in text and "losses_v0_config.general_info.CSE" in text
    epoch_values = _extract_epoch_values(text)
    assert max(epoch_values) >= 1000, "General behavior sweep should include 1000+ epoch budget"


def test_behavior_latent_sweep_is_termination_bounded() -> None:
    text = _assert_common_termination_fields_from_text(_SWEEP_LATENT)
    assert 'value: "True"' in text and "losses_v0_config.general_info.CSE" in text


def test_latent_sweep_does_not_repeat_general_training_knobs() -> None:
    text = _SWEEP_LATENT.read_text()
    assert "training_config.training_params.learning_rate" not in text
    assert "training_config.training_params.num_epochs" not in text
    assert "losses_v0_config.general_info.deactivate_scheduler_mode" not in text
    assert "losses_v0_config.losses.aero._disabled_epochs" not in text
    assert "losses_v0_config.losses.solar._disabled_epochs" not in text


def test_behavior_launchers_enforce_single_gpu_and_6h() -> None:
    _assert_launcher_limits(_LAUNCH_GENERAL)
    _assert_launcher_limits(_LAUNCH_LATENT)
