"""Utility to write W&B run name to a file for shell script log renaming."""

import os
from pathlib import Path

import wandb


def write_wandb_run_name_to_file() -> None:
    """
    Write W&B run name to a temporary file that shell scripts can read.

    This is used by SLURM scripts to rename log files to include the W&B run name.
    The file is written to /tmp/wandb_run_name_{SLURM_JOB_ID}.txt

    Should be called after wandb.init().
    """
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if not slurm_job_id or wandb.run is None:
        return

    run_name_file = Path(f"/tmp/wandb_run_name_{slurm_job_id}.txt")
    try:
        run_name_file.write_text(wandb.run.name, encoding="utf-8")
    except Exception as e:
        print(f"[LOG_RENAMER] Warning: Could not write W&B run name to file: {e}")
