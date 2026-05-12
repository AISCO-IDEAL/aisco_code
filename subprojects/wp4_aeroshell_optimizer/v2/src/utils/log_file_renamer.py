"""Utility to rename SLURM log files to include W&B run name immediately after init."""

import os
import sys
from pathlib import Path

import wandb


def rename_log_file_with_wandb_run_name() -> None:
    """
    Rename the SLURM log file immediately after wandb.init() to include the W&B run name.

    On Linux, renaming an open file does not affect the open file descriptor —
    SLURM holds the log file open by inode and continues writing to it under
    the new name.  This means the rename is safe to do at any point and all
    output (before and after the call) ends up in the final file.

    Expects environment variables set by the SLURM script:
        - SLURM_JOB_ID:   SLURM job ID
        - SLURM_JOB_NAME: SLURM job name  (set by #SBATCH --job-name)
        - SLURM_LOG_DIR:  Directory where logs are stored

    Should be called immediately after wandb.init().
    """
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    slurm_job_name = os.environ.get("SLURM_JOB_NAME", "")
    log_dir = os.environ.get("SLURM_LOG_DIR", "")

    if not all([slurm_job_id, slurm_job_name, log_dir]) or wandb.run is None:
        print(
            f"[LOG_RENAMER] Skipping rename: "
            f"SLURM_JOB_ID={slurm_job_id!r} SLURM_JOB_NAME={slurm_job_name!r} "
            f"SLURM_LOG_DIR={log_dir!r} wandb.run={wandb.run}",
            flush=True,
        )
        return

    original = Path(log_dir) / f"{slurm_job_name}_{slurm_job_id}.out"
    renamed = Path(log_dir) / f"{slurm_job_name}_{slurm_job_id}_{wandb.run.name}.out"

    try:
        if original.exists():
            original.rename(renamed)
            print(f"[LOG_RENAMER] ✓ {original.name} → {renamed.name}", flush=True)
        else:
            print(f"[LOG_RENAMER] ⚠ Log file not found (rename skipped): {original}", flush=True)
    except Exception as e:
        print(f"[LOG_RENAMER] ✗ Error renaming log file: {e}", flush=True)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
