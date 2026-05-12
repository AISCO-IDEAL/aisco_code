"""Utility to rename SLURM log files to include W&B run name immediately after init."""

import os
import shutil
import sys
from pathlib import Path

import wandb


def rename_log_file_with_wandb_run_name() -> None:
    """
    Rename the SLURM log file immediately after wandb.init() to include the W&B run name.

    This function should be called right after wandb.init() to capture and rename the
    log file EARLY, so the final filename includes the W&B run name from the start.
    This ensures the correct name even if the job times out or crashes later.

    Expects environment variables:
        - SLURM_JOB_ID: Job ID from SLURM
        - SLURM_JOB_NAME: Job name from SLURM
        - SLURM_LOG_DIR: Directory where SLURM logs are stored (passed by shell script)
        - SLURM_OUTPUT_TEMP: Temporary output file path (optional, constructed if not set)

    Should be called after wandb.init().
    """
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    slurm_job_name = os.environ.get("SLURM_JOB_NAME", "")
    log_dir = os.environ.get("SLURM_LOG_DIR", "")

    print("[LOG_RENAMER] rename_log_file_with_wandb_run_name() called", flush=True)
    print(f"[LOG_RENAMER]   SLURM_JOB_ID={slurm_job_id}", flush=True)
    print(f"[LOG_RENAMER]   SLURM_JOB_NAME={slurm_job_name}", flush=True)
    print(f"[LOG_RENAMER]   SLURM_LOG_DIR={log_dir}", flush=True)
    print(f"[LOG_RENAMER]   wandb.run={wandb.run}", flush=True)
    sys.stdout.flush()

    if not all([slurm_job_id, slurm_job_name, log_dir]) or wandb.run is None:
        print("[LOG_RENAMER] Skipping: missing required env vars or wandb.run not available", flush=True)
        sys.stdout.flush()
        return

    # Construct paths
    temp_log_path = Path(log_dir) / f"{slurm_job_name}_{slurm_job_id}_temp.out"
    final_log_path = Path(log_dir) / f"{slurm_job_name}_{slurm_job_id}_{wandb.run.name}.out"

    print(f"[LOG_RENAMER]   Temp log: {temp_log_path}", flush=True)
    print(f"[LOG_RENAMER]   Final log: {final_log_path}", flush=True)
    sys.stdout.flush()

    try:
        if temp_log_path.exists():
            # Rename temp file to final name with W&B run name
            shutil.move(str(temp_log_path), str(final_log_path))
            print("[LOG_RENAMER] ✓ Successfully renamed log file to include W&B run name", flush=True)
            print(f"[LOG_RENAMER]   {temp_log_path.name} → {final_log_path.name}", flush=True)

            # Update stdout/stderr to write to the new file
            # This ensures all subsequent output goes to the renamed file
            try:
                log_fd = os.open(str(final_log_path), os.O_WRONLY | os.O_APPEND)
                os.dup2(log_fd, 1)  # Redirect stdout
                os.dup2(log_fd, 2)  # Redirect stderr
                os.close(log_fd)
                print("[LOG_RENAMER] ✓ Redirected stdout/stderr to renamed file", flush=True)
            except Exception as redirect_err:
                print(f"[LOG_RENAMER] ⚠ Could not redirect stdout/stderr: {redirect_err}", flush=True)

        else:
            print(f"[LOG_RENAMER] ⚠ Temp log file not found: {temp_log_path}", flush=True)

    except Exception as e:
        print(f"[LOG_RENAMER] ✗ Error renaming log file: {e}", flush=True)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
