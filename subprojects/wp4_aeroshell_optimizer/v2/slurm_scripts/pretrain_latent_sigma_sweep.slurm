#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=latent_sigma_sweep
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out
## Submitted by launch_sigma_sweep.sh — do not run directly.
## Sigma values are passed via --export:
##   TIGHT_SIGMA  (required)  near_surface_tight_sigma factor × bbox_extent_mm
##   WIDE_SIGMA   (optional)  near_surface_wide_sigma  — omit or empty for single-band
##
## Example manual submit:
##   sbatch --export=ALL,TIGHT_SIGMA=0.001,WIDE_SIGMA=0.01 pretrain_latent_sigma_sweep.slurm
#------------------------------------------------------------------------------
set -euo pipefail

umask 0002

ts() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Allow W&B to route through the cluster proxy
export no_proxy=$(echo "${no_proxy:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "${NO_PROXY:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME=$(whoami)
#------------------------------------------------------------------------------
echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated memory per CPU: $SLURM_MEM_PER_CPU MB"
echo "Number of GPUs: $SLURM_GPUS"
echo "GPU type: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Allocated memory per GPU: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
echo "Working directory: $PWD"
echo "TIGHT_SIGMA: ${TIGHT_SIGMA:-<not set>}"
echo "WIDE_SIGMA:  ${WIDE_SIGMA:-<not set — single-band mode>}"
echo "=========================================="
#------------------------------------------------------------------------------
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"
ts "Checkpoint: virtualenv activated ($VENV_DIR)"
ts "Checkpoint: python executable = $PYTHON_BIN"
"$PYTHON_BIN" --version
#------------------------------------------------------------------------------
echo "Verifying GPU access..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
ts "Checkpoint: GPU visibility check completed"
#------------------------------------------------------------------------------
export no_proxy=$(echo "${no_proxy:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "${NO_PROXY:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:${PYTHONPATH:-}"
cd /cluster/home/$USERNAME/AISCO_code/
ts "Checkpoint: changed directory to $(pwd)"

# Build sigma flags from env vars
SIGMA_FLAGS=""
if [ -n "${TIGHT_SIGMA:-}" ]; then
    SIGMA_FLAGS="$SIGMA_FLAGS --tight-sigma $TIGHT_SIGMA"
fi
if [ -n "${WIDE_SIGMA:-}" ]; then
    SIGMA_FLAGS="$SIGMA_FLAGS --wide-sigma $WIDE_SIGMA"
fi

ts "Checkpoint: launching pretraining_v5_latent.py (tight=${TIGHT_SIGMA:-config}, wide=${WIDE_SIGMA:-single-band})"
# shellcheck disable=SC2086  # SIGMA_FLAGS is intentionally word-split
if timeout --signal=TERM --kill-after=30s 13800s "$PYTHON_BIN" -u \
    subprojects/wp4_aeroshell_optimizer/v2/src/main_files/pretraining_v5_latent.py \
    --config pretraining_latent_sweep_v1_config.yaml \
    --vis-sampling \
    --make-gif \
    --wandb \
    $SIGMA_FLAGS 2>&1
then
    ts "Checkpoint: pretraining_v5_latent.py finished successfully"
else
    rc=$?
    if [ "$rc" -eq 124 ]; then
        ts "ERROR: pretraining_v5_latent.py hit timeout (13800s)"
    else
        ts "ERROR: pretraining_v5_latent.py exited non-zero (rc=$rc)"
    fi
    exit "$rc"
fi
#------------------------------------------------------------------------------
echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="
