#!/bin/bash
# =============================================================================
# Launch script for behavior_general_loss_schedule_sweep.yaml
# Single GPU, hard SLURM wall time 6 hours.
# =============================================================================

#SBATCH --partition=gpu
#SBATCH --job-name=wp4_behav_glsweep
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=5G
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out
#SBATCH --mail-user=%u@ethz.ch

set -euo pipefail
umask 0002

# ---- Configure this after `wandb sweep ...` --------------------------------
SWEEP_ID="aisco/behavior-general-loss-schedule/uthde7da"  # full id: entity/project/sweep_id
COUNT=8

export no_proxy=$(echo "${no_proxy:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "${NO_PROXY:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME="$(whoami)"
VENV_DIR="/cluster/home/${USERNAME}/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/${USERNAME}/AISCO_code/subprojects:${PYTHONPATH:-}"
export PYTHONPATH="/cluster/home/${USERNAME}/AISCO_code:${PYTHONPATH}"
cd /cluster/home/${USERNAME}/AISCO_code

if [[ "$SWEEP_ID" == "aisco/behavior-general-loss-schedule/uthde7da" || -z "$SWEEP_ID" ]]; then
    echo "ERROR: set SWEEP_ID to the id returned by 'wandb sweep ...'"
    exit 1
fi

echo "[$(date)] Starting single-agent sweep on GPU 0 for $SWEEP_ID (count=$COUNT)"
CUDA_VISIBLE_DEVICES=0 wandb agent --count "$COUNT" "$SWEEP_ID"
echo "[$(date)] Sweep agent finished"
