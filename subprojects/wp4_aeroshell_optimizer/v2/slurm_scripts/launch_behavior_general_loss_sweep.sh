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

ts() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ---- Configure this after `wandb sweep ...` --------------------------------
SWEEP_ID="aisco/behavior-general-loss-schedule/uthde7da"  # full id: entity/project/sweep_id
COUNT=8

export no_proxy=$(echo "${no_proxy:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "${NO_PROXY:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')

if command -v module >/dev/null 2>&1; then
    module purge
    module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy
    ts "Checkpoint: environment modules loaded"
else
    ts "Checkpoint: 'module' command not found; running local mode"
fi

SOURCE_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "$(readlink -f -- "$SOURCE_PATH")")" && pwd)"

# Expected repo layout: <root>/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts
PROJECT_ROOT_CANDIDATE="$(cd -- "$SCRIPT_DIR/../../../../.." && pwd)"
if [[ -f "$PROJECT_ROOT_CANDIDATE/pyproject.toml" && -d "$PROJECT_ROOT_CANDIDATE/subprojects" ]]; then
    PROJECT_ROOT="$PROJECT_ROOT_CANDIDATE"
else
    PROJECT_ROOT="$PWD"
fi

if [[ -n "${AISCO_VENV_DIR:-}" ]]; then
    VENV_DIR="$AISCO_VENV_DIR"
else
    VENV_DIR="$PROJECT_ROOT/aisco_env"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    ts "ERROR: Python env not found: $VENV_DIR"
    ts "Hint: set AISCO_VENV_DIR or create env at $PROJECT_ROOT/aisco_env"
    exit 1
fi

source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"
ts "Checkpoint: virtualenv activated ($VENV_DIR)"
ts "Checkpoint: python executable = $PYTHON_BIN"
"$PYTHON_BIN" --version

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="$PROJECT_ROOT/subprojects:${PYTHONPATH:-}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH}"
cd "$PROJECT_ROOT"
ts "Checkpoint: changed directory to $(pwd)"

SWEEP_ID_CLEANED="$(echo "$SWEEP_ID" | xargs)"
if [[ -z "$SWEEP_ID_CLEANED" || "$SWEEP_ID_CLEANED" == "REPLACE_WITH_SWEEP_ID" ]]; then
    echo "ERROR: set SWEEP_ID to the id returned by 'wandb sweep ...'"
    exit 1
fi

echo "[$(date)] Starting single-agent sweep on GPU 0 for $SWEEP_ID_CLEANED (count=$COUNT)"
CUDA_VISIBLE_DEVICES=0 wandb agent --count "$COUNT" "$SWEEP_ID_CLEANED"
echo "[$(date)] Sweep agent finished"
