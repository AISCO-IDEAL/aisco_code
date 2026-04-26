#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=Main_v2_local
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out
## run this command:
## bash ~/Documents/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/beast_testing.sh

set -euo pipefail

umask 0002

ts() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

export no_proxy=$(echo "${no_proxy:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "${NO_PROXY:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../../" && pwd)"

if command -v module >/dev/null 2>&1; then
    module purge
    module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy
    ts "Checkpoint: environment modules loaded"
else
    ts "Checkpoint: 'module' command not found; running local mode"
fi

echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Running on node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Number of CPUs: ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Allocated memory per CPU: ${SLURM_MEM_PER_CPU:-N/A} MB"
echo "Number of GPUs: ${SLURM_GPUS:-$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null || echo N/A)}"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU type: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "Allocated memory per GPU: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
else
    echo "GPU type: nvidia-smi not found"
    echo "Allocated memory per GPU: N/A"
fi
echo "Working directory: $PWD"
echo "=========================================="

# Activate python environment (deterministic selection)
if [ -n "${AISCO_VENV_DIR:-}" ]; then
    VENV_DIR="$AISCO_VENV_DIR"
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        ts "ERROR: AISCO_VENV_DIR is set but invalid: $VENV_DIR"
        exit 1
    fi
    ts "Checkpoint: using explicitly requested venv from AISCO_VENV_DIR"
else
    VENV_DIR="$PROJECT_ROOT/aisco_env"
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        ts "ERROR: required local env not found: $VENV_DIR"
        ts "Hint: run ./setup_beast.sh or export AISCO_VENV_DIR to another valid venv"
        exit 1
    fi
    ts "Checkpoint: using default local env: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"
ts "Checkpoint: virtualenv activated ($VENV_DIR)"
ts "Checkpoint: python executable = $PYTHON_BIN"
"$PYTHON_BIN" --version

echo "Verifying GPU access..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not-set>}"
ts "Checkpoint: GPU visibility check completed"

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="$PROJECT_ROOT/subprojects:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"
ts "Checkpoint: changed directory to $(pwd)"

PRECHECK_TIMEOUT_SECONDS="${PRECHECK_TIMEOUT_SECONDS:-60}"
MAIN_TIMEOUT_SECONDS="${MAIN_TIMEOUT_SECONDS:-13800}"

ts "Checkpoint: starting import preflight (${PRECHECK_TIMEOUT_SECONDS}s timeout)"
if timeout "${PRECHECK_TIMEOUT_SECONDS}s" "$PYTHON_BIN" -u - <<'PY'
print('preflight: interpreter ok', flush=True)
import torch
print(f'preflight: torch {torch.__version__}', flush=True)
import subprojects.wp4_aeroshell_optimizer.v2.src.main_files.main_v2 as _m
print('preflight: main_v2 import ok', flush=True)
PY
then
    ts "Checkpoint: import preflight passed"
else
    rc=$?
    ts "ERROR: import preflight failed or timed out (rc=$rc)"
    exit "$rc"
fi

ts "Checkpoint: launching main_v2.py with hard timeout"
if timeout --signal=TERM --kill-after=30s "${MAIN_TIMEOUT_SECONDS}s" "$PYTHON_BIN" -u subprojects/wp4_aeroshell_optimizer/v2/src/main_files/main_v2.py \
    --save_model \
    --problem_id="AISCO_WP4_Test1" 2>&1
then
    ts "Checkpoint: main_v2.py finished successfully"
else
    rc=$?
    if [ "$rc" -eq 124 ]; then
        ts "ERROR: main_v2.py hit timeout (${MAIN_TIMEOUT_SECONDS}s)"
    else
        ts "ERROR: main_v2.py exited non-zero (rc=$rc)"
    fi
    exit "$rc"
fi

echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="
