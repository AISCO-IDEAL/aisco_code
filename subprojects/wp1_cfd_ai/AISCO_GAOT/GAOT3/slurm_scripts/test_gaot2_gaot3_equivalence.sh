#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=GAOT3_EQUIV
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/home/%u/logs/GAOT3_EQUIV_%j.out
#SBATCH --error=/cluster/home/%u/logs/GAOT3_EQUIV_%j.err

# RUN COMMAND:
#   sbatch subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/slurm_scripts/test_gaot2_gaot3_equivalence.sh
#
# Optional:
#   TIMEOUT_SECONDS=7200 sbatch subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/slurm_scripts/test_gaot2_gaot3_equivalence.sh

set -euo pipefail

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME="$USER"
REPO_DIR="/cluster/home/$USERNAME/AISCO_code"
VENV_DIR="$REPO_DIR/aisco_env"

source "$VENV_DIR/bin/activate"

echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Running on node: ${SLURMD_NODENAME:-local}"
echo "Number of CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Working directory before cd: $PWD"
echo "=========================================="

export PYTHONUNBUFFERED=TRUE
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$REPO_DIR:$REPO_DIR/subprojects:${PYTHONPATH:-}"
export WANDB_DISABLED=true

cd "$REPO_DIR"

find subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2 subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3 \
    -type d -name "__pycache__" -prune -exec rm -rf {} +

TEST_SCRIPT="subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/tests/test_gaot2_gaot3_equivalence.py"
GAOT2_CONFIG="${GAOT2_CONFIG:-subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/config/main_config.yaml}"
GAOT3_CONFIG="${GAOT3_CONFIG:-subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/config/main_config.yaml}"
LOG_DIR="${LOG_DIR:-/cluster/home/$USERNAME/logs/gaot2_gaot3_equivalence_${SLURM_JOB_ID:-local}}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-}"

if [ ! -f "$TEST_SCRIPT" ]; then
    echo "Test script not found: $TEST_SCRIPT"
    exit 1
fi

echo "Running GAOT2/GAOT3 main/config execution check"
echo "GAOT2 config: $GAOT2_CONFIG"
echo "GAOT3 config: $GAOT3_CONFIG"
echo "Log directory: $LOG_DIR"
echo "Timeout per run: ${TIMEOUT_SECONDS:-none}"

EXTRA_ARGS=()
if [ -n "$TIMEOUT_SECONDS" ]; then
    EXTRA_ARGS+=("--timeout" "$TIMEOUT_SECONDS")
fi

python "$TEST_SCRIPT" \
    --gaot2-config "$GAOT2_CONFIG" \
    --gaot3-config "$GAOT3_CONFIG" \
    --log-dir "$LOG_DIR" \
    "${EXTRA_ARGS[@]}"

echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="
