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
#   DEVICE=cuda:0 ATOL=1e-5 RTOL=1e-5 TRAIN_EPOCHS=5 KEEP_ARTIFACTS=1 sbatch subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/slurm_scripts/test_gaot2_gaot3_equivalence.sh

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
export PYTHONPATH="$REPO_DIR:$REPO_DIR/subprojects:${PYTHONPATH:-}"
export WANDB_DISABLED=true

cd "$REPO_DIR"

DEVICE="${DEVICE:-cpu}"
ATOL="${ATOL:-1e-6}"
RTOL="${RTOL:-1e-6}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
LR="${LR:-1e-3}"
SKIP_NORMAL_TRAINING="${SKIP_NORMAL_TRAINING:-0}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"
TEST_SCRIPT="subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/tests/test_gaot2_gaot3_equivalence.py"

if [ ! -f "$TEST_SCRIPT" ]; then
    echo "Test script not found: $TEST_SCRIPT"
    exit 1
fi

echo "Running GAOT2/GAOT3 equivalence check"
echo "Device: $DEVICE"
echo "Tolerance: atol=$ATOL rtol=$RTOL"
echo "Training check: epochs=$TRAIN_EPOCHS lr=$LR"
echo "Full main.py training check: skip=$SKIP_NORMAL_TRAINING keep_artifacts=$KEEP_ARTIFACTS"

EXTRA_ARGS=()
if [ "$SKIP_NORMAL_TRAINING" = "1" ]; then
    EXTRA_ARGS+=("--skip-normal-training")
fi
if [ "$KEEP_ARTIFACTS" = "1" ]; then
    EXTRA_ARGS+=("--keep-normal-training-artifacts")
fi

python "$TEST_SCRIPT" --device "$DEVICE" --atol "$ATOL" --rtol "$RTOL" --train-epochs "$TRAIN_EPOCHS" --lr "$LR" "${EXTRA_ARGS[@]}"

echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="
