#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=WP1_sweep
#SBATCH --time=24:00:00          # adjust as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/home/%u/logs/WP1_sweep_%j.out

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME=$(whoami)
LOG_DIR="/cluster/home/$USERNAME/logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Sweep job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated memory per CPU: $SLURM_MEM_PER_CPU MB"
echo "Number of GPUs: $SLURM_GPUS"
echo "GPU type: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Allocated memory per GPU: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
echo "Working directory: $PWD"
echo "=========================================="

# Activate venv
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

echo "Verifying GPU access..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:$PYTHONPATH"
export WANDB_CACHE_DIR="/cluster/scratch/$USERNAME/wandb_cache"

cd /cluster/home/$USERNAME/AISCO_code/
umask 0002
ulimit -n 65536

# Path to sweep config (relative to repo root)
SWEEP_CONFIG="subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/config/sweeping_config.yaml"

if [ ! -f "$SWEEP_CONFIG" ]; then
    echo "Sweep config not found: $SWEEP_CONFIG"
    exit 1
fi

# Strategy:
# 1) Create the sweep once (first job or manually).
# 2) Run this script with SWEEP_ID set in the environment to attach as an agent.
#
# Example:
#   cd /cluster/home/$USERNAME/AISCO_code
#   wandb sweep $SWEEP_CONFIG
#   # Note the generated SWEEP_ID (e.g. aisco/WP1_drag/abcd1234)
#   SWEEP_ID="aisco/WP1_drag/abcd1234" sbatch subprojects/wp1_cfd_ai/AISCO_GAOT/slurm_scripts/WP1_sweeping.sh

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID is not set."
    echo "Create a sweep first with:"
    echo "  cd /cluster/home/$USERNAME/AISCO_code"
    echo "  wandb sweep $SWEEP_CONFIG"
    echo "Then submit this job with:"
    echo "  SWEEP_ID=\"entity/project/xxxx\" sbatch subprojects/wp1_cfd_ai/AISCO_GAOT/slurm_scripts/WP1_sweeping.sh"
    exit 1
fi

echo "Starting W&B agent for sweep: $SWEEP_ID"
wandb agent "$SWEEP_ID"

echo "=========================================="
echo "Sweep job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="