#!/bin/bash
# =============================================================================
# Unified SLURM Sweep Launcher
# =============================================================================
# Usage:
#   1. Create a sweep:    wandb sweep configs/alm_sweep.yaml
#   2. Set SWEEP_ID below (the entity/project/id string wandb prints)
#   3. Submit:            sbatch launch_sweep.sh
#
# The script auto-detects GPUs and launches one wandb agent per GPU.
# Each agent runs COUNT runs (default 1) then exits.
#
# To change the sweep, just update SWEEP_ID — no other edits needed.
# =============================================================================

# ---- Configurable variables ------------------------------------------------
SWEEP_ID="aisco/AISCO_code-subprojects_wp4_aeroshell_optimizer_v1_src_main_files/xa27etq2"   # <-- EDIT THIS
COUNT=3                                         # runs per agent

# ---- SLURM directives ------------------------------------------------------
# NOTE: keep --ntasks and --gpus in sync with each other (1 task = 1 GPU)
#SBATCH --job-name=WP4_Sweep
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=5G
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ---- Environment setup -----------------------------------------------------
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME="$(whoami)"
LOG_DIR="/cluster/scratch/${USERNAME}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

VENV_DIR="/cluster/home/${USERNAME}/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

# ---- Job info ---------------------------------------------------------------
echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS"
echo "Sweep: $SWEEP_ID"
echo "Count per agent: $COUNT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "=========================================="

# ---- Launch agents ----------------------------------------------------------
cd /cluster/home/${USERNAME}/AISCO_code
export PYTHONPATH="/cluster/home/${USERNAME}/AISCO_code:$PYTHONPATH"

echo "[$(date)] Launching $SLURM_NTASKS wandb agents (1 per task, $COUNT runs each)..."

for i in $(seq 0 $((SLURM_NTASKS - 1))); do
    echo "[$(date)]   Task $i → agent starting"
    srun --exclusive --ntasks=1 \
        bash -c "
            echo '[$(date)] Agent $i on $(hostname)' >> ${LOG_DIR}/agent_${SLURM_JOB_ID}_${i}.log
            wandb agent --count $COUNT $SWEEP_ID >> ${LOG_DIR}/agent_${SLURM_JOB_ID}_${i}.log 2>&1
            echo '[$(date)] Agent $i finished' >> ${LOG_DIR}/agent_${SLURM_JOB_ID}_${i}.log
        " &
done
wait

echo "=========================================="
echo "[$(date)] All agents finished."
echo "Total elapsed: $SECONDS seconds"
echo "=========================================="
