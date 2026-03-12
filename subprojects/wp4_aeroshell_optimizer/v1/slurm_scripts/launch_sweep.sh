#!/bin/bash
# =============================================================================
# Unified SLURM Sweep Launcher — edit SWEEP_ID and COUNT below, then sbatch
# =============================================================================

# ---- SLURM directives (must come before any executable code) ---------------
#SBATCH --partition=gpu
#SBATCH --job-name=WP4_Sweep
#SBATCH --time=04:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:3
#SBATCH --mem-per-cpu=5G
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
################################### RUN THIS CMD ##############################################
##  sbatch ~/AISCO_code/subprojects/wp4_aeroshell_optimizer/v1/slurm_scripts/launch_sweep.sh ##
###############################################################################################
# ---- Configurable variables ------------------------------------------------
SWEEP_ID="aisco/new_50_meshes/ms1tekky"   # <-- EDIT THIS
COUNT=16                                      # runs per agent (no empty space, otherwise it will recognise count as empty string)

# ---- Job info ---------------------------------------------------------------
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
echo "=========================================="

# ---- Verify GPU access ------------------------------------------------------
echo "Verifying GPU access..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ---- Environment setup -----------------------------------------------------
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME="$(whoami)"
LOG_DIR="/cluster/scratch/${USERNAME}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

VENV_DIR="/cluster/home/${USERNAME}/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/${USERNAME}/AISCO_code/subprojects:$PYTHONPATH"
export PYTHONPATH="/cluster/home/${USERNAME}/AISCO_code:$PYTHONPATH"
cd /cluster/home/${USERNAME}/AISCO_code

# ---- Launch agents ----------------------------------------------------------
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "[$(date)] Detected $NUM_GPUS GPUs — launching 1 agent each, $COUNT runs per agent..."

for ((i=0; i<NUM_GPUS; i++)); do
    AGENT_LOG="${LOG_DIR}/agent_gpu${i}_${SLURM_JOB_ID}.log"
    echo "[$(date)]   GPU $i → agent starting. Log: $AGENT_LOG"
    CUDA_VISIBLE_DEVICES=$i wandb agent --count "$COUNT" "$SWEEP_ID" > "$AGENT_LOG" 2>&1 &
    sleep 10  # stagger startup to avoid race on wandb init
done
wait

echo "=========================================="
echo "[$(date)] All agents finished."
echo "Total elapsed: $SECONDS seconds"
echo "=========================================="
