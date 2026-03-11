#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=50_Meshes_CFD_AI
#SBATCH --time=12:00:00                   # Adjust time as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G                  # Total 40G memory
#SBATCH --gpus=rtx_4090:3
#SBATCH --output=/cluster/home/%u/AISCO_code/subprojects/wp4_aeroshell_optimizer/logs/50_Meshes_CFD_AI_%j.out
## run this command: sbatch ~/AISCO_code/subprojects/wp4_aeroshell_optimizer/v1/slurm_scripts/main_sweeping.sh
#------------------------------------------------------------------------------
# Print job information
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
echo "=========================================="

#------------------------------------------------------------------------------
# Verify GPU access
#------------------------------------------------------------------------------
echo "Verifying GPU access..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ------------------------------------------------------------------------------
# Load Modules
# ------------------------------------------------------------------------------
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6  eth_proxy

#------------------------------------------------------------------------------
# Define username variable (must come after all #SBATCH directives)
#------------------------------------------------------------------------------
USERNAME=$(whoami)
LOG_DIR="/cluster/home/$USERNAME/AISCO_code/logs"
# Ensure this directory exists
mkdir -p "/cluster/home/$USERNAME/AISCO_code/subprojects/wp4_aeroshell_optimizer/logs/agents"

VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:$PYTHONPATH"
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code:$PYTHONPATH"
cd /cluster/home/$USERNAME/AISCO_code/

SWEEP_ID="aisco/50_Meshes_CFD_AI/3bu22zld"
AGENTS_PER_GPU=1  # REDUCED to 1 for debugging. Run 2 only if stable.

# Get the list of GPU IDs assigned by Slurm (e.g. 0,1,2 or 0)
# We assume standard 0-indexed local IDs for the job
NUM_GPUS_ALLOCATED=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Detected $NUM_GPUS_ALLOCATED GPUs allocated."

for ((i=0; i<NUM_GPUS_ALLOCATED; i++)); do
    for ((j=1; j<=AGENTS_PER_GPU; j++)); do
        AGENT_LOG="/cluster/home/$USERNAME/AISCO_code/subprojects/wp4_aeroshell_optimizer/logs/agents/agent_gpu${i}_id${j}_${SLURM_JOB_ID}.log"

        echo "[$(date)] Launching wandb agent #$j on logical GPU $i. Log: $AGENT_LOG"

        # Launch in background, explicitly setting device, and redirecting OUTPUT
        CUDA_VISIBLE_DEVICES=$i wandb agent --count 50 "$SWEEP_ID" > "$AGENT_LOG" 2>&1 &

        sleep 10
    done
done

wait
echo "[$(date)] All wandb agents finished."

#------------------------------------------------------------------------------
# Print job end information
#------------------------------------------------------------------------------
echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="
