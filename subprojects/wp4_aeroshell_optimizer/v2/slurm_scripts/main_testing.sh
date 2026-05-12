#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=test_output_name #CVM2_Diversity_gradients_per_slot
#SBATCH --time=00:01:20                   # Extended for adaptive mesh extraction (L=4)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G                  # Total 40G memory
#SBATCH --gpus=pro_6000:1
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j_temp.out
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=%u@ethz.ch
## run this command: sbatch ~/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/main_testing.sh
#------------------------------------------------------------------------------
# Define username variable (must come after all #SBATCH directives)
#------------------------------------------------------------------------------
set -euo pipefail

umask 0002 # ensure group write permissions for logs and models

ts() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Allow W&B to route through the cluster proxy (eth_proxy sets no_proxy which blocks api.wandb.ai)
export no_proxy=$(echo "${no_proxy:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "${NO_PROXY:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6  eth_proxy


USERNAME=$(whoami)
LOG_DIR="/cluster/scratch/$USERNAME/logs"
##mkdir -p "$LOG_DIR"  # Ensure the log directory exists
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
# Load modules
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Activate the Python virtual environment
#------------------------------------------------------------------------------
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"
ts "Checkpoint: virtualenv activated ($VENV_DIR)"
ts "Checkpoint: python executable = $PYTHON_BIN"
"$PYTHON_BIN" --version
#------------------------------------------------------------------------------
# Verify GPU access
#------------------------------------------------------------------------------
echo "Verifying GPU access..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
ts "Checkpoint: GPU visibility check completed"
#------------------------------------------------------------------------------

# Set Hugging Face environment variables (not used for now)
#------------------------------------------------------------------------------
# export HF_HOME="$SCRATCH/hf_models"
# export HF_DATASETS_CACHE="$SCRATCH/hf_datasets"
# export HF_TOKEN="HuggingFace_TOKEN"  # Replace with your actual Huggingface token, create in Settings, Access Tokens, Read
#------------------------------------------------------------------------------
# Log GPU/CPU info to the same log directory
#------------------------------------------------------------------------------

##echo "GPU Info:" >> "$LOG_DIR/AISCO_TEST_${SLURM_JOB_ID}.out"
##nvidia-smi >> "$LOG_DIR/AISCO_TEST_${SLURM_JOB_ID}.out"
##echo "CPU Info:" >> "$LOG_DIR/AISCO_TEST_${SLURM_JOB_ID}.out"
##lscpu >> "$LOG_DIR/AISCO_TEST_${SLURM_JOB_ID}.out"
#------------------------------------------------------------------------------
# Run the Relevant script
#------------------------------------------------------------------------------
export no_proxy=$(echo "${no_proxy:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "${NO_PROXY:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export PYTHONUNBUFFERED=TRUE
## export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Allow PyTorch to expand GPU memory allocation as needed
# Redirect W&B local run data to scratch to avoid home-directory quota issues
export WANDB_DIR="/cluster/scratch/$USERNAME/wandb_runs"
export WANDB_CACHE_DIR="/cluster/scratch/$USERNAME/wandb_cache"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:${PYTHONPATH:-}"
# Pass log directory and job info to Python script for log renaming with W&B run name
export SLURM_LOG_DIR="$LOG_DIR"
cd /cluster/home/$USERNAME/AISCO_code/
ts "Checkpoint: changed directory to $(pwd)"


#python -m src.main

ts "Checkpoint: launching main_v2.py with hard timeout"
EXIT_CODE=0
if ! timeout --signal=TERM --kill-after=30s 13800s "$PYTHON_BIN" -u subprojects/wp4_aeroshell_optimizer/v2/src/main_files/main_v2.py 2>&1
then
    EXIT_CODE=$?
    if [ "$EXIT_CODE" -eq 124 ]; then
        ts "ERROR: main_v2.py hit timeout (13800s)"
    else
        ts "ERROR: main_v2.py exited non-zero (EXIT_CODE=$EXIT_CODE)"
    fi
else
    ts "Checkpoint: main_v2.py finished successfully"
fi
#------------------------------------------------------------------------------
# Print job end information
#------------------------------------------------------------------------------
echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="

# Exit with the original exit code
exit "$EXIT_CODE"
