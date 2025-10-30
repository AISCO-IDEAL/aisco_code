#!/bin/bash
# Define username variable
USERNAME="eschaerer"

#SBATCH --partition=gpu
#SBATCH --job-name=AISCO_Test_SLURM
#SBATCH --time=00:10:00                   # Adjust time as needed

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=7G                  # Total 28G memory
#SBATCH --gpus=1
#SBATCH --constraint=gpu:nvidia_geforce_rtx_4090

#SBATCH --output=/cluster/home/eschaerer/logs/AISCO_TEST_%j.out     #adjust path as needed
#SBATCH --error=/cluster/home/eschaerer/logs/AISCO_TEST_%j.err      #adjust path as needed

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eschaerer@ethz.ch

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
# Create the folders on $SCRATCH if they don't exist
#------------------------------------------------------------------------------
mkdir -p "$SCRATCH/logs" "$SCRATCH/datasets" "$SCRATCH/models"

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.1.1 eth_proxy

#------------------------------------------------------------------------------
# Activate the Python virtual environment
#------------------------------------------------------------------------------
VENV_DIR="/cluster/home/$USERNAME/AISCO-code/aisco_env"
source "$VENV_DIR/bin/activate"

#------------------------------------------------------------------------------
# Set WANDB environment variables (not used for now)
#------------------------------------------------------------------------------
# export WANDB_API_KEY="API_KEY_HERE"  # connect to wandb, then go to wandb.ai/settings, scroll down to API key and reveal
# export WANDB_PROJECT="test"
# export WANDB_ENTITY="aisco"

#------------------------------------------------------------------------------
# Set Hugging Face environment variables (not used for now)
#------------------------------------------------------------------------------
# export HF_HOME="$SCRATCH/hf_models"
# export HF_DATASETS_CACHE="$SCRATCH/hf_datasets"
# export HF_TOKEN="HuggingFace_TOKEN"  # Replace with your actual Huggingface token, create in Settings, Access Tokens, Read

#------------------------------------------------------------------------------
# Log GPU/CPU info
#------------------------------------------------------------------------------
echo "GPU Info:" >> "$SCRATCH/logs/AISCO_TEST_${SLURM_JOB_ID}.out"
nvidia-smi >> "$SCRATCH/logs/AISCO_TEST_${SLURM_JOB_ID}.out"
echo "CPU Info:" >> "$SCRATCH/logs/AISCO_TEST_${SLURM_JOB_ID}.out"
lscpu >> "$SCRATCH/logs/AISCO_TEST_${SLURM_JOB_ID}.out"

#------------------------------------------------------------------------------
# Run the Relevant script
#------------------------------------------------------------------------------
python3 "$HOME/AISCO-code/subprojects/wp4_aeroshell_optimizer/src/main.py" \
    --save_model \
    --problem_id="AISCO_WP4_Test1"

#------------------------------------------------------------------------------
# Print job end information
#------------------------------------------------------------------------------
echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="