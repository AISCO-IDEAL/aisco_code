#!/bin/bash
# Define username variable
USERNAME="eschaerer"

#SBATCH --job-name=AISCO_Test_SLURM
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=7G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/home/$USERNAME/logs/AISCO_TEST_%j.out
#SBATCH --error=/cluster/home/$USERNAME/logs/AISCO_TEST_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USERNAME@ethz.ch

# Create the folders on $SCRATCH if they don't exist
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/datasets"
mkdir -p "$SCRATCH/models"

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.4.1
module load eth_proxy

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
# Run the Relevant script
#------------------------------------------------------------------------------
python3 -m "$HOME/AISCO-code/subprojects/wp4_aeroshell_optimiser/src/main" \
    --save_model \
    --problem_id="AISCO_WP4_Test1"
