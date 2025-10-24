#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=AISCO_elihuber_Test1
#SBATCH --time=00:20:00                    # Adjust time as needed

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=7G                  # Total 28G memory
#SBATCH --gpus=rtx_4090:2

# --- Use %x for job name and %j for job ID ---
#SBATCH --output=/cluster/home/elihuber/AISCO-code/subprojects/WP1-CFD/euler_tests/logs/%x_%j.out
#SBATCH --error=/cluster/home/elihuber/AISCO-code/subprojects/WP1-CFD/euler_tests/logs/%x_%j.err

##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=elihuber@ethz.ch

# Create the folders on $SCRATCH if it doesn't exist
mkdir -p "$HOME/logs"
mkdir -p "$HOME/datasets"
mkdir -p "$HOME/models"

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.1.1
module load eth_proxy

#------------------------------------------------------------------------------
# Activate the Python virtual environment (engibench)
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venvs/aisco-code"
source "$VENV_DIR/bin/activate"

#------------------------------------------------------------------------------
# Set WANDB environment variables
#------------------------------------------------------------------------------
export WANDB_API_KEY="1a4ee339e3efa412724849b14253d8b04f49ac3a"  # connect to wandb, then go to wandb.ai/settings, scroll down to API key and reveal
export WANDB_PROJECT="CFD-AI"
export WANDB_ENTITY="aisco"

#------------------------------------------------------------------------------
# Set Hugging Face environment variables
#------------------------------------------------------------------------------
export HF_HOME="$SCRATCH/hf_models"
export HF_DATASETS_CACHE="$SCRATCH/hf_datasets"
export HF_TOKEN="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"  # Replace with your actual Huggingface token, create in Settings, Access Tokens, Read

#------------------------------------------------------------------------------
# Run the Relevant script
#------------------------------------------------------------------------------
python "/cluster/home/elihuber/cuda.py"
