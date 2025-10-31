#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=test_diff
#SBATCH --time=00:20:00

# Use your original, correct directives
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=7G
#SBATCH --gpus=rtx_4090:1

# --- Use %x for job name and %j for job ID ---
#SBATCH --output=/cluster/home/elihuber/AISCO-code/subprojects/wp1_cfd_ai/data/logs/%x_%j.out
#SBATCH --error=/cluster/home/elihuber/AISCO-code/subprojects/wp1_cfd_ai/data/logs/%x_%j.err

##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=elihuber@ethz.ch

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
# Activate Python virtual environment
#------------------------------------------------------------------------------
VENV_DIR="$HOME/venvs/aisco-code"
source "$VENV_DIR/bin/activate"

#------------------------------------------------------------------------------
# Set WANDB environment variables
#------------------------------------------------------------------------------
export WANDB_API_KEY="1a4ee339e3efa412724849b14253d8b04f49ac3a"
export WANDB_PROJECT="CFD-AI"
export WANDB_ENTITY="aisco"

#------------------------------------------------------------------------------
# Set Hugging Face environment variables
#------------------------------------------------------------------------------
export HF_HOME="$SCRATCH/hf_models"
export HF_DATASETS_CACHE="$SCRATCH/hf_datasets"
export HF_TOKEN="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"

#------------------------------------------------------------------------------
# Run the Relevant script
#------------------------------------------------------------------------------

srun python "$HOME/AISCO-code/subprojects/wp1_cfd_ai/crude_estimator/test_diff.py"