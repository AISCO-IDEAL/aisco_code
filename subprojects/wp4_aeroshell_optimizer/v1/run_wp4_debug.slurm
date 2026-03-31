#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=wp4_debug
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/%u/logs/wp4_debug_%j.out
#SBATCH --error=/cluster/scratch/%u/logs/wp4_debug_%j.err

echo "=== Job started on $(hostname) at $(date) ==="

# --- Modules ---
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.1.1
module load eth_proxy

# --- Venv ---
source "$HOME/AISCO_code/aisco_env/bin/activate"

# --- Environment ---
export WANDB_API_KEY="1a4ee339e3efa412724849b14253d8b04f49ac3a"
export WANDB_PROJECT="AISCO-WP4"
export WANDB_ENTITY="aisco"

# --- Run WP4 optimization ---
cd "$HOME/AISCO_code"
srun python -u subprojects/wp4_aeroshell_optimizer/v1/src/main_files/main_v1_1_one_model.py

echo "=== Job finished at $(date) ==="
