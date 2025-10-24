#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=00:10:00        # 10 minutes is plenty
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=7G
#SBATCH --gpus=rtx_4090:2      # <-- We are requesting one GPU
#SBATCH --output=/cluster/home/elihuber/logs/compute_test_%j.out

echo "Job started on $(hostname) at $(date)"

# 1. Load the CORRECT modules
module purge
# Use the exact CUDA version the python module asked for:
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cuda/12.1.1

# !! Make sure this venv path is correct for you !!
source ~/venvs/aisco-code/bin/activate

# 2. Run your Python test script
python test_gpu.py

echo "Job finished at $(date)"
