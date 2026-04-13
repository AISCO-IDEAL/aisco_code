#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=Main_test
#SBATCH --time=03:59:00                   # No less than 4 hours as you're at the same queue acc. to Elias
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G                  # Total 40G memory
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=%u@ethz.ch
## run this command: sbatch ~/AISCO_code/subprojects/wp4_aeroshell_optimizer/v1/slurm_scripts/main_testing.sh
#------------------------------------------------------------------------------
# Define username variable (must come after all #SBATCH directives)
#------------------------------------------------------------------------------
umask 0002 # ensure group write permissions for logs and models

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
#------------------------------------------------------------------------------
# Verify GPU access
#------------------------------------------------------------------------------
echo "Verifying GPU access..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
#------------------------------------------------------------------------------
# Run the Relevant script
#------------------------------------------------------------------------------
export PYTHONUNBUFFERED=TRUE
## export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Allow PyTorch to expand GPU memory allocation as needed
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:$PYTHONPATH"
cd /cluster/home/$USERNAME/AISCO_code/
#python -m src.main

python subprojects/wp4_aeroshell_optimizer/v1/src/main_files/main_v1_1_one_model.py \
    --save_model \
    --problem_id="AISCO_WP4_Test1"  2>&1
#------------------------------------------------------------------------------
# Print job end information
#------------------------------------------------------------------------------
echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="
