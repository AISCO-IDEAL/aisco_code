#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=GAOT2_Training
#SBATCH --time=24:00:00                  # Adjust time as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G                 # Total 64G memory
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=%u@ethz.ch
## run this command: sbatch ~/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/slurm/train_gaot2.slurm

#------------------------------------------------------------------------------
# Setup environment
#------------------------------------------------------------------------------
umask 0002
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME=$(whoami)

#------------------------------------------------------------------------------
# Print job information
#------------------------------------------------------------------------------
echo "=========================================="
echo "GAOT2 Training Job"
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated memory per CPU: $SLURM_MEM_PER_CPU MB"
echo "Number of GPUs: $SLURM_GPUS"
echo "GPU type: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Working directory: $PWD"
echo "=========================================="

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
echo ""

#------------------------------------------------------------------------------
# Set environment variables
#------------------------------------------------------------------------------
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:$PYTHONPATH"
cd /cluster/home/$USERNAME/AISCO_code/

#------------------------------------------------------------------------------
# Run the training
# Options:
#   -c/--config: Path to single config file
#   -f/--folder: Path to folder with config files
#   --debug: Debug mode (disables multiprocessing)
#   --num_works_per_device: Number of workers per device (default: 10)
#   --visible_devices: List of visible GPU device IDs
#------------------------------------------------------------------------------
echo "Starting GAOT2 training..."
echo ""

python subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/main.py \
    --config subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/config/euler_field.yaml \
    --debug

#------------------------------------------------------------------------------
# Print job end information
#------------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="
