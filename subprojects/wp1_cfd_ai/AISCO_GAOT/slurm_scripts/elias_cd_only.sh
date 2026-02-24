#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=GAOT_cd
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:2
#SBATCH --output=/cluster/scratch/elihuber/scratch_logs/CD_%j.out
#SBATCH --error=/cluster/scratch/elihuber/scratch_logs/CD_%j.err

#------------------------------------------------------------------------------
# Define username variable
#------------------------------------------------------------------------------

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6  eth_proxy

USERNAME="elihuber"
LOG_DIR="/cluster/scratch/elihuber/scratch_logs"

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
# Run the training
#------------------------------------------------------------------------------
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
cd /cluster/home/$USERNAME/AISCO_code/

ulimit -n 65536

# Define your base config
BASE_CONFIG="subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT_working/config/pressure_cd_only.yaml"

# Archive folder for run configs
ARCHIVE_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_drag_results/configs"
mkdir -p $ARCHIVE_DIR

# Set the path for the temporary unique config
TEMP_CONFIG="$ARCHIVE_DIR/cd_run_${SLURM_JOB_ID}.yaml"

# 1. Inject the Slurm Job ID into the YAML and save it to the archive folder
sed "s/_RUN_ID_/${SLURM_JOB_ID}/g" $BASE_CONFIG > $TEMP_CONFIG

# 2. Run training with the cd-only config (multi-GPU via torchrun)
torchrun --standalone --nproc_per_node=2 \
    subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT_working/main.py \
    -c $TEMP_CONFIG \
    2>&1

#------------------------------------------------------------------------------
# Print job end information
#------------------------------------------------------------------------------
echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="

## chmod +x ~/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/slurm_scripts/elias_cd_only.sh
## sbatch ~/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/slurm_scripts/elias_cd_only.sh
## watch -n 5 squeue -u elihuber
