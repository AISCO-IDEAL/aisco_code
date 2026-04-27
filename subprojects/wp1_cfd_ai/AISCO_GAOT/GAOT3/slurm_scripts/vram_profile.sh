#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=VRAM_Profile
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/%u/scratch_logs/VRAM_Profile_%j.out
#SBATCH --error=/cluster/scratch/%u/scratch_logs/VRAM_Profile_%j.err

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME="$USER"
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

echo "=========================================="
echo "VRAM Profiling Job"
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=========================================="

nvidia-smi

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /cluster/home/$USERNAME/AISCO_code/

# Use the newer config (CP-full dataset, batch_size=6)
CONFIG="subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/config/newer_main_config.yaml"

# Create a temp config with placeholders filled
TEMP_CONFIG="/tmp/vram_profile_${SLURM_JOB_ID}_config.yaml"
sed "s|_NAME_|vram_profile|g; s|_RUN_ID_|${SLURM_JOB_ID}|g" $CONFIG > $TEMP_CONFIG

python subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/vram_profile.py \
    -c $TEMP_CONFIG \
    2>&1

echo "=========================================="
echo "Job ended on: $(date)"
echo "=========================================="

## sbatch subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/slurm_scripts/vram_profile.sh
