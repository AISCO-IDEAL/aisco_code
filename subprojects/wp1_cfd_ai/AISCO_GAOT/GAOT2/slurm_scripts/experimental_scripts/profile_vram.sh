#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=VRAM_profile
#SBATCH --time=0:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/%u/scratch_logs/VRAM_profile_%j.out
#SBATCH --error=/cluster/scratch/%u/scratch_logs/VRAM_profile_%j.err

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME="$USER"
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

GAOT2_DIR="/cluster/home/$USERNAME/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2"
CONFIG="${1:-$GAOT2_DIR/config/experimental_configs/euler_field_cp_full_oom.yaml}"

echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "GPU: $SLURM_GPUS"
echo "Config: $CONFIG"
echo "=========================================="

cd "$GAOT2_DIR"
python scripts/profile_vram.py --config "$CONFIG"

echo "=========================================="
echo "Job ended on: $(date)"
echo "=========================================="
