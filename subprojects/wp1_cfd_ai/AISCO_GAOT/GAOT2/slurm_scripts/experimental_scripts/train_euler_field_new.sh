#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=Euler_New
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:2
#SBATCH --output=/cluster/scratch/elihuber/scratch_logs/Euler_New_%j.out
#SBATCH --error=/cluster/scratch/elihuber/scratch_logs/Euler_New_%j.err

# Train GAOT with pressure field decoder on Euler NEW dataset (58 samples, ~2.3K verts)
# Small meshes — uses full_grid strategy (no subsampling needed)

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME="elihuber"
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

echo "=========================================="
echo "GAOT Field Training — Euler NEW Dataset"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS"
echo "=========================================="

nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

cd /cluster/home/$USERNAME/AISCO_code/
ulimit -n 65536

BASE_CONFIG="subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT_working/config/euler_field_new.yaml"
ARCHIVE_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_drag_results/configs"
mkdir -p $ARCHIVE_DIR

TEMP_CONFIG="$ARCHIVE_DIR/euler_field_new_${SLURM_JOB_ID}.yaml"
sed "s/_RUN_ID_/${SLURM_JOB_ID}/g" $BASE_CONFIG > $TEMP_CONFIG

torchrun --standalone --nproc_per_node=2 \
    subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT_working/main.py \
    -c $TEMP_CONFIG \
    2>&1

echo "=========================================="
echo "Finished: $(date)"
echo "Total elapsed: $SECONDS seconds"
echo "=========================================="

## sbatch ~/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/slurm_scripts/train_euler_field_new.sh
