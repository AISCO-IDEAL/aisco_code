#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=GAOT_overfit
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:2
#SBATCH --output=/cluster/scratch/elihuber/scratch_logs/Overfit_%j.out
#SBATCH --error=/cluster/scratch/elihuber/scratch_logs/Overfit_%j.err

#------------------------------------------------------------------------------
# OVERFIT SANITY CHECK
#
# This job trains on only 10 samples for 500 epochs.
# If the model works, training loss should reach near-zero.
# If it doesn't, the architecture has a fundamental problem.
#
# Submit with:  sbatch ~/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/slurm_scripts/elias_overfit.sh
#------------------------------------------------------------------------------

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME="elihuber"

echo "=========================================="
echo "OVERFIT SANITY CHECK"
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

nvidia-smi

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
cd /cluster/home/$USERNAME/AISCO_code/

ulimit -n 65536

BASE_CONFIG="subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT_working/config/pressure_overfit.yaml"
ARCHIVE_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_drag_results/configs"
mkdir -p $ARCHIVE_DIR
TEMP_CONFIG="$ARCHIVE_DIR/overfit_run_${SLURM_JOB_ID}.yaml"

sed "s/_RUN_ID_/${SLURM_JOB_ID}/g" $BASE_CONFIG > $TEMP_CONFIG

torchrun --standalone --nproc_per_node=2 \
    subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT_working/main.py \
    -c $TEMP_CONFIG \
    2>&1

echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="
