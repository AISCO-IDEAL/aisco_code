#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=Ahmed_Infer
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/elihuber/scratch_logs/Ahmed_Infer_%j.out
#SBATCH --error=/cluster/scratch/elihuber/scratch_logs/Ahmed_Infer_%j.err

# ----------------------------------------------------------------
# GAOT Field Inference — Ahmed dataset
#
# Usage:
#   sbatch infer_ahmed_field.sh /path/to/checkpoint.pt
# ----------------------------------------------------------------

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME="elihuber"
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

WORK_DIR="/cluster/home/$USERNAME/AISCO_code"
GAOT_DIR="$WORK_DIR/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT_working"
CONFIG="$GAOT_DIR/config/ahmed_field.yaml"

CKPT="${1:-$(ls -t /cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_drag_results/ckpt/ahmed_field*.pt 2>/dev/null | head -1)}"
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: No checkpoint found. Pass path as argument."
    exit 1
fi

OUT_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_drag_results/inference/ahmed_field_${SLURM_JOB_ID}"

echo "=========================================="
echo "GAOT Field Inference — Ahmed"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "Config: $CONFIG"
echo "Checkpoint: $CKPT"
echo "Output: $OUT_DIR"
echo "=========================================="

nvidia-smi

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:$PYTHONPATH"

cd "$GAOT_DIR"

python scripts/inference_field.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --out "$OUT_DIR" \
    --split test \
    --device cuda:0 \
    --save-fields

echo "Finished: $(date) | Results: $OUT_DIR"
