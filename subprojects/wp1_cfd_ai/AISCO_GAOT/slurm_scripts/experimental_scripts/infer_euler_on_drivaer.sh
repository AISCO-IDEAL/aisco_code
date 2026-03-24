#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=Euler_on_DrivAer
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/elihuber/scratch_logs/Euler_on_DrivAer_%j.out
#SBATCH --error=/cluster/scratch/elihuber/scratch_logs/Euler_on_DrivAer_%j.err

# ----------------------------------------------------------------
# Cross-dataset inference: Euler-trained GAOT → DrivAer data
#
# Usage:
#   sbatch infer_euler_on_drivaer.sh /path/to/euler_checkpoint.pt
# ----------------------------------------------------------------

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME="elihuber"
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

WORK_DIR="/cluster/home/$USERNAME/AISCO_code"
GAOT_DIR="$WORK_DIR/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT_working"

# Model config = Euler (architecture + normalization stats)
MODEL_CONFIG="$GAOT_DIR/config/euler_field.yaml"
# Data config = DrivAer (where to load test samples from)
DATA_CONFIG="$GAOT_DIR/config/drivaer_field.yaml"

CKPT="${1:-$(ls -t /cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_drag_results/ckpt/euler_field*.pt 2>/dev/null | head -1)}"
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: No Euler checkpoint found. Pass path as argument."
    exit 1
fi

OUT_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_drag_results/inference/euler_on_drivaer_${SLURM_JOB_ID}"

echo "=========================================="
echo "Cross-Dataset Inference: Euler → DrivAer"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "Model config: $MODEL_CONFIG"
echo "Data config:  $DATA_CONFIG"
echo "Checkpoint:   $CKPT"
echo "Output:       $OUT_DIR"
echo "=========================================="

nvidia-smi

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:$PYTHONPATH"

cd "$GAOT_DIR"

python scripts/inference_field.py \
    --config "$MODEL_CONFIG" \
    --data-config "$DATA_CONFIG" \
    --ckpt "$CKPT" \
    --out "$OUT_DIR" \
    --split test \
    --device cuda:0 \
    --save-fields

echo "Finished: $(date) | Results: $OUT_DIR"
