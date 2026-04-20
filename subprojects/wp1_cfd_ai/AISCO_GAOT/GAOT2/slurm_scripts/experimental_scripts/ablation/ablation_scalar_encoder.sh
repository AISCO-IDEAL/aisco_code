#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=Ablation_Scalar
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:1
#SBATCH --array=0-4
#SBATCH --output=/cluster/scratch/%u/scratch_logs/Ablation_Scalar_%A_%a.out
#SBATCH --error=/cluster/scratch/%u/scratch_logs/Ablation_Scalar_%A_%a.err

# Ablation study: effect of the ScalarEncoder (scale_factors) on Cp field prediction.
#
# Five runs launched as a SLURM job array (--array=0-4):
#   0 → ablation_no_scalar          (A: control — no scalar encoder)
#   1 → ablation_scale_factors      (B: early fusion, hidden_dim=32)
#   2 → ablation_scale_factors_h64  (C: early fusion, hidden_dim=64)
#   3 → ablation_late_fusion        (D: late fusion only — post-transformer)
#   4 → ablation_both_fusion        (E: early + late fusion simultaneously)
#
# All runs share the same dataset (euler_cp_full, 153 samples), architecture,
# and training hyperparams — only the scalar_encoder block differs.
#
# Results end up in:
#   W&B project  : WP1_drag  (group: ablation_scalar_encoder)
#   Checkpoints  : /cluster/work/fuge/AISCO/wp1_cfd_ai/results/ablation_*/
#   Per-epoch CSV: <result_path>_epoch_metrics.csv  (one per run)
#
# Launch:
#   sbatch subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/slurm_scripts/experimental_scripts/ablation/ablation_scalar_encoder.sh

# --------------------------------------------------------------------------
CONFIGS=(
    "subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/config/experimental_configs/ablation/ablation_no_scalar.yaml"
    "subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/config/experimental_configs/ablation/ablation_scale_factors.yaml"
    "subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/config/experimental_configs/ablation/ablation_scale_factors_hidden64.yaml"
    "subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/config/experimental_configs/ablation/ablation_late_fusion.yaml"
    "subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/config/experimental_configs/ablation/ablation_both_fusion.yaml"
)

BASE_CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
# --------------------------------------------------------------------------

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME="$USER"
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

echo "=========================================="
echo "Ablation: ScalarEncoder sweep"
echo "Array task : $SLURM_ARRAY_TASK_ID / 4"
echo "Config     : $BASE_CONFIG"
echo "Job ID     : $SLURM_JOB_ID  (array: $SLURM_ARRAY_JOB_ID)"
echo "Started    : $(date)"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_CACHE_DIR="/cluster/scratch/$USERNAME/wandb_cache"

cd /cluster/home/$USERNAME/AISCO_code/
ulimit -n 65536

if [ ! -f "$BASE_CONFIG" ]; then
    echo "ERROR: config not found: $BASE_CONFIG"
    exit 1
fi

RUN_NAME=$(grep "name:" "$BASE_CONFIG" | head -1 | awk '{print $2}' | tr -d '"')
RESULTS_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/results"
mkdir -p "$RESULTS_DIR"

# Stamp the config with the SLURM job ID so each run gets a unique output dir
TEMP_CONFIG="/tmp/ablation_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"
sed "s/_RUN_ID_/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/g;
     s/_NAME_/${RUN_NAME}/g" \
    "$BASE_CONFIG" > "$TEMP_CONFIG"

echo "--- Resolved config ---"
cat "$TEMP_CONFIG"
echo "-----------------------"

python subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/main.py \
    -c "$TEMP_CONFIG" \
    2>&1

echo "=========================================="
echo "Finished: $(date)"
echo "Total elapsed: $SECONDS seconds"
echo "=========================================="
