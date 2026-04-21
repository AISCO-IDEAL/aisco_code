#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=LR_Sweep
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:1
#SBATCH --array=0-3
#SBATCH --output=/cluster/scratch/%u/scratch_logs/LR_Sweep_%A_%a.out
#SBATCH --error=/cluster/scratch/%u/scratch_logs/LR_Sweep_%A_%a.err

# Learning rate sweep on euler_cp_full (scalar_encoder disabled).
# All LR settings scaled uniformly from the newer_main_config.yaml baseline
# (lr=6e-4, max_lr=1e-3, min_lr=2e-4, final_lr=2e-5).
#
#   0 → lr_sweep_0.25x  (1/4x: lr=1.5e-4)
#   1 → lr_sweep_0.5x   (1/2x: lr=3e-4)
#   2 → lr_sweep_2x     (2x:   lr=1.2e-3)
#   3 → lr_sweep_4x     (4x:   lr=2.4e-3)
#
# Results:
#   W&B project : WP1_drag  (group: lr_sweep)
#   Checkpoints : /cluster/work/fuge/AISCO/wp1_cfd_ai/results/lr_sweep_*/
#
# Launch:
#   sbatch subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/slurm_scripts/experimental_scripts/lr_sweep.sh

# --------------------------------------------------------------------------
CONFIGS=(
    "subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/config/experimental_configs/lr_sweep/lr_sweep_0.25x.yaml"
    "subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/config/experimental_configs/lr_sweep/lr_sweep_0.5x.yaml"
    "subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/config/experimental_configs/lr_sweep/lr_sweep_2x.yaml"
    "subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/config/experimental_configs/lr_sweep/lr_sweep_4x.yaml"
)

BASE_CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
# --------------------------------------------------------------------------

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME="$USER"
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

echo "=========================================="
echo "LR Sweep"
echo "Array task : $SLURM_ARRAY_TASK_ID / 3"
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

TEMP_CONFIG="/tmp/lr_sweep_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"
sed "s/_RUN_ID_/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/g;
     s/_NAME_/${RUN_NAME}/g" \
    "$BASE_CONFIG" > "$TEMP_CONFIG"

echo "--- Resolved config ---"
cat "$TEMP_CONFIG"
echo "-----------------------"

python subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/main.py \
    -c "$TEMP_CONFIG" \
    2>&1

echo "=========================================="
echo "Finished: $(date)"
echo "Total elapsed: $SECONDS seconds"
echo "=========================================="
