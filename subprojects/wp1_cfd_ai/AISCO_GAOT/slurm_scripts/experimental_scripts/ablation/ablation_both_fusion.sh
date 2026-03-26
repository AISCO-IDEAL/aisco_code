#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=Ablation_BothFusion
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/%u/scratch_logs/Ablation_BothFusion_%j.out
#SBATCH --error=/cluster/scratch/%u/scratch_logs/Ablation_BothFusion_%j.err

# Ablation E: Both fusion — scale_factors injected before AND after the transformer.
# Tests whether dual conditioning outperforms either early or late fusion alone.
#
# Launch:
#   sbatch subprojects/wp1_cfd_ai/AISCO_GAOT/slurm_scripts/experimental_scripts/ablation/ablation_both_fusion.sh

BASE_CONFIG="subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/config/experimental_configs/ablation/ablation_both_fusion.yaml"

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME="$USER"
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

echo "=========================================="
echo "Ablation E: Both early + late scalar fusion"
echo "Config     : $BASE_CONFIG"
echo "Job ID     : $SLURM_JOB_ID"
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

TEMP_CONFIG="/tmp/ablation_both_fusion_${SLURM_JOB_ID}.yaml"
sed "s/_RUN_ID_/${SLURM_JOB_ID}/g;
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
