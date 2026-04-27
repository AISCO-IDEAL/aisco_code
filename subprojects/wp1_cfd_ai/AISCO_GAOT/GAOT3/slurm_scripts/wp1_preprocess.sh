#!/bin/bash
#SBATCH --job-name=GAOT3_preprocess
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/scratch_logs/GAOT3_preprocess_%j.out
#SBATCH --error=/cluster/scratch/%u/scratch_logs/GAOT3_preprocess_%j.err

# RUN COMMAND:
# sbatch subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/slurm_scripts/wp1_preprocess.sh
# Optional: provide a CPU partition explicitly if your cluster requires it,
# e.g. sbatch --partition=<cpu_partition> subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/slurm_scripts/wp1_preprocess.sh

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME="$USER"
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

echo "=========================================="
echo "Preprocessing job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated memory per CPU: $SLURM_MEM_PER_CPU MB"
echo "GPU request: none (CPU-only preprocessing)"
echo "Working directory: $PWD"
echo "=========================================="

export no_proxy=$(echo "$no_proxy" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "$NO_PROXY" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:$PYTHONPATH"

cd /cluster/home/$USERNAME/AISCO_code/

umask 0002
ulimit -n 65536

# --------------------
# Preprocessing options
# --------------------
# Choose datasets from: euler drivaer ahmed blended_wing
# Example: DATASETS=("euler" "drivaer")
# Existing <output_path>/<dataset>/<sample>/data.pt files are skipped on reruns.
DATASETS=("ahmed")

# Leave empty ("") to use PREPROCESSING_CONFIG default output_path,
# or set a custom base output path.
OUTPUT_PATH=""

# Worker overrides per dataset.
EULER_WORKERS=4
DRIVAER_WORKERS=2
AHMED_WORKERS=2
BLENDED_WING_WORKERS=8

# Decimation ratio for drivaer/ahmed only, used 0.5 for ahmed and 0.0 for drivaer
TARGET_REDUCTION=0.0

PREPROCESS_SCRIPT="subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/dataset/preprocess_all_datasets.py"
if [ ! -f "$PREPROCESS_SCRIPT" ]; then
    echo "Preprocess script not found: $PREPROCESS_SCRIPT"
    exit 1
fi

echo "Datasets: ${DATASETS[*]}"
if [ -n "$OUTPUT_PATH" ]; then
    echo "Output path override: $OUTPUT_PATH"
else
    echo "Output path override: <none, using PREPROCESSING_CONFIG default>"
fi

aargs=(
    --datasets "${DATASETS[@]}"
    --euler-workers "$EULER_WORKERS"
    --drivaer-workers "$DRIVAER_WORKERS"
    --ahmed-workers "$AHMED_WORKERS"
    --blended-wing-workers "$BLENDED_WING_WORKERS"
    --target-reduction "$TARGET_REDUCTION"
)

if [ -n "$OUTPUT_PATH" ]; then
    aargs+=(--output-path "$OUTPUT_PATH")
fi

python "$PREPROCESS_SCRIPT" "${aargs[@]}" 2>&1

echo "=========================================="
echo "Preprocessing job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="
