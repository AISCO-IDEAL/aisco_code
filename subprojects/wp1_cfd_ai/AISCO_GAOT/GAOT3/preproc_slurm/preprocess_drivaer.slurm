#!/bin/bash
#SBATCH --job-name=preprocess_drivaer
#SBATCH --partition=normal.4h
#SBATCH --account=es_fuge
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/home/roarnold/logs/preprocess_drivaer_%j.out
#SBATCH --error=/cluster/home/roarnold/logs/preprocess_drivaer_%j.err

umask 0002

echo "Job started on $(hostname) at $(date)"
echo "CPUs available: $SLURM_CPUS_PER_TASK"

module purge
module load stack/2024-06 python_cuda/3.11.6
source /cluster/home/roarnold/AISCO_code/aisco_env/bin/activate

SCRIPT_PATH=/cluster/home/roarnold/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/dataset/preprocess_all_datasets.py

OUTPUT_ROOT=/cluster/work/fuge/AISCO/wp1_cfd_ai
echo "=== Processing canonical DrivAer dataset into $OUTPUT_ROOT/processed_drivaer_cp_downsampled ==="

python "$SCRIPT_PATH" \
    --datasets drivaer \
    --output-root "$OUTPUT_ROOT" \
    --drivaer-workers 10 \
    --target-reduction 0.99

echo "Job finished at $(date)"
