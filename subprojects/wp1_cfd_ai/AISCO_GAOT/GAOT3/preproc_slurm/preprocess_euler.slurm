#!/bin/bash
#SBATCH --job-name=preprocess_euler
#SBATCH --partition=normal.4h
#SBATCH --account=es_fuge
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/home/elihuber/logs/preprocess_euler_%j.out
#SBATCH --error=/cluster/home/elihuber/logs/preprocess_euler_%j.err

umask 0002

echo "Job started on $(hostname) at $(date)"
echo "CPUs available: $SLURM_CPUS_PER_TASK"

module purge
module load stack/2024-06 python_cuda/3.11.6
source /cluster/home/elihuber/AISCO_code/aisco_env/bin/activate

SCRIPT_PATH=/cluster/home/elihuber/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/dataset/preprocess_all_datasets.py

OUTPUT_ROOT=/cluster/work/fuge/AISCO/wp1_cfd_ai
echo "=== Processing canonical Euler dataset into $OUTPUT_ROOT/processed_euler_cp_full ==="

python "$SCRIPT_PATH" \
    --datasets euler \
    --output-root "$OUTPUT_ROOT" \
    --euler-workers 4

echo "Job finished at $(date)"
