#!/bin/bash
#SBATCH --job-name=preprocess_bwb
#SBATCH --partition=normal.4h
#SBATCH --account=es_fuge
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --output=/cluster/home/elihuber/logs/preprocess_bwb_%j.out
#SBATCH --error=/cluster/home/elihuber/logs/preprocess_bwb_%j.err

umask 0002

echo "Job started on $(hostname) at $(date)"
echo "CPUs available: $SLURM_CPUS_PER_TASK"
echo "Cases to process: 8832"

module purge
module load stack/2024-06 python_cuda/3.11.6
source /cluster/home/elihuber/AISCO_code/aisco_env/bin/activate

SCRIPT_PATH=/cluster/home/elihuber/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/dataset/preprocess_all_datasets.py

OUTPUT_ROOT=/cluster/work/fuge/AISCO/wp1_cfd_ai
echo "=== Processing canonical Blended Wing dataset into $OUTPUT_ROOT/processed_blended_wing_cp_full ==="

python "$SCRIPT_PATH" \
    --datasets blended_wing \
    --output-root "$OUTPUT_ROOT" \
    --blended-wing-workers 8

echo "Job finished at $(date)"
