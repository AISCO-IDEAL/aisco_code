#!/bin/bash
#SBATCH --job-name=preprocess_euler
#SBATCH --partition=normal.4h
#SBATCH --account=es_fuge
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/home/elihuber/logs/preprocess_euler_%j.out
#SBATCH --error=/cluster/home/elihuber/logs/preprocess_euler_%j.err

echo "Job started on $(hostname) at $(date)"
echo "CPUs available: $SLURM_CPUS_PER_TASK"

module purge
module load stack/2024-06 python_cuda/3.11.6
source /cluster/home/elihuber/AISCO_code/aisco_env/bin/activate

SCRIPT_DIR=/cluster/home/elihuber/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT_working/dataset/euler

OUTPUT_DIR=/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler_multi_new
echo "=== Clearing old data in $OUTPUT_DIR ==="
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=== Processing Euler multi-head (pos + force) ==="
python "$SCRIPT_DIR/process_my_euler_multi.py" \
    --output-dir "$OUTPUT_DIR" \
    --num-workers 12

echo "Job finished at $(date)"
