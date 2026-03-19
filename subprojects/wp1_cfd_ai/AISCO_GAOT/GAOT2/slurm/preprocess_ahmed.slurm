#!/bin/bash
#SBATCH --job-name=preprocess_ahmed
#SBATCH --partition=normal.4h
#SBATCH --account=es_fuge
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/cluster/home/elihuber/logs/preprocess_ahmed_%j.out
#SBATCH --error=/cluster/home/elihuber/logs/preprocess_ahmed_%j.err

echo "Job started on $(hostname) at $(date)"
echo "CPUs available: $SLURM_CPUS_PER_TASK"

module purge
module load stack/2024-06 python_cuda/3.11.6
source /cluster/home/elihuber/AISCO_code/aisco_env/bin/activate

SCRIPT_DIR=/cluster/home/elihuber/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT_working/dataset/ahmed

OUTPUT_DIR=/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_ahmed_multi_new
echo "=== Clearing old data in $OUTPUT_DIR ==="
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=== Processing Ahmed body PLY dataset ==="
python "$SCRIPT_DIR/process_ahmed_drag_params.py" \
    --output-dir "$OUTPUT_DIR" \
    --num-workers 28

echo "Job finished at $(date)"
