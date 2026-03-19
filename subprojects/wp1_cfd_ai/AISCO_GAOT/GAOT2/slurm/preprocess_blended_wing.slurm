#!/bin/bash
#SBATCH --job-name=preprocess_bwb
#SBATCH --partition=normal.4h
#SBATCH --account=es_fuge
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --output=/cluster/home/elihuber/logs/preprocess_bwb_%j.out
#SBATCH --error=/cluster/home/elihuber/logs/preprocess_bwb_%j.err

echo "Job started on $(hostname) at $(date)"
echo "CPUs available: $SLURM_CPUS_PER_TASK"
echo "Cases to process: 8832"

module purge
module load stack/2024-06 python_cuda/3.11.6
source /cluster/home/elihuber/AISCO_code/aisco_env/bin/activate

SCRIPT_DIR=/cluster/home/elihuber/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT_working/dataset/blended_wing

OUTPUT_DIR=/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_blended_wing_new
echo "=== Resuming processing (keeping existing files) ==="
mkdir -p "$OUTPUT_DIR"

echo "=== Processing Blended Wing Body VTK dataset ==="
python "$SCRIPT_DIR/process_blended_wing.py" \
    --output-dir "$OUTPUT_DIR"

echo "Job finished at $(date)"
