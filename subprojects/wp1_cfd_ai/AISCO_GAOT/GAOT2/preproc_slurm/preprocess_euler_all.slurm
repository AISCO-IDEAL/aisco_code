#!/bin/bash
#SBATCH --job-name=preprocess_euler_all
#SBATCH --partition=normal.4h
#SBATCH --account=es_fuge
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/home/elihuber/logs/preprocess_euler_all_%j.out
#SBATCH --error=/cluster/home/elihuber/logs/preprocess_euler_all_%j.err

umask 0002

echo "Job started on $(hostname) at $(date)"
echo "CPUs available: $SLURM_CPUS_PER_TASK"

module purge
module load stack/2024-06 python_cuda/3.11.6
source /cluster/home/elihuber/AISCO_code/aisco_env/bin/activate

SCRIPT_PATH=/cluster/home/elihuber/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/dataset/euler/process_fluent_pressure.py
OUTPUT_DIR=/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler_cp_full_all

echo "=== Processing all available Euler/FLUENT runs into $OUTPUT_DIR ==="
echo "    air_density=1.2041 kg/m3  v_inf=25 m/s (new run reference values)"

python "$SCRIPT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --air-density 1.2041

echo "Job finished at $(date)"
