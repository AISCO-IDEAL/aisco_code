#!/bin/bash
#SBATCH --partition=cn-long
#SBATCH --job-name=proc_drivaer
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/elihuber/scratch_logs/proc_drivaer_%j.out
#SBATCH --error=/cluster/scratch/elihuber/scratch_logs/proc_drivaer_%j.err

# Process all 500 DrivAer VTP files into pressure-field .pt files
# Each DrivAer sample is ~8.8M vertices → ~400MB per .pt file
# Total: ~200GB output — make sure you have enough space
# Memory: each worker needs ~16GB peak (8.8M vertices × multiple float arrays)

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME="elihuber"
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

echo "=========================================="
echo "Processing DrivAer VTP → pressure .pt files"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

cd /cluster/home/$USERNAME/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT_working

# Use only 2 workers for DrivAer — each worker needs ~16GB for 8.8M vertices
python -u dataset/process_vtp_pressure.py drivaer --workers 2

echo "=========================================="
echo "Finished: $(date)"
echo "Total elapsed: $SECONDS seconds"
echo "=========================================="

## sbatch ~/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/slurm_scripts/process_drivaer_pressure.sh
