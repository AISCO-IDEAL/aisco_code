#!/bin/bash
#SBATCH --partition=cn-long
#SBATCH --job-name=proc_ahmed
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/elihuber/scratch_logs/proc_ahmed_%j.out
#SBATCH --error=/cluster/scratch/elihuber/scratch_logs/proc_ahmed_%j.err

# Process all 500 ahmed VTP files into pressure-field .pt files
# Each ahmed sample is ~1.1M vertices → ~50MB per .pt file
# Total: ~25GB output

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME="elihuber"
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

echo "=========================================="
echo "Processing Ahmed VTP → pressure .pt files"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

cd /cluster/home/$USERNAME/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT_working

python -u dataset/process_vtp_pressure.py ahmed --workers 4

echo "=========================================="
echo "Finished: $(date)"
echo "Total elapsed: $SECONDS seconds"
echo "=========================================="

## sbatch ~/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/slurm_scripts/process_ahmed_pressure.sh
