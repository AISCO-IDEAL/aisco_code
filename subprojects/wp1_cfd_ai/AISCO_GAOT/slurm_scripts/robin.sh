#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=Preprocess_Drivaer_Test
#SBATCH --time=04:00:00                   # Adjust time as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/roarnold/scratch_logs/Robin_TEST_%j.out
#SBATCH --error=/cluster/scratch/roarnold/scratch_logs/Robin_TEST_%j.err

#------------------------------------------------------------------------------
# Define username variable (must come after all #SBATCH directives)
#------------------------------------------------------------------------------

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python/3.11.6  eth_proxy


USERNAME="roarnold"
LOG_DIR="/cluster/scratch/roarnold/scratch_logs"  # Define a log directory for GPU/CPU info
##mkdir -p "$LOG_DIR"  # Ensure the log directory exists
#------------------------------------------------------------------------------
# Print job information
#------------------------------------------------------------------------------
echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated memory per CPU: $SLURM_MEM_PER_CPU MB"
#echo "Number of GPUs: $SLURM_GPUS"
#echo "GPU type: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
#echo "Allocated memory per GPU: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
echo "Working directory: $PWD"
echo "=========================================="
#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Activate the Python virtual environment
#------------------------------------------------------------------------------
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"
#------------------------------------------------------------------------------
# Verify GPU access
#------------------------------------------------------------------------------
#echo "Verifying GPU access..."
#nvidia-smi
#echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
#------------------------------------------------------------------------------

# Set Hugging Face environment variables (not used for now)
#------------------------------------------------------------------------------
# export HF_HOME="$SCRATCH/hf_models"
# export HF_DATASETS_CACHE="$SCRATCH/hf_datasets"
# export HF_TOKEN="HuggingFace_TOKEN"  # Replace with your actual Huggingface token, create in Settings, Access Tokens, Read
#------------------------------------------------------------------------------
# Log GPU/CPU info to the same log directory
#------------------------------------------------------------------------------

##echo "GPU Info:" >> "$LOG_DIR/AISCO_TEST_${SLURM_JOB_ID}.out"
##nvidia-smi >> "$LOG_DIR/AISCO_TEST_${SLURM_JOB_ID}.out"
##echo "CPU Info:" >> "$LOG_DIR/AISCO_TEST_${SLURM_JOB_ID}.out"
##lscpu >> "$LOG_DIR/AISCO_TEST_${SLURM_JOB_ID}.out"
#------------------------------------------------------------------------------
# Run the Relevant script
#------------------------------------------------------------------------------
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:$PYTHONPATH"
cd /cluster/home/$USERNAME/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT_working
#python -m src.main

python src/dataloader/test_dataloader.py

#------------------------------------------------------------------------------
# Print job end information
#------------------------------------------------------------------------------
echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="

## chmod +x ~/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/slurm_scripts/robin.sh
## sbatch ~/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/slurm_scripts/robin.sh
## watch -n 5 squeue -u roarnold
