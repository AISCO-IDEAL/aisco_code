#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=WP1_laeuft
#SBATCH --time=2:00:00          # Adjust time as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/home/%u/logs/WP1_main_TEST_%j.out

#------------------------------------------------------------------------------
# Define username variable (must come after all #SBATCH directives)
#------------------------------------------------------------------------------

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6  eth_proxy


USERNAME=$(whoami)
LOG_DIR="/cluster/home/$USERNAME/logs"  # Define a log directory for GPU/CPU info
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
echo "Number of GPUs: $SLURM_GPUS"
echo "GPU type: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Allocated memory per GPU: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
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
echo "Verifying GPU access..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
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
export WANDB_CACHE_DIR="/cluster/scratch/$USERNAME/wandb_cache"

# Change to GAOT3 directory
cd /cluster/home/$USERNAME/AISCO_code/

umask 0002
ulimit -n 65536

# Define your base config
BASE_CONFIG="subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/config/main_config.yaml"

# Extract the 'name' parameter from the config
RUN_NAME=$(grep "name:" $BASE_CONFIG | head -1 | awk '{print $2}' | tr -d '"')

# Create results folder structure
RESULTS_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/results"
mkdir -p $RESULTS_DIR

# Set the path for the temporary unique config
TEMP_CONFIG="/tmp/run_${SLURM_JOB_ID}_config.yaml"

# Substitute both _NAME_ and _RUN_ID_ tokens in the config
sed "s|_NAME_|${RUN_NAME}|g; s|_RUN_ID_|${SLURM_JOB_ID}|g" $BASE_CONFIG > $TEMP_CONFIG

echo "Contents of the temporary config file:"
cat $TEMP_CONFIG

if [ ! -f "$BASE_CONFIG" ]; then
    echo "Base config file not found: $BASE_CONFIG"
    exit 1
fi

if [ ! -s "$BASE_CONFIG" ]; then
    echo "Base config file is empty: $BASE_CONFIG"
    exit 1
fi

# 2. Run the AI using the newly generated unique config file
python subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/main.py \
    -c $TEMP_CONFIG \
    2>&1

#------------------------------------------------------------------------------
# Print job end information
#------------------------------------------------------------------------------
echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="

## chmod +x ~/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/slurm_scripts/wp1_main.sh
## sbatch ~/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/slurm_scripts/wp1_main.sh
## watch -n 1 squeue -u $USER
