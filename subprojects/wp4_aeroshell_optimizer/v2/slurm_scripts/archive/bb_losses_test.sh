#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=BBLoss_Test_SLURM
#SBATCH --time=00:10:00                   # Adjust time as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_4090:1
##SBATCH --output=/cluster/home/eschaerer/AISCO_code/subprojects/wp4_aeroshell_optimizer/logs/AISCO_TEST_%j.out     #adjust path as needed
#SBATCH --gpus=1  # rtx_4090:1
#SBATCH --output=/cluster/home/clhuber/logs/AISCO_TEST_%j.out     #adjust path as needed
#SBATCH --error=/cluster/home/clhuber/logs/AISCO_TEST_%j.err      #adjust path as needed
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=eschaerer@ethz.ch
#------------------------------------------------------------------------------
# Define username variable (must come after all #SBATCH directives)
#------------------------------------------------------------------------------
umask 0002 # ensure group write permissions for logs and models

# Allow W&B to route through the cluster proxy (eth_proxy sets no_proxy which blocks api.wandb.ai)
export no_proxy=$(echo "$no_proxy" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "$NO_PROXY" | sed 's/api\.wandb\.ai,\?//g; s/,$//')

USERNAME=$(whoami)
LOG_DIR="/cluster/home/$USERNAME/AISCO_code/subprojects/wp4_aeroshell_optimizer/logs"
mkdir -p "$LOG_DIR"  # Ensure the log directory exists
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
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6  eth_proxy
#------------------------------------------------------------------------------
# Activate the Python virtual environment
#------------------------------------------------------------------------------
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"
### DEBUG INFO ###
echo "Python executable: $(which python3)"
echo "Python version: $(python3 --version)"

### END DEBUG INFO ###
#------------------------------------------------------------------------------
# Verify GPU access
#------------------------------------------------------------------------------
echo "Verifying GPU access..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
#------------------------------------------------------------------------------
# Set WANDB environment variables (not used for now)
#------------------------------------------------------------------------------
# export WANDB_API_KEY="API_KEY_HERE"  # connect to wandb, then go to wandb.ai/settings, scroll down to API key and reveal
# export WANDB_PROJECT="test"
# export WANDB_ENTITY="aisco"
#------------------------------------------------------------------------------
# Set Hugging Face environment variables (not used for now)
#------------------------------------------------------------------------------
# export HF_HOME="$SCRATCH/hf_models"
# export HF_DATASETS_CACHE="$SCRATCH/hf_datasets"
# export HF_TOKEN="HuggingFace_TOKEN"  # Replace with your actual Huggingface token, create in Settings, Access Tokens, Read
#------------------------------------------------------------------------------
# Log GPU/CPU info to the same log directory
#------------------------------------------------------------------------------
echo "GPU Info:" >> "$LOG_DIR/AISCO_TEST_${SLURM_JOB_ID}.out"
nvidia-smi >> "$LOG_DIR/AISCO_TEST_${SLURM_JOB_ID}.out"
echo "CPU Info:" >> "$LOG_DIR/AISCO_TEST_${SLURM_JOB_ID}.out"
lscpu >> "$LOG_DIR/AISCO_TEST_${SLURM_JOB_ID}.out"
#------------------------------------------------------------------------------
# Run the Relevant script
#------------------------------------------------------------------------------
################################################################################
# Set PYTHONPATH to include AISCO_code directory
################################################################################
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code:$PYTHONPATH" # new
################################################################################
# Navigate to the subprojects root
################################################################################

cd /cluster/home/$USERNAME/AISCO_code
################################################################################
# Run wp4 main as a module (so relative imports work!)
################################################################################
python3 -m  tests.wp4_aeroshell_optimizer.test_bb_losses\
    --save_model \
    --problem_id="AISCO_WP4_Test1" 2>&1 # 2>&1 leitet alle Fehler (stderr) in die .out-Datei um, sodass du siehst, was schiefgeht
#------------------------------------------------------------------------------
# Print job end information
#------------------------------------------------------------------------------
echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="

## chmod +x ~/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/bb_losses_test.sh
## sbatch ~/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/bb_losses_test.sh
## watch -n 5 squeue -u $USER
