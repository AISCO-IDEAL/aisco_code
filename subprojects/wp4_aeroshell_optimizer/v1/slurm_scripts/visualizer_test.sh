#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=Clemens_Test_SLURM
#SBATCH --time=00:30:00                   # Adjust time as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G                  # Total 5G memory
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/home/clhuber/logs/Clemens_TEST_%j.out     #adjust path as needed
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=eschaerer@ethz.ch
#------------------------------------------------------------------------------
# Define username variable (must come after all #SBATCH directives)
#------------------------------------------------------------------------------
umask 0002 # ensure group write permissions for logs and models 

USERNAME=$(whoami)
LOG_DIR="/cluster/home/$USERNAME/AISCO_code/logs"
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
# Run the Relevant script
#------------------------------------------------------------------------------
################################################################################
# Set PYTHONPATH to include AISCO_code directory
################################################################################
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:$PYTHONPATH" # new
################################################################################
# Navigate to the subprojects root
################################################################################
cd /cluster/home/$USERNAME/AISCO_code/subprojects
################################################################################
# Run wp4 main as a module (so relative imports work!)
################################################################################
cd /cluster/home/$USERNAME/AISCO_code
python3 -m subprojects.wp4_aeroshell_optimizer.v1.src.losses.bb_loss.s_d_ffrom_cad_query\
    --save_model \
    --problem_id="AISCO_WP4_Test1" 2>&1 # 2>&1 leitet alle Fehler (stderr) in die .out-Datei um, sodass du siehst, was schiefgeht
#------------------------------------------------------------------------------
# Print job end information
#------------------------------------------------------------------------------
echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="

## chmod +x ~/AISCO_code/subprojects/wp4_aeroshell_optimizer/v1/slurm_scripts/visualizer_test.sh
## sbatch ~/AISCO_code/subprojects/wp4_aeroshell_optimizer/v1/slurm_scripts/visualizer_test.sh
## watch -n 5 squeue -u $USER