#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=GAOT3_CP_new
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/%u/scratch_logs/GAOT3_CP_new_%j.out
#SBATCH --error=/cluster/scratch/%u/scratch_logs/GAOT3_CP_new_%j.err

#RUN COMMAND: sbatch subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/slurm_scripts/wp1_main.sh

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME="$USER"
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

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

echo "Verifying GPU access..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export no_proxy=$(echo "$no_proxy" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "$NO_PROXY" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:$PYTHONPATH"
export WANDB_CACHE_DIR="/cluster/scratch/$USERNAME/wandb_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /cluster/home/$USERNAME/AISCO_code/

umask 0002
ulimit -n 65536

BASE_CONFIG="subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/config/main_config.yaml"

if [ ! -f "$BASE_CONFIG" ]; then
    echo "Base config file not found: $BASE_CONFIG"
    exit 1
fi

RUN_NAME=$(grep "name:" $BASE_CONFIG | head -1 | awk '{print $2}' | tr -d '"')

RESULTS_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/results"
mkdir -p $RESULTS_DIR

TEMP_CONFIG="/tmp/run_${SLURM_JOB_ID}_config.yaml"
sed "s|_NAME_|${RUN_NAME}|g; s|_RUN_ID_|${SLURM_JOB_ID}|g" $BASE_CONFIG > $TEMP_CONFIG

echo "Contents of the temporary config file:"
cat $TEMP_CONFIG

python subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/main.py \
    -c $TEMP_CONFIG \
    2>&1

echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="
