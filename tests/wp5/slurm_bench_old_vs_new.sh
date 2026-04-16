#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=MeshExtractor_Bench
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5120
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out
## Usage: sbatch ~/AISCO_code/tests/wp5/slurm_bench_old_vs_new.sh
umask 0002
export no_proxy=$(echo "${no_proxy:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "${NO_PROXY:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy
USERNAME=$(whoami)
echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS"
echo "GPU type: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "GPU memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "=========================================="
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"
nvidia-smi
cd /cluster/home/$USERNAME/AISCO_code
export WANDB_MODE=disabled
python tests/wp5/benchmark_old_vs_new_extractor.py \
    --resolutions 64 128 256 \
    --new-base-res 32 \
    --new-refine-levels 3 4 5 \
    --new-leaf-res 3 \
    --device cuda
echo "=========================================="
echo "Benchmark finished at: $(date)"
echo "=========================================="
