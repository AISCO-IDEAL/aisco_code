#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=sdf2mesh_smoke
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5120
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out
umask 0002
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy
USERNAME=$(whoami)
source "/cluster/home/$USERNAME/AISCO_code/aisco_env/bin/activate"
cd "/cluster/home/$USERNAME/AISCO_code"
nvidia-smi
python tests/wp5/src/smoke_sdf2mesh.py
