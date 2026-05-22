#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=latent_rollout_gif
#SBATCH --time=00:15:00                  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G                  # 16G total memory
#SBATCH --gpus=pro_6000:1                 # Matches your cluster's GPU profile
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out

umask 0002
set -euo pipefail

ts() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Load cluster environment modules
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

USERNAME=$(whoami)

# Activate python virtual environment
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"

# Set up paths and headless rendering variables
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=TRUE

# ==============================================================================
# FORCE PYVISTA / VTK INTO TRULY HEADLESS SURFACELESS HARDWARE RENDERING
# ==============================================================================
export PYOPENGL_PLATFORM="egl"
export EGL_PLATFORM="surfaceless"
export VTK_DEFAULT_OPENGL_WINDOW="vtkEGLRenderWindow"
export PYVISTA_OFF_SCREEN=true
# ==============================================================================

cd "/cluster/home/$USERNAME/AISCO_code/"

ts "Launching latent_rollout_gif.py..."

# Execute script with defaults 
"$PYTHON_BIN" /cluster/home/kdiehl/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
    --checkpoint "/cluster/work/fuge/AISCO/wp9/for_victor/6166b4e953/109_398564_good-paper/models/109_398564_good-paper_epoch2000.pth" \
    --n_frames 240 \
    --resolution 60 \
    --fps 24 \
    --mode spiral \
    --show_assembly \
    --height 2160 \
    --width 2560 \
    --panel_width 1280 \
    --no-show-anchors \
    --out /cluster/work/fuge/AISCO/wp9/4k/good-paper_epoch2000.gif 

ts "Job finished successfully."