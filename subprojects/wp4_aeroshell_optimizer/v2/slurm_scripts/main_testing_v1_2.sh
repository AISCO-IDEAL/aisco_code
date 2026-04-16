#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=Main_v1.2_newMesh
#SBATCH --time=03:59:00                   # Extended for adaptive mesh extraction (L=4)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G                  # Total 40G memory
#SBATCH --gpus=rtx_3090:1
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=%u@ethz.ch
## run this command: sbatch ~/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/main_testing_v1_2.sh
#------------------------------------------------------------------------------
# Define username variable (must come after all #SBATCH directives)
#------------------------------------------------------------------------------
umask 0002 # ensure group write permissions for logs and models

# Allow W&B to route through the cluster proxy (eth_proxy sets no_proxy which blocks api.wandb.ai)
export no_proxy=$(echo "$no_proxy" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "$NO_PROXY" | sed 's/api\.wandb\.ai,\?//g; s/,$//')

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6  eth_proxy


USERNAME=$(whoami)
LOG_DIR="/cluster/scratch/$USERNAME/logs"
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
# PyVista Headless Environment Setup
#------------------------------------------------------------------------------
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code:$PYTHONPATH"
export PYVISTA_OFF_SCREEN=true

XVFB_PID=""
cleanup_xvfb() {
    if [[ -n "$XVFB_PID" ]] && kill -0 "$XVFB_PID" 2>/dev/null; then
        kill "$XVFB_PID" || true
        wait "$XVFB_PID" 2>/dev/null || true
    fi
}
trap cleanup_xvfb EXIT

stop_xvfb() {
    cleanup_xvfb
    XVFB_PID=""
}

start_xvfb() {
    local displays=(99 100 101 102)
    for d in "${displays[@]}"; do
        local disp=":${d}"
        Xvfb "$disp" -screen 0 1920x1080x24 -ac +extension GLX +render +iglx -noreset >/tmp/xvfb_${SLURM_JOB_ID}_${d}.log 2>&1 &
        local pid=$!
        sleep 1
        if kill -0 "$pid" 2>/dev/null; then
            XVFB_PID="$pid"
            export DISPLAY="$disp"
            return 0
        fi
    done
    return 1
}

run_pyvista_probe() {
    local log_file="/tmp/pyvista_probe_${SLURM_JOB_ID}.log"

    if python - <<'PY' >"$log_file" 2>&1
import os

import pyvista as pv

pv.OFF_SCREEN = True
plotter = pv.Plotter(off_screen=True, window_size=(128, 128))
plotter.add_mesh(pv.Sphere(radius=1.0), color="white")
img = plotter.screenshot(transparent_background=False)
plotter.close()

if img is None or getattr(img, "size", 0) == 0:
    raise RuntimeError("PyVista preflight produced empty image")

print(f"PyVista preflight successful. DISPLAY={os.environ.get('DISPLAY', '<unset>')}")
PY
    then
        echo "✓ PyVista preflight succeeded"
        return 0
    fi

    echo "✗ PyVista preflight failed"
    echo "  Log: $log_file"
    tail -n 20 "$log_file" || true
    return 1
}

echo "[GLX via Xvfb] Initializing PyVista headless environment..."
unset PYOPENGL_PLATFORM
unset EGL_PLATFORM
unset LIBGL_ALWAYS_SOFTWARE
unset MESA_LOADER_DRIVER_OVERRIDE
unset GALLIUM_DRIVER
unset LIBGL_ALWAYS_INDIRECT

stop_xvfb
if ! start_xvfb; then
    echo "ERROR: Failed to start Xvfb. Aborting."
    exit 1
fi

if ! run_pyvista_probe; then
    echo "ERROR: PyVista preflight failed. Aborting before training."
    exit 1
fi

echo "Headless rendering env:"
echo "  SELECTED_MODE=GLX_XVFB_DEFAULT"
echo "  PYVISTA_OFF_SCREEN=$PYVISTA_OFF_SCREEN"
echo "  PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-<unset>}"
echo "  EGL_PLATFORM=${EGL_PLATFORM:-<unset>}"
echo "  LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-<unset>}"
echo "  MESA_LOADER_DRIVER_OVERRIDE=${MESA_LOADER_DRIVER_OVERRIDE:-<unset>}"
echo "  GALLIUM_DRIVER=${GALLIUM_DRIVER:-<unset>}"
echo "  LIBGL_ALWAYS_INDIRECT=${LIBGL_ALWAYS_INDIRECT:-<unset>}"
echo "  DISPLAY=${DISPLAY:-<unset>}"
echo "  XVFB_PID=${XVFB_PID:-<unset>}"

#------------------------------------------------------------------------------
# Run the Relevant script
#------------------------------------------------------------------------------
cd /cluster/home/$USERNAME/AISCO_code/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure v1.2 model config (ConcatWIREV2) regardless of shared general_config.yaml state
CONFIG_FILE="subprojects/wp4_aeroshell_optimizer/v2/configs/general_config.yaml"
sed -i 's/model_name: "WIREV2"/model_name: "ConcatWIREV2"/' "$CONFIG_FILE"
sed -i 's|model_file_path: "subprojects/wp4_aeroshell_optimizer/v2/src/models/WIRE_v2.py"|model_file_path: "subprojects/wp4_aeroshell_optimizer/v2/src/models/concat_WIRE_v2.py"|' "$CONFIG_FILE"
sed -i 's/model_params: "WIRE_v2.yaml"/model_params: "concat_WIRE_v2.yaml"/' "$CONFIG_FILE"

python subprojects/wp4_aeroshell_optimizer/v2/src/main_files/main_v1_2_GINN_concat.py \
    --save_model \
    --problem_id="AISCO_WP4_Test1"  2>&1
#------------------------------------------------------------------------------
# Print job end information
#------------------------------------------------------------------------------
echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="
