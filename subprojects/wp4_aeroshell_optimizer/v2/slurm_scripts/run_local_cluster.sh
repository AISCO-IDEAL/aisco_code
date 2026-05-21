#!/bin/bash

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

#------------------------------------------------------------------------------
# 1. Environment Setup
#------------------------------------------------------------------------------
# Instead of 'module load', we ensure our local environment is ready.
# Make sure you have installed: python3, python3-venv, and CUDA.
###########################################################################
# bash subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/run_local_cluster.sh
###########################################################################

umask 0002 # ensure group write permissions for logs and models


# Resolve repository root from this script location (v2/slurm_scripts -> repo root)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd -P)"
PROJECT_ROOT="${AISCO_PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
VENV_DIR="$PROJECT_ROOT/aisco_env"

#------------------------------------------------------------------------------
# 2. Print System Info (Simulating Slurm output)
#------------------------------------------------------------------------------
echo "=========================================="
echo "Starting local execution: $(date)"
echo "Working directory: $PWD"
echo "=========================================="

#------------------------------------------------------------------------------
# 3. Activate Virtual Environment
#------------------------------------------------------------------------------
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_DIR"
    exit 1
fi

#------------------------------------------------------------------------------
# 4. Verify GPU (WSL specific)
#------------------------------------------------------------------------------
echo "Verifying GPU access..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found. Ensure NVIDIA drivers are installed on Windows."
fi

#------------------------------------------------------------------------------
# 5. Run the script
#------------------------------------------------------------------------------
export PYTHONUNBUFFERED=TRUE
# Add the local subprojects folder to path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export AISCO_CODE_PATH="$PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Run the python command
python3 subprojects/wp4_aeroshell_optimizer/v2/src/main_files/main_v2.py  \
    --save_model \
    --problem_id="AISCO_WP4_Test1" \
    "$@"

echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
