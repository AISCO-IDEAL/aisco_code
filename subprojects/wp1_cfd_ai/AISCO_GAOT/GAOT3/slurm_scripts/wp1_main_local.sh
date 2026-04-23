#!/bin/bash

#------------------------------------------------------------------------------
# 1. Environment Setup
#------------------------------------------------------------------------------
# Local execution for WP1 GAOT3 on native Linux
###########################################################################
# bash ~/Documents/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/slurm_scripts/wp1_main_local.sh
###########################################################################

umask 0002 # ensure group write permissions for logs and models

USERNAME=$(whoami)
PROJECT_ROOT="$HOME/Documents/AISCO_code"
VENV_DIR="$PROJECT_ROOT/aisco_env"

#------------------------------------------------------------------------------
# 2. Print System Info (Simulating Slurm output)
#------------------------------------------------------------------------------
echo "=========================================="
echo "Starting local execution on Linux: $(date)"
echo "User: $USERNAME"
echo "Running on: $(hostname)"
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
# 4. Verify GPU
#------------------------------------------------------------------------------
echo "Verifying GPU access..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
fi

#------------------------------------------------------------------------------
# 5. Environment Configuration
#------------------------------------------------------------------------------
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="$PROJECT_ROOT/subprojects:$PYTHONPATH"
export AISCO_CODE_PATH="$PROJECT_ROOT"
export WANDB_CACHE_DIR="$HOME/.cache/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Allow W&B to route through the cluster proxy (eth_proxy sets no_proxy which blocks api.wandb.ai)
export no_proxy=$(echo "$no_proxy" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "$NO_PROXY" | sed 's/api\.wandb\.ai,\?//g; s/,$//')

cd "$PROJECT_ROOT"
ulimit -n 65536

#------------------------------------------------------------------------------
# 6. Handle Configuration
#------------------------------------------------------------------------------
BASE_CONFIG="subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/config/main_config.yaml"

if [ ! -f "$BASE_CONFIG" ]; then
    echo "Base config file not found: $BASE_CONFIG"
    exit 1
fi

RUN_NAME=$(grep "name:" $BASE_CONFIG | head -1 | awk '{print $2}' | tr -d '"')

RESULTS_DIR="$HOME/.local/AISCO/wp1_cfd_ai/results"
mkdir -p $RESULTS_DIR

TEMP_CONFIG="/tmp/run_local_$(date +%s)_config.yaml"
sed "s|_NAME_|${RUN_NAME}|g; s|_RUN_ID_|local_$(date +%s)|g" $BASE_CONFIG > $TEMP_CONFIG

echo "Contents of the temporary config file:"
cat $TEMP_CONFIG

#------------------------------------------------------------------------------
# 7. Run the script
#------------------------------------------------------------------------------
python3 subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/main.py \
    -c $TEMP_CONFIG \
    2>&1

echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
