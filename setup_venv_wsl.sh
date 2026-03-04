#!/bin/bash
# setup_venv.sh

set -e  # Exit on error

ENV_NAME="aisco_env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_PIP="$PROJECT_ROOT/$ENV_NAME/bin/pip"
VENV_PYTHON="$PROJECT_ROOT/$ENV_NAME/bin/python3"

# 1. Option to skip the long base installation
read -p "Do you want to SKIP the standard environment creation/install (Select 'y' if aisco_env already exists)? [y/n]: " skip_base

if [[ "$skip_base" != "y" ]]; then
    if [ -d "$ENV_NAME" ]; then
        echo "clearing old environment..."
        rm -rf "$ENV_NAME"
    fi
    echo "🐍 Creating venv environment '$ENV_NAME'..."
    python3 -m venv "$ENV_NAME"

    source "$ENV_NAME/bin/activate"
    echo "📦 Upgrading pip and installing standard dependencies..."
    "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel build
    "$VENV_PYTHON" -m pip install -e .
else
    echo "⏭️ Skipping base install. Proceeding to CUDA tools..."
    if [ ! -f "$VENV_PIP" ]; then
        echo "❌ Error: $ENV_NAME/bin/pip not found. You must run the full install once."
        exit 1
    fi
fi

# 2. Improved CUDA Detection (Ensures pathing is correct for builds)
echo "🔧 Detecting CUDA toolkit..."
if command -v nvcc >/dev/null 2>&1; then
    NVCC_PATH=$(readlink -f "$(command -v nvcc)")
    CUDA_HOME_DETECTED=$(dirname "$(dirname "$NVCC_PATH")")
    if [[ "$CUDA_HOME_DETECTED" == "/usr" && -d "/usr/local/cuda" ]]; then
        CUDA_HOME_DETECTED="/usr/local/cuda"
    fi
    export CUDA_HOME="$CUDA_HOME_DETECTED"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "✅ Set CUDA_HOME=$CUDA_HOME"
fi

# 3. Kaolin Install (Using explicit VENV_PIP to bypass System Python)
echo "🔍 Checking if Kaolin should be installed..."
read -p "Do you want to install Kaolin? [y/n]: " install_kaolin
if [[ "$install_kaolin" == "y" ]]; then
    echo "🐍 Installing Kaolin using $VENV_PIP..."
    "$VENV_PIP" install -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html kaolin==0.18.0
    echo "✅ Kaolin installation complete."
fi

# 4. DISO Install
read -p "Do you want to install DISO? [y/n]: " install_diso
if [[ "$install_diso" == "y" ]]; then
    echo "🐍 Installing DISO using $VENV_PIP..."
    "$VENV_PIP" install diso --no-build-isolation --no-cache-dir
    echo "✅ DISO installation complete."
fi

echo "✅ Script finished! Activate with: source aisco_env/bin/activate"
