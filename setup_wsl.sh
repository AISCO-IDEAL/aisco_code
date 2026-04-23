#!/bin/bash
# setup_linux.sh-compatible workflow (kept in setup_wsl.sh for backwards compatibility)

set -euo pipefail

ENV_NAME="${ENV_NAME:-aisco_env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_DIR="$PROJECT_ROOT/$ENV_NAME"
VENV_PIP="$VENV_DIR/bin/pip"
VENV_PYTHON="$VENV_DIR/bin/python3"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ask_yn() {
    local prompt="$1"
    local default="${2:-n}"
    local val=""

    if [ -t 0 ]; then
        read -r -p "$prompt" val
        val="${val:-$default}"
    else
        val="$default"
    fi
    printf '%s' "$val"
}

cd "$PROJECT_ROOT"

# Pick an interpreter that can actually create virtual environments.
if ! "$PYTHON_BIN" -m venv --help >/dev/null 2>&1; then
    if [ -x "/home/clhuber/.pyenv/versions/3.11.6/bin/python" ] && /home/clhuber/.pyenv/versions/3.11.6/bin/python -m venv --help >/dev/null 2>&1; then
        PYTHON_BIN="/home/clhuber/.pyenv/versions/3.11.6/bin/python"
        echo "Using fallback interpreter for venv: $PYTHON_BIN"
    else
        echo "Error: no Python interpreter with working 'venv' module found."
        echo "Install python3-venv or pass PYTHON_BIN=/path/to/python"
        exit 1
    fi
fi

SKIP_BASE="${SKIP_BASE:-$(ask_yn "Do you want to skip standard env creation/install? [y/n]: " "y")}"
INSTALL_KAOLIN="${INSTALL_KAOLIN:-$(ask_yn "Do you want to install Kaolin? [y/n]: " "n")}"
INSTALL_DISO="${INSTALL_DISO:-$(ask_yn "Do you want to install DISO? [y/n]: " "n")}"
BUILD_RENDERER="${BUILD_RENDERER:-$(ask_yn "Do you want to build Neural Renderer? [y/n]: " "n")}"

if [[ "$SKIP_BASE" != "y" ]]; then
    if [ -d "$VENV_DIR" ]; then
        echo "Clearing old environment at $VENV_DIR"
        rm -rf "$VENV_DIR"
    fi
    echo "Creating venv environment '$ENV_NAME'"
    "$PYTHON_BIN" -m venv "$VENV_DIR"

    echo "Upgrading pip and installing dependencies"
    "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel build
    "$VENV_PYTHON" -m pip install -e .
    "$VENV_PYTHON" -m pip install '.[all]'
else
    echo "Skipping base install"
    if [ ! -x "$VENV_PYTHON" ]; then
        echo "Error: $VENV_PYTHON not found. Run once with SKIP_BASE=n."
        exit 1
    fi
fi

# Detect CUDA toolkit for building extensions
echo "Detecting CUDA toolkit"
if command -v nvcc >/dev/null 2>&1; then
    NVCC_PATH="$(readlink -f "$(command -v nvcc)")"
    CUDA_HOME_DETECTED="$(dirname "$(dirname "$NVCC_PATH")")"
    if [[ "$CUDA_HOME_DETECTED" == "/usr" && -d "/usr/local/cuda" ]]; then
        CUDA_HOME_DETECTED="/usr/local/cuda"
    fi
    export CUDA_HOME="$CUDA_HOME_DETECTED"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "Set CUDA_HOME=$CUDA_HOME"
else
    echo "Warning: nvcc not found. CUDA extensions may fail to build."
fi

if [[ "$INSTALL_KAOLIN" == "y" ]]; then
    echo "Installing Kaolin"
    TORCH_MM="$($VENV_PYTHON - <<'PY'
import torch
v = torch.__version__.split('+')[0].split('.')
print(f"{v[0]}.{v[1]}.0")
PY
)"
    TORCH_CUDA_TAG="$($VENV_PYTHON - <<'PY'
import torch
cuda = torch.version.cuda or ""
print("cu" + cuda.replace(".", "") if cuda else "cpu")
PY
)"
    KAOLIN_INDEX_URL="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${TORCH_MM}_${TORCH_CUDA_TAG}.html"

    # Remove placeholder package if present
    "$VENV_PYTHON" -m pip uninstall -y kaolin >/dev/null 2>&1 || true

    set +e
    "$VENV_PYTHON" -m pip install -f "$KAOLIN_INDEX_URL" kaolin==0.18.0
    KAOLIN_RC=$?
    set -e
    if [ "$KAOLIN_RC" -ne 0 ]; then
        echo "Warning: Kaolin install failed for index: $KAOLIN_INDEX_URL"
        echo "Try a matching torch/kaolin pair (for example torch 2.4.x + cu121 with kaolin 0.18.0)."
    else
        echo "Kaolin installation complete"
    fi
fi

if [[ "$BUILD_RENDERER" == "y" ]]; then
    echo "Building fixed_renderer CUDA extensions"
    pushd "$PROJECT_ROOT/subprojects/fixed_renderer" >/dev/null
    "$VENV_PYTHON" setup.py build_ext --inplace
    popd >/dev/null
fi

if [[ "$INSTALL_DISO" == "y" ]]; then
    echo "Installing DISO"
    "$VENV_PYTHON" -m pip install diso --no-build-isolation --no-cache-dir
fi

echo "Setup script finished"
echo "Activate with: source $ENV_NAME/bin/activate"
