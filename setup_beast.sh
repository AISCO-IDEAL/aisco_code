#!/usr/bin/env bash
# Setup script for AISCO_code on a local Ubuntu machine with RTX 5090.
# This script is intentionally strict and only supports RTX 5090 (sm_120).

set -euo pipefail

ENV_NAME="${ENV_NAME:-aisco_env}"
REQUIRED_PYTHON_VERSION="3.11.6"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_DIR="$PROJECT_ROOT/$ENV_NAME"
VENV_PYTHON="$VENV_DIR/bin/python3"

cd "$PROJECT_ROOT"

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command '$cmd' was not found."
        exit 1
    fi
}

check_python() {
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "Error: Python binary '$PYTHON_BIN' not found."
        echo "Use Python $REQUIRED_PYTHON_VERSION and rerun with:"
        echo "  PYTHON_BIN=/path/to/python3.11 ./setup_beast.sh"
        exit 1
    fi

    local version
    version="$($PYTHON_BIN --version 2>&1 | awk '{print $2}')"
    if [[ "$version" != "$REQUIRED_PYTHON_VERSION" ]]; then
        echo "Error: Python $REQUIRED_PYTHON_VERSION is required, found $version."
        exit 1
    fi

    if ! "$PYTHON_BIN" -c "import _ctypes" >/dev/null 2>&1; then
        echo "Error: Python is missing _ctypes. Install libffi-dev and rebuild Python."
        exit 1
    fi
}

check_gpu_5090() {
    require_cmd nvidia-smi

    local gpu_name
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | sed 's/^ *//;s/ *$//')"

    if [[ "$gpu_name" != *"RTX 5090"* && "$gpu_name" != *"GeForce RTX 5090"* ]]; then
        echo "Error: this setup script only supports RTX 5090. Detected: '$gpu_name'"
        exit 1
    fi

    echo "Detected supported GPU: $gpu_name"
}

detect_cuda_home() {
    if command -v nvcc >/dev/null 2>&1; then
        local nvcc_path
        nvcc_path="$(readlink -f "$(command -v nvcc)")"
        export CUDA_HOME="$(dirname "$(dirname "$nvcc_path")")"
    elif [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
        export CUDA_HOME="/usr/local/cuda"
    elif [[ -x "/usr/local/cuda-12.8/bin/nvcc" ]]; then
        export CUDA_HOME="/usr/local/cuda-12.8"
    else
        echo "Error: CUDA toolkit (nvcc) not found. Install CUDA toolkit and retry."
        exit 1
    fi

    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "CUDA_HOME=$CUDA_HOME"
}

create_or_reuse_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "Creating venv: $VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        echo "Reusing existing venv: $VENV_DIR"
    fi

    if [[ ! -x "$VENV_PYTHON" ]]; then
        echo "Error: venv python not found at $VENV_PYTHON"
        exit 1
    fi
}

install_dependencies() {
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel build

    # Beast path (RTX 5090): install a CUDA 12.8 torch build with sm_120 support.
    "$VENV_PYTHON" -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128

    # Install project dependencies and editable package.
    "$VENV_PYTHON" -m pip install -e .

    "$VENV_PYTHON" -m pre_commit install || true
}

verify_torch_cuda_5090() {
    "$VENV_PYTHON" - <<'PY'
import sys
import torch

print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)

if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    print("Error: CUDA is not available to PyTorch.")
    sys.exit(1)

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
arch_list = torch.cuda.get_arch_list()

print("device", name)
print("capability", cap)
print("arch_list", arch_list)

if "5090" not in name:
    print(f"Error: expected RTX 5090, got '{name}'")
    sys.exit(1)

if cap != (12, 0):
    print(f"Error: expected capability (12, 0), got {cap}")
    sys.exit(1)

if "sm_120" not in arch_list:
    print("Error: installed torch build does not include sm_120 kernels.")
    sys.exit(1)

x = torch.tensor([1.0], device="cuda")
y = x + 1.0
print("CUDA smoke test OK:", y.item())
PY
}

echo "Starting RTX 5090-only setup..."
require_cmd awk
require_cmd sed
check_python
check_gpu_5090
detect_cuda_home
create_or_reuse_venv
install_dependencies
verify_torch_cuda_5090

echo ""
echo "Setup complete (RTX 5090 validated)."
echo "Activate with: source $ENV_NAME/bin/activate"
