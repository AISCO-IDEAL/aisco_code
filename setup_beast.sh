#!/usr/bin/env bash
# Local Ubuntu setup for AISCO_code on Beast (RTX 5090 / sm_120).

set -euo pipefail

ENV_NAME="${ENV_NAME:-aisco_env}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
PYTHON_VERSION_REQUIRED="3.11.6"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0+cu128}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0+cu128}"
CUDA_HOME_DEFAULT="${CUDA_HOME_DEFAULT:-/usr/local/cuda-12.8}"
INSTALL_KAOLIN="${INSTALL_KAOLIN:-0}"
BUILD_RENDERER="${BUILD_RENDERER:-1}"
INSTALL_DISO="${INSTALL_DISO:-0}"
BUILD_PYDEVELOPABILITY="${BUILD_PYDEVELOPABILITY:-0}"
PYDEVELOPABILITY_STRICT="${PYDEVELOPABILITY_STRICT:-0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/$ENV_NAME"
VENV_PYTHON="$VENV_DIR/bin/python3"
RENDERER_DIR="$PROJECT_ROOT/subprojects/fixed_renderer"

cd "$PROJECT_ROOT"

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || { echo "Error: missing required command '$1'."; exit 1; }
}

confirm_env_flag() {
    local name="$1"
    local prompt="$2"
    local default="$3"
    local current="${!name}"

    if [[ "$current" == "1" || "$current" == "0" ]]; then
        return 0
    fi

    if [[ ! -t 0 ]]; then
        printf -v "$name" '%s' "$default"
        return 0
    fi

    local answer=""
    read -r -p "$prompt [y/n]: " answer
    if [[ "${answer:-$default}" =~ ^[Yy]$ ]]; then
        printf -v "$name" '%s' "1"
    else
        printf -v "$name" '%s' "0"
    fi
}

echo "Checking Python..."
require_cmd awk
require_cmd sed
require_cmd nvidia-smi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python binary '$PYTHON_BIN' not found."
    exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" --version 2>&1 | awk '{print $2}')"
if [[ "$PYTHON_VERSION" != "$PYTHON_VERSION_REQUIRED" ]]; then
    echo "Error: Python $PYTHON_VERSION_REQUIRED is required, found $PYTHON_VERSION."
    exit 1
fi

if ! "$PYTHON_BIN" -c "import _ctypes" >/dev/null 2>&1; then
    echo "Error: Python is missing _ctypes."
    echo "Rebuild Python 3.11.6 with libffi present, then rerun this script."
    exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | sed 's/^ *//;s/ *$//')"
if [[ "$GPU_NAME" != *"RTX 5090"* && "$GPU_NAME" != *"GeForce RTX 5090"* ]]; then
    echo "Error: this script only supports RTX 5090. Detected '$GPU_NAME'."
    exit 1
fi
echo "Detected GPU: $GPU_NAME"

if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
elif [[ -x "$CUDA_HOME_DEFAULT/bin/nvcc" ]]; then
    export CUDA_HOME="$CUDA_HOME_DEFAULT"
else
    echo "Error: nvcc not found. Install CUDA 12.8 toolkit first."
    exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "CUDA_HOME=$CUDA_HOME"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating venv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "Installing base tooling..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel build

echo "Installing PyTorch for CUDA 12.8 / RTX 5090..."
"$VENV_PYTHON" -m pip install --upgrade --index-url "$TORCH_INDEX_URL" \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION"

echo "Installing project dependencies..."
"$VENV_PYTHON" -m pip install --no-deps -e .
"$VENV_PYTHON" -m pip install \
    "gymnasium>=1.0.0" \
    "pandas>=2.2.3" \
    "matplotlib>=3.9.3" \
    "numpy>=2.0.0" \
    "ruff>=0.1.0" \
    "pre-commit>=3.0.0" \
    "mypy>=1.19.1" \
    "fast-simplification>=0.1.13" \
    "wandb[media]>=0.23.1" \
    "types-paramiko" \
    "types-pytz" \
    "types-requests" \
    "types-PyYAML"
"$VENV_PYTHON" -m pre_commit install || true

confirm_env_flag INSTALL_KAOLIN "Install Kaolin?" "n"
confirm_env_flag BUILD_RENDERER "Build fixed_renderer for RTX 5090?" "y"
confirm_env_flag INSTALL_DISO "Install DISO?" "n"
confirm_env_flag BUILD_PYDEVELOPABILITY "Build pydevelopability C++ extension?" "n"

if [[ "$INSTALL_KAOLIN" == "1" ]]; then
    TORCH_PLAIN_VERSION="$("$VENV_PYTHON" - <<'PY'
import torch
print(torch.__version__.split("+")[0])
PY
)"
    CUDA_TAG="$("$VENV_PYTHON" - <<'PY'
import torch
print((torch.version.cuda or "").replace(".", ""))
PY
)"
    KAOLIN_INDEX="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${TORCH_PLAIN_VERSION}_cu${CUDA_TAG}.html"
    echo "Installing Kaolin from $KAOLIN_INDEX"
    "$VENV_PYTHON" -m pip uninstall -y kaolin >/dev/null 2>&1 || true
    "$VENV_PYTHON" -m pip install -f "$KAOLIN_INDEX" kaolin
fi

if [[ "$BUILD_RENDERER" == "1" ]]; then
    echo "Building fixed_renderer with TORCH_CUDA_ARCH_LIST=12.0..."
    (
        cd "$RENDERER_DIR"
        export TORCH_CUDA_ARCH_LIST=12.0
        "$VENV_PYTHON" setup.py build_ext --inplace --force
    )

    echo "Verifying fixed_renderer imports..."
    PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$VENV_PYTHON" - <<'PY'
import importlib
for name in (
    "subprojects.fixed_renderer.cuda.load_textures",
    "subprojects.fixed_renderer.cuda.rasterize",
    "subprojects.fixed_renderer.cuda.create_texture_image",
):
    importlib.import_module(name)
    print("import ok:", name)
PY
fi

if [[ "$INSTALL_DISO" == "1" ]]; then
    echo "Installing DISO..."
    "$VENV_PYTHON" -m pip install diso --no-build-isolation
fi

if [[ "$BUILD_PYDEVELOPABILITY" == "1" ]]; then
    echo "Building pydevelopability..."
    "$VENV_PYTHON" -m pip install --upgrade pybind11 cmake
    PYDEV_ROOT="$PROJECT_ROOT/subprojects/pydevelopability"
    BUILD_DIR="$PYDEV_ROOT/build_manual"
    CC_ROOT="${CC_ROOT:-$PROJECT_ROOT/subprojects/concentrate_curvature}"
    DEV_REPO="${DEV_REPO:-$CC_ROOT/DevelopabilityOfTriangleMeshes}"
    LIBIGL_INCLUDE="${LIBIGL_INCLUDE:-$CC_ROOT/libigl-modern/include}"
    PYBIND11_DIR="$("$VENV_PYTHON" -m pybind11 --cmakedir 2>/dev/null || true)"

    if [[ ! -d "$PYBIND11_DIR" || ! -f "$DEV_REPO/include/developableflow/fixed_topology_core.cpp" || ! -f "$LIBIGL_INCLUDE/igl/is_border_vertex.h" ]]; then
        if [[ "$PYDEVELOPABILITY_STRICT" == "1" ]]; then
            echo "Error: missing pydevelopability build prerequisites."
            exit 1
        fi
        echo "Skipping pydevelopability: missing prerequisites."
    else
        rm -rf "$BUILD_DIR"
        mkdir -p "$BUILD_DIR"
        (
            cd "$BUILD_DIR"
            cmake .. \
                -DCMAKE_BUILD_TYPE=Release \
                -Dpybind11_DIR="$PYBIND11_DIR" \
                -DCC_ROOT="$CC_ROOT" \
                -DDEV_REPO="$DEV_REPO" \
                -DLIBIGL_INCLUDE="$LIBIGL_INCLUDE"
            make -j"$(nproc)"
            cp "$(find . -name '_devcore*.so' | head -n1)" "$PYDEV_ROOT/pydevelopability/"
        )
        "$VENV_PYTHON" -m pip install --no-deps -e .
    fi
fi

echo "Verifying PyTorch CUDA support..."
"$VENV_PYTHON" - <<'PY'
import sys
import torch

print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)

if not torch.cuda.is_available():
    print("Error: torch.cuda.is_available() is False")
    sys.exit(1)

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
arch_list = torch.cuda.get_arch_list()

print("device", name)
print("capability", cap)
print("arch_list", arch_list)

if "5090" not in name:
    print("Error: expected RTX 5090")
    sys.exit(1)
if cap != (12, 0):
    print(f"Error: expected capability (12, 0), got {cap}")
    sys.exit(1)
if "sm_120" not in arch_list:
    print("Error: installed torch build does not include sm_120 kernels")
    sys.exit(1)

x = torch.tensor([1.0], device="cuda")
print("CUDA smoke test OK:", (x + 1).item())
PY

cat <<EOF

Setup complete.
Activate with: source $ENV_NAME/bin/activate

Optional extras:
  $VENV_PYTHON -m pip install '.[wp1_cfd_ai]'
  $VENV_PYTHON -m pip install '.[wp2_assembler]'
  $VENV_PYTHON -m pip install '.[wp3_solar_estimator]'
  $VENV_PYTHON -m pip install '.[wp4_aeroshell_optimizer]'
  $VENV_PYTHON -m pip install '.[wp7_gui]'
  $VENV_PYTHON -m pip install '.[testing]'
  $VENV_PYTHON -m pip install '.[pydevelopability]'
  $VENV_PYTHON -m pip install '.[all]'
EOF
