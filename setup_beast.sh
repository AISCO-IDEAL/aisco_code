#!/usr/bin/env bash
# Run from project root with:
#   chmod +x ./setup_beast.sh && PYTHON_BIN="/home/clhuber/.pyenv/versions/3.11.6/bin/python" ./setup_beast.sh
# If Python was built without _ctypes, this script can install required Ubuntu build deps,
# then instruct (or optionally auto-run) a Python 3.11.6 rebuild via pyenv.
# Setup script for AISCO_code on a local Ubuntu machine with RTX 5090.
# This script is intentionally strict and only supports RTX 5090 (sm_120).

set -euo pipefail

ENV_NAME="${ENV_NAME:-aisco_env}"
REQUIRED_PYTHON_VERSION="3.11.6"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
AUTO_INSTALL_SYSTEM_DEPS="${AUTO_INSTALL_SYSTEM_DEPS:-1}"
AUTO_REBUILD_PYTHON="${AUTO_REBUILD_PYTHON:-0}"
SYSTEM_DEPS_ATTEMPTED="${SYSTEM_DEPS_ATTEMPTED:-0}"
INSTALL_KAOLIN="${INSTALL_KAOLIN:-}"
INSTALL_DISO="${INSTALL_DISO:-}"
BUILD_PYDEVELOPABILITY="${BUILD_PYDEVELOPABILITY:-}"
PYDEVELOPABILITY_STRICT="${PYDEVELOPABILITY_STRICT:-0}"

APT_BUILD_DEPS=(
    libffi-dev
    build-essential
    zlib1g-dev
    libssl-dev
    libbz2-dev
    libreadline-dev
    libsqlite3-dev
    liblzma-dev
    tk-dev
    uuid-dev
)

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

install_python_build_deps_ubuntu() {
    if [[ "$SYSTEM_DEPS_ATTEMPTED" == "1" ]]; then
        return 0
    fi

    SYSTEM_DEPS_ATTEMPTED="1"

    if [[ "$AUTO_INSTALL_SYSTEM_DEPS" != "1" ]]; then
        return 0
    fi

    if ! command -v apt-get >/dev/null 2>&1; then
        return 0
    fi

    local sudo_prefix=""
    if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
        if command -v sudo >/dev/null 2>&1; then
            sudo_prefix="sudo"
        else
            echo "Warning: apt-get detected but sudo is unavailable; skipping system dependency install."
            return 0
        fi
    fi

    echo "Installing Python build dependencies (Ubuntu/Debian)..."
    if ! $sudo_prefix apt-get update; then
        echo "Warning: apt-get update failed. Continuing with existing environment."
        return 0
    fi

    if ! $sudo_prefix apt-get install -y "${APT_BUILD_DEPS[@]}"; then
        echo "Warning: apt-get install failed. Continuing with existing environment."
        return 0
    fi
}

print_python_rebuild_help() {
    local pyenv_hint=""
    if ! command -v pyenv >/dev/null 2>&1; then
        pyenv_hint=$(cat <<'EOF'
pyenv is not installed in this shell.

Install pyenv, then rebuild Python:

  curl https://pyenv.run | bash
  export PYENV_ROOT="$HOME/.pyenv"
  export PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init -)"
EOF
)
    fi

    cat <<EOF
Python $REQUIRED_PYTHON_VERSION was found, but _ctypes is missing.

Run the following commands, then rerun this setup script:

  sudo apt update
  sudo apt install -y ${APT_BUILD_DEPS[*]}
  pyenv install $REQUIRED_PYTHON_VERSION -f
  pyenv shell $REQUIRED_PYTHON_VERSION
  python -c "import _ctypes; print('ctypes ok')"
  cd "$PROJECT_ROOT"
    PYTHON_BIN="\$(which python)" ./setup_beast.sh

$pyenv_hint

Alternative without pyenv (manual source build):

    cd /tmp
    wget -q https://www.python.org/ftp/python/$REQUIRED_PYTHON_VERSION/Python-$REQUIRED_PYTHON_VERSION.tgz
    tar -xzf Python-$REQUIRED_PYTHON_VERSION.tgz
    cd Python-$REQUIRED_PYTHON_VERSION
    ./configure --prefix="\$HOME/.local/python-$REQUIRED_PYTHON_VERSION" --enable-optimizations --with-system-ffi
    make -j"\$(nproc)"
    make install
    "\$HOME/.local/python-$REQUIRED_PYTHON_VERSION/bin/python3.11" -c "import _ctypes; print('ctypes ok')"
    cd "$PROJECT_ROOT"
    PYTHON_BIN="\$HOME/.local/python-$REQUIRED_PYTHON_VERSION/bin/python3.11" ./setup_beast.sh

Optional: set AUTO_REBUILD_PYTHON=1 to let this script auto-run the pyenv rebuild step.
EOF
}

try_pyenv_rebuild_python() {
    if [[ "$AUTO_REBUILD_PYTHON" != "1" ]]; then
        return 1
    fi

    if ! command -v pyenv >/dev/null 2>&1; then
        echo "AUTO_REBUILD_PYTHON=1 was set, but pyenv is not installed."
        return 1
    fi

    echo "AUTO_REBUILD_PYTHON=1 set. Rebuilding Python $REQUIRED_PYTHON_VERSION via pyenv..."
    pyenv install "$REQUIRED_PYTHON_VERSION" -f
    pyenv shell "$REQUIRED_PYTHON_VERSION"

    local rebuilt_bin
    rebuilt_bin="$(command -v python)"
    if [[ -z "$rebuilt_bin" ]]; then
        echo "Error: pyenv rebuild completed but no python executable found in PATH."
        return 1
    fi

    if ! "$rebuilt_bin" -c "import _ctypes" >/dev/null 2>&1; then
        echo "Error: rebuilt Python still missing _ctypes."
        return 1
    fi

    PYTHON_BIN="$rebuilt_bin"
    echo "Using rebuilt Python: $PYTHON_BIN"
    return 0
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
        echo "Python is missing _ctypes."
        install_python_build_deps_ubuntu

        if try_pyenv_rebuild_python; then
            if ! "$PYTHON_BIN" -c "import _ctypes" >/dev/null 2>&1; then
                echo "Error: _ctypes still missing after attempted rebuild."
                print_python_rebuild_help
                exit 1
            fi
        else
            print_python_rebuild_help
            exit 1
        fi
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
    "$VENV_PYTHON" -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torchvision==0.23.0+cu128

    # Install base project dependencies.
    "$VENV_PYTHON" -m pip install -e .

    "$VENV_PYTHON" -m pre_commit install || true

    echo ""
    echo "Standard dependencies installed."
    echo ""
    echo "Optional dependencies can be installed manually:"
    echo "   - Workpackage 1 (wp1_cfd_ai):                  $VENV_PYTHON -m pip install '.[wp1_cfd_ai]'"
    echo "   - Workpackage 2 (wp2_assembler):               $VENV_PYTHON -m pip install '.[wp2_assembler]'"
    echo "   - Workpackage 3 (wp3_solar_estimator):         $VENV_PYTHON -m pip install '.[wp3_solar_estimator]'"
    echo "   - Workpackage 4 (wp4_aeroshell_optimizer):     $VENV_PYTHON -m pip install '.[wp4_aeroshell_optimizer]'"
    echo "   - Workpackage 7 GUI (wp7_gui):                 $VENV_PYTHON -m pip install '.[wp7_gui]'"
    echo "   - Testing dependencies:                        $VENV_PYTHON -m pip install '.[testing]'"
    echo "   - pydevelopability build dependencies:         $VENV_PYTHON -m pip install '.[pydevelopability]'"
    echo "   - All optional dependencies:                   $VENV_PYTHON -m pip install '.[all]'"
    echo ""
}

ask_yes_no() {
    local prompt="$1"
    local answer

    read -r -p "$prompt [y/n]: " answer
    [[ "$answer" == "y" || "$answer" == "Y" ]]
}

configure_optional_steps() {
    if [[ -z "$INSTALL_KAOLIN" ]]; then
        if ask_yes_no "Do you want to install Kaolin (GPU/CUDA-dependent)?"; then
            INSTALL_KAOLIN="1"
        else
            INSTALL_KAOLIN="0"
        fi
    fi

    if [[ -z "$INSTALL_DISO" ]]; then
        if ask_yes_no "Do you want to install DISO (GPU/CUDA-dependent)?"; then
            INSTALL_DISO="1"
        else
            INSTALL_DISO="0"
        fi
    fi

    if [[ -z "$BUILD_PYDEVELOPABILITY" ]]; then
        if ask_yes_no "Do you want to build pydevelopability C++ extension?"; then
            BUILD_PYDEVELOPABILITY="1"
        else
            BUILD_PYDEVELOPABILITY="0"
        fi
    fi
}

install_kaolin() {
    if [[ "$INSTALL_KAOLIN" != "1" ]]; then
        echo "Skipping Kaolin install (INSTALL_KAOLIN=$INSTALL_KAOLIN)."
        return 0
    fi

    local torch_version
    local cuda_version
    torch_version="$($VENV_PYTHON - <<'PY'
import torch
print(torch.__version__.split('+')[0])
PY
)"
    cuda_version="$($VENV_PYTHON - <<'PY'
import torch
v = torch.version.cuda
print(v.replace('.', '') if v else '')
PY
)"

    if [[ -z "$torch_version" || -z "$cuda_version" ]]; then
        echo "Error: could not determine torch/cuda versions for Kaolin installation."
        echo "       torch='$torch_version' cuda='$cuda_version'"
        exit 1
    fi

    local kaolin_index
    kaolin_index="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${torch_version}_cu${cuda_version}.html"

    echo "Installing Kaolin using index: $kaolin_index"
    if ! "$VENV_PYTHON" -m pip install -f "$kaolin_index" kaolin; then
        echo ""
        echo "Error: Kaolin installation failed for the current torch build."
        echo "       Torch version: $torch_version"
        echo "       CUDA version:  $cuda_version"
        echo "       Attempted URL: $kaolin_index"
        echo ""
        echo "This usually means NVIDIA does not provide a Kaolin wheel for this exact torch/CUDA combo."
        echo "Use a supported torch build for Kaolin, or rerun setup with INSTALL_KAOLIN=0 if you intentionally want to skip it."
        echo "Note: WP4 requires Kaolin and will fail at import time without it."
        exit 1
    fi
}

install_diso() {
    if [[ "$INSTALL_DISO" != "1" ]]; then
        echo "Skipping DISO install (INSTALL_DISO=$INSTALL_DISO)."
        return 0
    fi

    echo "Installing DISO..."
    "$VENV_PYTHON" -m pip install diso --no-build-isolation
}

build_pydevelopability_cpp() {
    if [[ "$BUILD_PYDEVELOPABILITY" != "1" ]]; then
        echo "Skipping pydevelopability C++ build (BUILD_PYDEVELOPABILITY=$BUILD_PYDEVELOPABILITY)."
        return 0
    fi

    require_cmd cmake
    require_cmd make

    local pydev_root
    pydev_root="$PROJECT_ROOT/subprojects/pydevelopability"
    local pydev_build
    pydev_build="$pydev_root/build_manual"
    local pybind11_dir
    local cc_root
    local dev_repo
    local libigl_include

    cc_root="${CC_ROOT:-$PROJECT_ROOT/subprojects/concentrate_curvature}"
    dev_repo="${DEV_REPO:-$cc_root/DevelopabilityOfTriangleMeshes}"
    libigl_include="${LIBIGL_INCLUDE:-$cc_root/libigl-modern/include}"

    pybind11_dir="$($VENV_PYTHON -m pybind11 --cmakedir 2>/dev/null || true)"
    if [[ -z "$pybind11_dir" || ! -d "$pybind11_dir" ]]; then
        echo "Warning: pybind11 CMake config directory was not found in this venv."
        echo "         Install pybind11 into $ENV_NAME to build pydevelopability."
        echo "         Example: $VENV_PYTHON -m pip install pybind11"
        if [[ "$PYDEVELOPABILITY_STRICT" == "1" ]]; then
            exit 1
        fi
        echo "Skipping pydevelopability C++ build (set PYDEVELOPABILITY_STRICT=1 to fail instead)."
        return 0
    fi

    if [[ ! -f "$dev_repo/include/developableflow/fixed_topology_core.cpp" ]]; then
        echo "Warning: required developability source not found: $dev_repo/include/developableflow/fixed_topology_core.cpp"
        echo "         Set DEV_REPO (or CC_ROOT) to your concentrate_curvature checkout."
        if [[ "$PYDEVELOPABILITY_STRICT" == "1" ]]; then
            exit 1
        fi
        echo "Skipping pydevelopability C++ build (set PYDEVELOPABILITY_STRICT=1 to fail instead)."
        return 0
    fi

    if [[ ! -f "$libigl_include/igl/is_border_vertex.h" ]]; then
        echo "Warning: required libigl header not found: $libigl_include/igl/is_border_vertex.h"
        echo "         Set LIBIGL_INCLUDE (or CC_ROOT) to your libigl include directory."
        if [[ "$PYDEVELOPABILITY_STRICT" == "1" ]]; then
            exit 1
        fi
        echo "Skipping pydevelopability C++ build (set PYDEVELOPABILITY_STRICT=1 to fail instead)."
        return 0
    fi

    echo "Building pydevelopability C++ extension..."
    # Reset stale CMake cache to avoid source-dir mismatch after moving/copying repos.
    rm -rf "$pydev_build"
    mkdir -p "$pydev_build"
    pushd "$pydev_build" >/dev/null
    if ! cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -Dpybind11_DIR="$pybind11_dir" \
        -DCC_ROOT="$cc_root" \
        -DDEV_REPO="$dev_repo" \
        -DLIBIGL_INCLUDE="$libigl_include"
    then
        popd >/dev/null
        if [[ "$PYDEVELOPABILITY_STRICT" == "1" ]]; then
            echo "Error: pydevelopability CMake configure failed in strict mode."
            exit 1
        fi
        echo "Warning: pydevelopability CMake configure failed; continuing without C++ extension."
        return 0
    fi

    if ! make -j"$(nproc)"; then
        popd >/dev/null
        if [[ "$PYDEVELOPABILITY_STRICT" == "1" ]]; then
            echo "Error: pydevelopability build failed in strict mode."
            exit 1
        fi
        echo "Warning: pydevelopability build failed; continuing without C++ extension."
        return 0
    fi

    local built_so
    built_so="$(find . -name '_devcore*.so' | head -n1 || true)"
    if [[ -z "$built_so" ]]; then
        popd >/dev/null
        if [[ "$PYDEVELOPABILITY_STRICT" == "1" ]]; then
            echo "Error: pydevelopability build finished but no _devcore*.so artifact was produced."
            exit 1
        fi
        echo "Warning: pydevelopability did not produce _devcore*.so; continuing without C++ extension."
        return 0
    fi

    cp "$built_so" "$pydev_root/pydevelopability/"
    popd >/dev/null

    "$VENV_PYTHON" -m pip install -e .
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
install_python_build_deps_ubuntu
check_python
check_gpu_5090
detect_cuda_home
create_or_reuse_venv
install_dependencies
configure_optional_steps
install_kaolin
install_diso
build_pydevelopability_cpp
verify_torch_cuda_5090

echo ""
echo "Setup complete (RTX 5090 validated)."
echo "Activate with: source $ENV_NAME/bin/activate"
