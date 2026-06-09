#!/usr/bin/env bash

set -e # Exit immediately if a command exits with a non-zero status
ENV_NAME="aisco_env"
GPU_PROFILE="${GPU_PROFILE:-blackwell}"

case "${GPU_PROFILE,,}" in
    blackwell)
        CUDA_MODULE="cuda/12.8.0"
        TORCH_VERSION="2.8.0"
        TORCHVISION_VERSION="0.23.0"
        TORCH_CUDA_TAG="cu128"
        TORCH_CUDA_ARCH_LIST_VALUE="12.0"
        EXPECTED_CUDA_HOME_MATCH="cuda-12.8"
        ;;
    ada4090|ada)
        CUDA_MODULE="cuda/12.1.1"
        TORCH_VERSION="2.4.1"
        TORCHVISION_VERSION="0.19.1"
        TORCH_CUDA_TAG="cu121"
        TORCH_CUDA_ARCH_LIST_VALUE="8.9"
        EXPECTED_CUDA_HOME_MATCH="cuda-12.1"
        ;;
    *)
        echo "❌ Unsupported GPU_PROFILE '$GPU_PROFILE'. Use 'ada4090' or 'blackwell'."
        exit 1
        ;;
esac

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
echo "🚀 Welcome to the AISCO setup script!"
echo "ℹ️ make sure you have deactivated your old env & that you are on a RTX6000"
echo "ℹ️ Project Root: $PROJECT_ROOT"

if [ -d "$ENV_NAME" ]; then
    echo -e "\n\033[1;33m--- Environment Update Check ---\033[0m"
    echo "An existing '$ENV_NAME' was found."
    echo "🚨 CRITICAL: Choose 'y' if you changed Torch versions or CUDA modules!"
    echo "💡 ADVICE:"
    echo " - Select 'y' (DELETE) if:"
    echo "   * You changed CUDA versions (e.g., 12.1 -> 12.8)."
    echo "   * You changed Torch or Torchvision versions in pyproject.toml."
    echo "   * You modified 'dependencies' or 'optional-dependencies'."
    echo "   * You are seeing 'libcublas.so' or 'GLIBC' errors."
    echo " - Select 'n' (KEEP)   if:"
    echo "   * You only edited .py files or .cpp files in your subprojects."
    echo "   * You are just re-running the script to ensure everything is linked."

    read -p "Delete old environment and start fresh? (y/n): " confirm
    if [ "$confirm" = "y" ]; then
        echo "🧹 Deleting old environment..."
        rm -rf "$ENV_NAME"
    else
        echo "♻️ Entering Update Mode (skipping fresh venv creation)..."
        source "$ENV_NAME/bin/activate"
    fi
else
    echo "✨ No environment found. Starting fresh install..."
fi

module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy
module load "$CUDA_MODULE"
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST_VALUE"
echo "Using GPU profile: $GPU_PROFILE"
echo "Torch stack: torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION ($TORCH_CUDA_TAG)"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "CUDA_HOME=$CUDA_HOME"
which nvcc || true
hash -r
nvcc --version
NVCC_PATH=$(readlink -f "$(command -v nvcc)")
module list
python3 -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"
CURRENT_PY=$(which python)
echo "🔄Installed and activated environment: $CURRENT_PY. Continuing..."
python -m pip install --upgrade pip setuptools wheel build ninja pybind11
python -m pip install --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_TAG" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
python -m pip install -e '.[wp1_cfd_ai,wp2_assembler,wp3_solar_estimator,wp4_aeroshell_optimizer,wp7_gui,testing]' --no-build-isolation -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${TORCH_CUDA_TAG}.html"
python -m pip install --upgrade jupyter-client ipykernel # install last to avoid downgrade conflicts
echo "Installed PyTorch and related packages. Continuing with Kaolin installation..."
python -m pip install -f "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${TORCH_VERSION}_${TORCH_CUDA_TAG}.html" kaolin==0.18.0
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST_VALUE"
echo "🔄 Installed Kaolin. Continuing with DISO installation..."
python -m pip install diso --no-build-isolation
echo "🔄 Installed Diso. Continuing with torch-cluster installation..."
module purge
module load stack/2024-06 gcc/12.2.0 "$CUDA_MODULE" python_cuda/3.11.6 eth_proxy
source "$PROJECT_ROOT/$ENV_NAME/bin/activate"
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST_VALUE"
python -m pip uninstall -y torch-cluster
python -m pip install --no-cache-dir torch-cluster -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${TORCH_CUDA_TAG}.html"

# echo "🔄 Installed torch-cluster. Continuing with pydevelopability C++ extension build..."
# cd subprojects/pydevelopability

# # Create build directory if it doesn't exist
# if [ ! -d build_manual ]; then
#     mkdir build_manual
# fi

# cd build_manual
# # Use the global Python and build tools (no local pyproject needed)
# cmake .. -DCMAKE_BUILD_TYPE=Release
# make -j4

# # Copy .so to package
# cp _devcore.cpython-311-x86_64-linux-gnu.so ../pydevelopability/

# cd ../../..
# python3 -m pip install -e .

echo "⏭️ Skipping pydevelopability build."
echo "🔄 Continuing with fixed_renderer build..."

cd "$PROJECT_ROOT/subprojects/fixed_renderer"
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST_VALUE"
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$NVCC_PATH")")")"

# Guard: Ensure we actually have the expected CUDA now
if [[ "$CUDA_HOME" != *"$EXPECTED_CUDA_HOME_MATCH"* ]]; then
    echo "❌ ERROR: Still detecting wrong CUDA version: $CUDA_HOME"
    echo "if you copy paste the .so files to fixed_renderer, it will work anyways. Your are done now"
    echo "Current nvcc is: $(which nvcc)"
    exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
python setup.py build_ext --inplace

echo "🔄 Fixed_renderer build complete."
echo "🎉 Setup script finished. Lets aCe the race!"
