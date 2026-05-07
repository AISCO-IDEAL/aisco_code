#!/usr/bin/env bash

set -e # Exit immediately if a command exits with a non-zero status

echo " 🤖 Welcome to the AISCO setup script!"
echo "📌 make sure you have deactivated your old env & that you are on a RTX6000"
rm -rf aisco_env
echo "🧹 Old aisco_env has been deleted. Continuing with new installation..."
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.8.0 python_cuda/3.11.6 eth_proxy
export TORCH_CUDA_ARCH_LIST="12.0"
module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy
module load cuda/12.8.0
export TORCH_CUDA_ARCH_LIST="12.0"
echo "CUDA_HOME=$CUDA_HOME"
which nvcc || true
hash -r
nvcc --version
module list
python3 -m venv aisco_env
source aisco_env/bin/activate
CURRENT_PY=$(which python)
echo "🔄Installed and activated environment: $CURRENT_PY. Continuing..." 
python -m pip install --upgrade pip setuptools wheel build ninja pybind11
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchvision==0.23.0+cu128
python -m pip install -e '.[all]' --no-build-isolation -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
echo "Installed PyTorch and related packages. Continuing with Kaolin installation..."
python -m pip install -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html kaolin==0.18.0
export TORCH_CUDA_ARCH_LIST="12.0"
echo "🔄 Installed Kaolin. Continuing with DISO installation..."
python -m pip install diso --no-build-isolation
echo "🔄 Installed Diso. Continuing with torch-cluster installation..."
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.8.0 python_cuda/3.11.6 eth_proxy
source /cluster/home/$USER/AISCO_code/aisco_env/bin/activate
export TORCH_CUDA_ARCH_LIST="12.0"
python -m pip uninstall -y torch-cluster
python -m pip install --no-cache-dir torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

echo "🔄 Installed torch-cluster. Continuing with pydevelopability C++ extension build..."
cd subprojects/pydevelopability

# Create build directory if it doesn't exist
if [ ! -d build_manual ]; then
    mkdir build_manual
fi

cd build_manual
# Use the global Python and build tools (no local pyproject needed)
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Copy .so to package
cp _devcore.cpython-311-x86_64-linux-gnu.so ../pydevelopability/

cd ../../..
python3 -m pip install -e .
echo "🔄 pydevelopability build complete. Continuing with fixed_renderer build..."

cd /cluster/home/$USER/AISCO_code/subprojects/fixed_renderer
# optionally set arch if building on a login node:
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
python setup.py build_ext --inplace

echo "🔄 Fixed_renderer build complete."
echo "🎉 Setup script finished. Lets aCe the race!"
