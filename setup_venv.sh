# eschaerer
# This is the automated setup script to create the Python virtual environment for AISCO-code, called aisco_env
# It is intended to be run on the Euler cluster at ETH Zurich.

#!/bin/bash
# setup_ven~/setup_venv.shv.sh – Setup Python project with venv + pyproject.toml on Euler cluster
set -e  # Exit on any error
ENV_NAME="aisco_env"

# 1. Load the required Python module (adjust as needed for your cluster)
echo "🔍 Loading Python module..."
module purge
module load fast_python_workshop_cpu/2025.0.0
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6  eth_proxy
# 2. Check if Python is available after loading the module
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
if [[ "$PYTHON_VERSION" != "3.11.6" ]]; then
    echo "❌ Python 3.11.6 not found. Current version: $PYTHON_VERSION"
    echo "Please load the correct Python module or check your environment."
    exit 1
fi

# avoid PEP 660 editable backend problems if your toolchain is older
export SETUPTOOLS_ENABLE_FEATURES="legacy-editable"

# avoid NVML architecture detection problems on non-GPU nodes
export TORCH_CUDA_ARCH_LIST="8.9;8.6;8.0;7.5"

echo "CUDA_HOME=$CUDA_HOME"
which nvcc || true

# 3. Create venv environment
echo "🐍 Creating venv environment '$ENV_NAME'..."
python3 -m venv "$ENV_NAME"

# 4. Activate environment
echo "🔄 Activating environment..."
source "$ENV_NAME/bin/activate"

# 4.1 [REQUIREMENT FOR WP4 NEURAL RENDERER] Attempt to autodetect CUDA toolkit and set CUDA_HOME so builds
# that rely on CUDA (e.g., PyTorch extensions) can find nvcc, headers
# and libraries. This helps users who load a CUDA module or have nvcc
# on PATH but haven't set CUDA_HOME themselves.
echo "🔧 Detecting CUDA toolkit (nvcc) and setting CUDA_HOME if present..."
if command -v nvcc >/dev/null 2>&1; then
    NVCC_PATH=$(readlink -f "$(command -v nvcc)")
    CUDA_HOME_DETECTED=$(dirname "$(dirname "$NVCC_PATH")")
    export CUDA_HOME="$CUDA_HOME_DETECTED"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    echo "✅ Detected nvcc at $NVCC_PATH; set CUDA_HOME=$CUDA_HOME"
else
    # try common locations used on clusters
    CUDA_CANDIDATES=("/usr/local/cuda" "/usr/local/cuda-12.1" "/opt/cuda" "/opt/rocm" "/cluster/software/cuda" "/cluster/software/stacks/2024-06/spack/opt/spack/*/cuda-*" )
    FOUND_CUDA=""
    for d in "${CUDA_CANDIDATES[@]}"; do
        # expand globs
        for p in $d; do
            if [ -x "$p/bin/nvcc" ]; then
                CUDA_HOME="$p"
                export CUDA_HOME
                export PATH="$CUDA_HOME/bin:$PATH"
                export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
                echo "✅ Found nvcc at $CUDA_HOME/bin/nvcc; set CUDA_HOME=$CUDA_HOME"
                FOUND_CUDA=1
                break 2
            fi
        done
    done
    if [ -z "$FOUND_CUDA" ]; then
        echo "⚠️  nvcc (CUDA toolkit) not found on PATH and no common CUDA location found."
        echo "   If you will build CUDA extensions, please load your CUDA module (e.g. 'module load cuda/...')"
        echo "   or set CUDA_HOME to the toolkit root before running builds."
    fi
fi

# 5. Upgrade pip and install standard dependencies in editable mode
echo "📦 Upgrading pip and installing standard dependencies from pyproject.toml..."
python3 -m pip install --upgrade pip setuptools wheel build
python3 -m pip install -e .

echo "🪝 Setting up Pre-commit Hooks..."
pre-commit install

# 6. Print instructions for optional dependencies
echo ""
echo "✅ Standard dependencies installed."
echo ""
echo "📌 Optional dependencies can be installed manually:"
echo "   - Workpackage 1 (WP1_WP1_CFD_AI):                  python3 -m pip install '.[wp1_cfd_ai]'"
echo "   - Workpackage 2 (wp2_assembler):                   python3 -m pip install '.[wp2_assembler]'"
echo "   - Workpackage 3 (WP_3_Solarestimator):             python3 -m pip install '.[WP3_Solarestimator]'"
echo "   - Workpackage 4 (WP_4_Aeroshell_optimizer):        python3 -m pip install '.[wp4_aeroshell_optimizer]'"
echo "   - testing dependencies:                            python3 -m pip install '.[testing]'"
echo "   - All optional dependencies:                       python3 -m pip install '.[all]'"
echo ""
############# eschaerer addition for kaolin install #############
echo "🔍 Checking if Kaolin should be installed..."
read -p "Do you want to install Kaolin (GPU/CUDA-dependent)? [y/n]: " install_kaolin
if [[ "$install_kaolin" == "y" ]]; then
    echo "🐍 Installing Kaolin from NVIDIA link..."
    python3 -m pip install -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html kaolin==0.18.0
    echo "✅ Kaolin installation complete."
else
    echo "⏭️ Skipping Kaolin installation."
fi


read -p "Do you want to BUILD Neural Renderer (DON'T DO THIS UNLESS SOMEONE SCREWED UP)? [y/n]: " install_renderer
if [[ "$install_renderer" == "y" ]]; then
    cd /cluster/home/aalthaus/AISCO_code/fixed_renderer
    # optionally set arch if building on a login node:
    export TORCH_CUDA_ARCH_LIST="8.9;8.6;8.0;7.5"
    python setup.py build_ext --inplace
else
    echo "⏭️ Skipping Renderer installation."
fi




echo ""
##############################################################
echo ""
echo "✅ Setup complete!"
echo ""
echo ""
echo ""
echo "Next steps:"
echo "1. Activate the environment: source $ENV_NAME/bin/activate"
echo "2. Start coding! 🚀"
