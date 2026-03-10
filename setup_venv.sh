#!/bin/bash
# setup_venv.sh – Setup AISCO environment for WSL (Local) or Euler (HPC)
# Modified for Matthews PC: WSL compatibility, TCC mode support, and venv management.
# !!!not applicable for for local windows and linux pc. Only for HPC and WSL!!!

ENV_NAME="aisco_env"

echo "🔍 Detecting environment..."

if command -v module >/dev/null 2>&1; then
    echo "🏗️  Environment: Euler HPC detected."
    module purge
    module load fast_python_workshop_cpu/2025.0.0
    module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

else
    echo "💻 Environment: WSL / Local Windows PC detected."
    # In WSL, we must point to the Windows-provided GPU libraries
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"

    # Check if Python 3.11.6 is available locally
    if ! command -v python3.11 >/dev/null 2>&1; then
        echo "❌ Python 3.11.6 not found in WSL."
        echo "Please install it: sudo apt update && sudo apt install python3.11"
        exit 1
    fi
    alias python3='python3.11'
fi

# 2. Verify Python Version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Using Python version: $PYTHON_VERSION"

# 3. Create venv
if [ ! -d "$ENV_NAME" ]; then
    echo "🐍 Creating venv environment '$ENV_NAME'..."
    python3 -m venv "$ENV_NAME"
else
    echo "ℹ️  Environment '$ENV_NAME' already exists."
fi

# 4. Activate
source "$ENV_NAME/bin/activate"

# 5. WSL-specific GPU Fix: Ensure we use the correct Torch for TCC mode
echo "🛠️  Configuring CUDA Toolchain..."
if [[ -d "/usr/lib/wsl/lib" ]]; then
    echo "🔗 Linking WSL GPU libraries..."
    export CUDA_HOME="/usr/local/cuda" # Standard local path
else
    # Euler CUDA detection (from your original script)
    if command -v nvcc >/dev/null 2>&1; then
        export CUDA_HOME=$(dirname $(dirname $(readlink -f $(command -v nvcc))))
    fi
fi

# 6. Install Base Dependencies
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel
# Force install CUDA-enabled Torch 2.4.0
echo "Installing pyTorch 2.4.0 with CUDA support..."
python3 -m pip install torch==2.4.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu121

# 7. Install Monorepo in Editable Mode
echo "📦 Installing AISCO standard dependecies from pyproject.toml..."
export SETUPTOOLS_ENABLE_FEATURES="legacy-editable"
python3 -m pip install -e .

# 8. Pre-commit Hooks
echo "🪝 Setting up Pre-commit Hooks..."
pre-commit install
echo ""
echo "✅ Standard dependencies & Pre-commit Hooks installed."

# 9. Optional Dependencies

echo ""
echo "📌 Optional dependencies can be installed manually:"
echo "   - Workpackage 1 (WP1_WP1_CFD_AI):                  python3 -m pip install '.[wp1_cfd_ai]'"
echo "   - Workpackage 2 (wp2_assembler):                   python3 -m pip install '.[wp2_assembler]'"
echo "   - Workpackage 3 (WP_3_Solarestimator):             python3 -m pip install '.[WP3_Solarestimator]'"
echo "   - Workpackage 4 (WP_4_Aeroshell_optimizer):        python3 -m pip install '.[wp4_aeroshell_optimizer]'"
echo "   - Workpackage 7 (WP_7_gui):                        pip install '.[wp7_gui]'"
echo "   - testing dependencies:                            python3 -m pip install '.[testing]'"
echo "   - All optional dependencies:                       python3 -m pip install '.[all]'"
echo ""

# 9. Kaolin / Neural Renderer logic (Kept from your original)
echo "🔍 Optional GPU-dependent dependencies:"
read -p "Install Kaolin? [y/n]: " install_kaolin
if [[ "$install_kaolin" == "y" ]]; then
    python3 -m pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
    pip install --no-deps -e .
fi

read -p "Install DISO? [y/n]: " install_diso
if [[ "$install_diso" == "y" ]]; then
    python3 -m pip install diso --no-build-isolation --no-cache-dir

fi
python3 -m pip install -e . # for incorporating new deps.
echo "✅ Setup complete!"
echo "------------------------------------------------"
echo "Next steps:"
echo "1. Activate: source $ENV_NAME/bin/activate"
echo "2. WSL GPU users: Run 'nvidia-smi' to verify L4 connectivity."
echo "------------------------------------------------"
