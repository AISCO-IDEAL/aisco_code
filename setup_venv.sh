# eschaerer
# This is the automated setup script to create the Python virtual environment for AISCO-code, called aisco_env
# It is intended to be run on the Euler cluster at ETH Zurich.

#!/bin/bash
# setup_ven~/setup_venv.shv.sh – Setup Python project with venv + pyproject.toml on Euler cluster
set -e  # Exit on any error
ENV_NAME="aisco_env"

# 1. Load the required Python module (adjust as needed for your cluster)
echo "🔍 Loading Python module..."
module load fast_python_workshop_cpu/2025.0.0

# 2. Check if Python is available after loading the module
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
if [[ "$PYTHON_VERSION" != "3.11.6" ]]; then
    echo "❌ Python 3.11.6 not found. Current version: $PYTHON_VERSION"
    echo "Please load the correct Python module or check your environment."
    exit 1
fi

# 3. Create venv environment
echo "🐍 Creating venv environment '$ENV_NAME'..."
python3 -m venv "$ENV_NAME"

# 4. Activate environment
echo "🔄 Activating environment..."
source "$ENV_NAME/bin/activate"

# 5. Upgrade pip and install standard dependencies in editable mode
echo "📦 Upgrading pip and installing standard dependencies from pyproject.toml..."
python3 -m pip install --upgrade pip
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
echo ""
##############################################################
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: source $ENV_NAME/bin/activate"
echo "2. Configure VS Code (install Ruff/MyPy extensions)."
echo "3. Start coding! 🚀"
