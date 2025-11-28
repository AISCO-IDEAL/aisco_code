#### DON'T USE THIS FILE ANYMORE, USE SETUP_VENV.SH AS WE USE EULER HPC NOW
#kdiehl
#!/bin/bash
# setup.sh – Setup Python project with Conda + pyproject.toml
set -e  # Exit on any error

ENV_NAME="aisco_env"
PYTHON_VERSION="3.11.6"

# 1. Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniforge/Anaconda:"
    echo "   https://github.com/conda-forge/miniforge"
    exit 1
fi

# 2. Create Conda environment
echo "🫙 Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

# 3. Activate environment
echo "🔄 Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# 4. Install standard dependencies in editable mode
echo "📦 Installing standard dependencies from pyproject.toml..."
pip install -e .

echo "🪝 Setting up Pre-commit Hooks..."
pre-commit install

# 5. Print instructions for optional dependencies
echo ""
echo "✅ Standard dependencies installed."
echo ""
echo "📌 Optional dependencies can be installed manually:"
echo "   - Workpackage 1 (WP1_WP1_CFD_AI):                  pip install '.[wp1_cfd_ai]'"
echo "   - Workpackage 2 (wp2_assembler):                   pip install '.[wp2_assembler]'"
echo "   - Workpackage 3 (WP_3_Solarestimator):             pip install '.[WP3_Solarestimator]'"
echo "   - Workpackage 4 (WP_4_Aeroshell_optimizer):        pip install '.[wp4_aeroshell_optimizer]'"
echo "   - testing dependencies:                            pip install '.[testing]'"
echo "   - All optional dependencies:                       pip install '.[all]'"
############# eschaerer addition for kaolin install #############
echo "🔍 Checking if Kaolin should be installed..."
read -p "Do you want to install Kaolin (GPU/CUDA-dependent)? [y/n]: " install_kaolin
if [[ "$install_kaolin" == "y" ]]; then
    echo "🐍 Installing Kaolin from NVIDIA link..."
    pip install -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html kaolin==0.18.0
    echo "✅ Kaolin installation complete."
else
    echo "⏭️ Skipping Kaolin installation."
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
