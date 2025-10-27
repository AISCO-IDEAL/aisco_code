# eschaerer
# This is the automated setup script to create the Python virtual environment for AISCO-code, called aisco_env
# It is intended to be run on the Euler cluster at ETH Zurich.

#!/bin/bash
# setup_venv.sh – Setup Python project with venv + pyproject.toml on Euler cluster
set -e  # Exit on any error
ENV_NAME="aisco_env"

# 1. Load the required Python module (adjust as needed for your cluster)
echo "🔍 Loading Python module..."
module load fast_python_workshop_cpu/2025.0.0 

# 2. Check if Python is available after loading the module
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please load the correct Python module or check your environment."
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
pip install --upgrade pip
pip install -e .

# 6. Print instructions for optional dependencies
echo ""
echo "✅ Standard dependencies installed."
echo ""
echo "📌 Optional dependencies can be installed manually:"
echo "   - Workpackage 1 (WP1_WP1_CFD_AI):                  pip install '.[wp1_cfd_ai]'"
echo "   - Workpackage 2 (wp2_assembler):                   pip install '.[wp2_assembler]'"
echo "   - Workpackage 3 (WP_3_Solarestimator):             pip install '.[WP3_Solarestimator]'"
echo "   - Workpackage 4 (WP_4_Aeroshell_optimizer):        pip install '.[wp4_aeroshell_optimizer]'"
echo "   - All optional dependencies:                       pip install '.[all]'"
echo ""
echo "🪝 Set up Pre-commit Hooks (if Dev Tools are installed):"
echo "   pre-commit install"
echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: source $ENV_NAME/bin/activate"
echo "2. Configure VS Code (install Ruff/MyPy extensions)."
echo "3. Start coding! 🚀"