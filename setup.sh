#!/bin/bash
# setup.sh – Setup Python project with Conda + pyproject.toml
set -e  # Exit on any error

ENV_NAME="aisco-code"
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

# 5. Print instructions for optional dependencies
echo ""
echo "✅ Standard dependencies installed."
echo ""
echo "📌 Optional dependencies can be installed manually:"
echo "   - Workpackage 1 (WP1_WP1_CFD_AI):                  pip install '.[WP1_CFD_AI]'"
echo "   - Workpackage 2 (WP2_Assembler):                   pip install '.[WP2_Assembler]'"
echo "   - Workpackage 3 (WP_3_Solarestimator):             pip install '.[WP3_Solarestimator]'"
echo "   - Workpackage 4 (WP_4_Aeroshell_optimizer):        pip install '.[WP4_Aeroshell_Optimizer]'"
echo "   - All optional dependencies:                       pip install '.[all]'"
echo ""
echo "🪝 Set up Pre-commit Hooks (if Dev Tools are installed):"
echo "   pre-commit install"
echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate $ENV_NAME"
echo "2. Configure VS Code (install Ruff/MyPy extensions)."
echo "3. Start coding! 🚀"


# #!/bin/bash
# # Setup script for Python Ruff Conda Template
# # Run this script to set up the development environment

# set -e  # Exit on any error

# echo "🚀 Setting up Python Ruff Conda Template..."

# # Check if conda is available
# if ! command -v conda &> /dev/null; then
#     echo "❌ Conda not found. Please install miniforge first:"
#     echo "   https://github.com/conda-forge/miniforge"
#     exit 1
# fi

# echo "✅ Conda found"

# echo "🫙 Creating conda environment..."
# conda env create -f environment.yml

# echo "🔄 Activating environment..."
# eval "$(conda shell.bash hook)"
# conda activate python-ruff-template

# echo "📦 Installing dev tools..."
# pip install .[dev]

# echo "🪝 Installing pre-commit hooks..."
# pre-commit install

# # echo "🧪 Testing the setup..."
# # python src/example.py

# echo ""
# echo "🎉 Setup complete!"
# echo ""
# echo "Next steps:"
# echo "1. Activate the environment: conda activate python-ruff-template"
# echo "2. Configure VS Code:"
# echo "   - Open project in VS Code (will prompt for extensions)"
# echo "   - Install Ruff and MyPy extensions: charliermarsh.ruff and ms-python.mypy-type-checker"
# echo "   - Copy settings: .vscode/settings_template.json at bottom of existing .vscode/settings.json"
# echo "3. Start coding! 🚀"
# echo ""
# echo "Useful commands:"
# echo "  ruff check .          # Check for issues"
# echo "  ruff check --fix .    # Fix auto-fixable issues"
# echo "  ruff format .         # Format code"
# echo "  mypy .                # Type checking"
# echo "  pre-commit run --all-files  # Run all pre-commit hooks"
