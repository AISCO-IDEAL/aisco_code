#!/bin/bash
# Setup script for AISCO-code (Linux / HPC)

echo "🚀 Setting up AISCO-code..."

# ---- Check if conda is available ----
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniforge or Mambaforge first:"
    echo "    https://github.com/conda-forge/miniforge"
    exit 1
fi
echo "✅ Conda found"

# ---- Create (or recreate) environment ----
ENV_NAME="AISCO-code"

if conda env list | grep -q "$ENV_NAME"; then
    echo "🔁 Environment '$ENV_NAME' already exists. Reusing it..."
else
    echo "📦 Creating conda environment '$ENV_NAME'..."
    conda create -y -n "$ENV_NAME" python=3.10
fi

# ---- Activate environment ----
echo "🔄 Activating environment..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---- Install requirements ----
REQ_FILE="AISCO-code/subprojects/WP4-Aeroshell/SDF_Prototype/requirements.txt"

if [[ -f "$REQ_FILE" ]]; then
    echo "📜 Installing requirements from $REQ_FILE..."
    pip install --upgrade pip
    pip install -r "$REQ_FILE"
else
    echo "⚠️ Requirements file not found at $REQ_FILE. Skipping pip install."
fi

# ---- Install pre-commit hooks (optional) ----
if command -v pre-commit &> /dev/null; then
    echo "🪝 Installing pre-commit hooks..."
    pre-commit install
else
    echo "ℹ️ pre-commit not found. Skipping hook installation."
fi

# ---- Test setup ----
if [[ -f "src/example.py" ]]; then
    echo "🧪 Testing setup..."
    python src/example.py
fi

echo
echo "🎉 Setup complete!"
echo
echo "Next steps:"
echo "1. Activate the environment: conda activate $ENV_NAME"
echo "2. Start coding! 🚀"
echo
echo "Useful commands:"
echo "  ruff check .           # Check for issues"
echo "  ruff check --fix .     # Fix auto-fixable issues"
echo "  ruff format .          # Format code"
echo "  mypy .                 # Type checking"
echo "  pre-commit run --all-files  # Run all pre-commit hooks"
