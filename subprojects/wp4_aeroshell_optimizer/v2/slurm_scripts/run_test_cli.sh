#!/bin/bash

# Lightweight test CLI wrapper - skips heavy setup for quick testing

PROJECT_ROOT="$HOME/AISCO_code"
VENV_DIR="$PROJECT_ROOT/aisco_env"
VENV_PYTHON="$VENV_DIR/bin/python3"

# Check venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Python not found at $VENV_PYTHON"
    exit 1
fi

# Set minimal environment
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export AISCO_CODE_PATH="$PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Run with CLI arguments using full path to venv python (avoids source hang)
"$VENV_PYTHON" subprojects/wp4_aeroshell_optimizer/v2/src/main_files/main_v2.py "$@"