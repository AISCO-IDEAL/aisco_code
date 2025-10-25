#kdiehl
@echo off
setlocal enabledelayedexpansion

set "ENV_NAME=aisco-code"
set "PYTHON_VERSION=3.11.6"

:: 1. Check if conda is installed
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Conda not found. Please install Miniforge/Anaconda:
    echo    https://github.com/conda-forge/miniforge
    exit /b 1
)

:: 2. Create Conda environment
echo 🫙 Creating Conda environment '%ENV_NAME%' with Python %PYTHON_VERSION%...
conda create -n "%ENV_NAME%" python="%PYTHON_VERSION%" -y

:: 3. Activate environment
echo 🔄 Activating environment...
call conda activate "%ENV_NAME%"

:: 4. Install standard dependencies in editable mode
echo 📦 Installing standard dependencies from pyproject.toml...
pip install -e .

:: 5. Print instructions for optional dependencies
echo.
echo ✅ Standard dependencies installed.
echo.
echo 📌 Optional dependencies can be installed manually:
echo    - Workpackage 1 (WP1_WP1_CFD_AI):                  pip install '.[wp1_cfd_ai]'
echo    - Workpackage 2 (wp2_assembler):                   pip install '.[wp2_assembler]'
echo    - Workpackage 3 (WP_3_Solarestimator):             pip install '.[WP3_Solarestimator]'
echo    - Workpackage 4 (WP_4_Aeroshell_optimizer):        pip install '.[wp4_aeroshell_optimizer]'
echo    - All optional dependencies:                       pip install '.[all]'
echo.
echo 🪝 Set up Pre-commit Hooks (if Dev Tools are installed):
echo    pre-commit install
echo.
echo ✅ Setup complete!
echo.
echo Next steps:
echo 1. Activate the environment: conda activate %ENV_NAME%
echo 2. Configure VS Code (install Ruff/MyPy extensions).
echo 3. Start coding! 🚀
