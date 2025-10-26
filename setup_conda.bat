REM kdiehl
@echo off
REM Setup script for AISCO-code (Windows)
REM Setup script for AISCO-code (Windows)
REM Run this script to set up the development environment

echo 🚀 Setting up AISCO-code...
set "ENV_NAME=aisco-code"
set "PYTHON_VERSION=3.11.6"

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Conda not found. Please install miniforge first:
    echo    https://github.com/conda-forge/miniforge
    pause
    exit /b 1
)

echo ✅ Conda found

REM Create conda environment
REM echo 📦 Creating conda environment...
REM conda env create -f environment.yml #needs to change

echo 🫙 Creating Conda environment "%ENV_NAME%" with Python %PYTHON_VERSION%...
REM Try default channels first
call conda create -n "%ENV_NAME%" -c conda-forge python=%PYTHON_VERSION% -y

IF %ERRORLEVEL% NEQ 0 (
  echo Conda create failed; trying python=3.11...
  conda create -n "%ENV_NAME%" -c conda-forge python=3.11 -y
  IF %ERRORLEVEL% NEQ 0 exit /B %ERRORLEVEL%
)
REM Activate environment
echo 🔄 Activating environment...
call conda activate "%ENV_NAME%"

IF EXIST pyproject.toml (
  echo 📦 Installing standard dependencies from pyproject.toml...
  pip install -e .
  IF %ERRORLEVEL% NEQ 0 exit /B %ERRORLEVEL%
) ELSE (
  echo No pyproject.toml found — skipping pip install.
)

REM Install pre-commit hooks
where pre-commit >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
echo 🪝 Installing pre-commit hooks...
pre-commit install
) ELSE (
  echo pre-commit not found; skipping.
)

echo 📌 Optional dependencies can be installed manually:
echo    - Workpackage 1 (wp1_WP1_CFD_AI):                  pip install '.[wp1_CFD_AI]'
echo    - Workpackage 2 (wp2_Assembler):                   pip install '.[wp2_Assembler]'
echo    - Workpackage 3 (wp3_Solarestimator):             pip install '.[wp3_Solarestimator]'
echo    - Workpackage 4 (wp4_Aeroshell_optimizer):        pip install '.[wp4_Aeroshell_optimizer]'
echo    - All optional dependencies:                       pip install '.[all]'

echo ✅ Setup complete!
echo
echo Next steps:
echo 1. Activate the environment: conda activate %ENV_NAME%
echo 2. Configure VS Code (install Ruff/MyPy extensions).
echo 3. Start coding! 🚀
