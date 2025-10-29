# Project Setup Guide for Collaborators of the Aisco-Engineeringtool
Welcome! This repository supports collaborative development in Python using a fully reproducible environment with **Conda** (via [Miniforge](https://github.com/conda-forge/miniforge)). This guide explains how to:

- Set up the project in your own code editor (e.g. VS Code)
- Install and activate Miniforge (if conda is not installed locally)
- Use the provided `environment.yml` to recreate the project environment
- Work with Conda or optionally use Python `venv`
- Add and share new packages

## 1. Colne the Repository

```bash
git clone https://github.com/acentauri-solar-racing/aisco-ET.git
cd aisco-ET
```
\texttit{Clone the project and move it into your personal working directory (You need to have access to the aCe GitHub).}

## 2. Install Miniforge (if Conda is not installed locally)
Install [Miniforge](https://github.com/conda-forge/miniforge) – a lightweight conda installer using the conda-forge channel.

### macOS (Intel CPU)
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
bash Miniforge3-MacOSX-x86_64.sh
```
### macOS (Apple Silicon - M1/M2/M3)
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```
### Linux (x86_64)
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```
### Linux (aarch64 / Raspberry Pi etc.)
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh
```
### Windows
1. Download the installer form [Miniforge releases](https://github.com/conda-forge/miniforge/releases)
2. Run the .exe and follow the setup wizard

## 3. Enable Conda in your Terminal
### macOS/Linux
```bash
source ~/miniforge3/bin/activate
```
### Windows (PowerShell)
```powershell
& "C:\Users\<USERNAME>\miniforge3\Scripts\activate"
```
> This enables conda commands in your terminal

## 4. Activate the Project Environment 
The repository includes a predefined Conda environment in `environment.yml`.
### Create and activate it (first time):
```bash
conda env create -f environment.yml
conda activate project-env
```
### Reactivate later:
```bash
conda activate project-env
```
> Ensures you’re using the correct environment with all dependencies installed.

## 5. Use the Conda Environment in VS Code
1. Open the project folder (aisco-ET) in VS Code
2. Press `Ctrl+Shift+P` \Rightarrow search `Python: Select Interpreter`
3. Choose the Python interpreter form `project-env`
> VS Code will now run your scripts using this environment

## 6. Add Packages to the Conda Environment
Install a new package (e.g. `matplotlib`):
```bash
conda install matplotlib
```
Or use pip inside the conda environment:
```bash
pip install somepackage
```
Then update the envionment file and share:
```bash
conda env export --no-builds > environment.yml
git add environment.yml
git commit -m "Update environment.yml"
git push
```
> Keeps everyone on the same version of dependencies.

## 7. (Optional) Use a Python .venv Instead
You may optionally use a `.venv` virtual environment.
### Create the virtual environment:
```bash
python3 -m venv .venv
```
### Activate it (first time):
```bash
pip install -r requirements.txt
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\Activate.ps1       # Windows PowerShell
```
### Reactivate later:
```bash
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\Activate.ps1       # Windows PowerShell
```
### Install packages:
```bash
pip install jupyter cadquery matplotlib
```
### Save installed packages (first time):
```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Add requirements.txt"
git push
```

For questions or help, please open an issue or contact the maintainers. Happy coding!
































