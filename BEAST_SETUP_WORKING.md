# RTX 5090 Beast Setup: What Finally Worked

This is the exact sequence that got AISCO_code running on my Linux Beast machine (RTX 5090).

## 1) Correct way to run setup script

From project root:

```bash
cd /home/clhuber/AISCO_code
PYTHON_BIN="$(which python)" ./setup_beast.sh
```

Important:
- Use `./setup_beast.sh` (not `setup_beast.sh` without `./`).
- Keep a space between env var assignment and script call.

## 2) First blocker: Python missing _ctypes

Symptom:
- setup stops with: `Python is missing _ctypes`

Reason:
- Python was compiled without ctypes support. Installing `libffi-dev` alone is not enough if Python is already built.

### Fix used

Install build dependencies:

```bash
sudo apt update
sudo apt install -y \
  libffi-dev build-essential zlib1g-dev libssl-dev libbz2-dev \
  libreadline-dev libsqlite3-dev liblzma-dev tk-dev uuid-dev
```

Install pyenv and rebuild Python 3.11.6:

```bash
curl https://pyenv.run | bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

pyenv install 3.11.6 -f
pyenv shell 3.11.6
python -c "import _ctypes; print('ctypes ok')"
```

Then rerun setup with the rebuilt Python:

```bash
cd /home/clhuber/AISCO_code
PYTHON_BIN="$(which python)" ./setup_beast.sh
```

## 3) Second blocker: CUDA toolkit not found (nvcc)

Symptom:
- setup stops with: `Error: CUDA toolkit (nvcc) not found`

Reason:
- NVIDIA driver was installed, but CUDA toolkit (compiler) was missing.

### Fix used

Add NVIDIA CUDA repo and install CUDA 12.8 toolkit:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

Verify:

```bash
/usr/local/cuda-12.8/bin/nvcc --version
```

Rerun setup (skip repeated apt deps step):

```bash
cd /home/clhuber/AISCO_code
AUTO_INSTALL_SYSTEM_DEPS=0 PYTHON_BIN="$(which python)" ./setup_beast.sh
```

## 4) Third blocker: broken existing venv

Symptom:
- setup stops with: `venv python not found at .../aisco_env/bin/python3`

Reason:
- Existing `aisco_env` folder was partially created/corrupted.

### Fix used

Delete and recreate environment via setup script:

```bash
cd /home/clhuber/AISCO_code
rm -rf aisco_env
AUTO_INSTALL_SYSTEM_DEPS=0 PYTHON_BIN="$(which python)" ./setup_beast.sh
```

## 5) Installing extras (wp1 / all)

If `python3 -m pip install '.[all]'` fails on `torch-cluster` build:
- install torch first in active venv
- then install extras with no build isolation

```bash
cd /home/clhuber/AISCO_code
source aisco_env/bin/activate

python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128
python -m pip install --no-build-isolation -e ".[all]"
```

Note:
- `torch-cluster` can fail when pip cannot find a matching wheel and falls back to source build.

## 6) Running WP1 locally

Command:

```bash
bash ~/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/slurm_scripts/wp1_main_local.sh
```

That script expects:
- project at `~/AISCO_code`
- venv at `~/AISCO_code/aisco_env`

## Quick sanity checks

```bash
cd /home/clhuber/AISCO_code
source aisco_env/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

If all of the above works, the setup is in a good state.
