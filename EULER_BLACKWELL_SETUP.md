# Euler Setup for RTX PRO 6000 Blackwell

This guide describes how to set up `aisco_env` on Euler for an NVIDIA RTX PRO 6000 Blackwell GPU.

The important difference from the old RTX 4090 setup is that Blackwell uses CUDA compute capability `12.0` (`sm_120`). The old environment used CUDA 12.1, PyTorch 2.4.0, and extension builds targeting older architectures such as `8.9`. For Blackwell, use CUDA 12.8 and PyTorch wheels that include `sm_120` support.

## Summary of Required Changes

Minimum changes compared with the old 4090 flow:

- Use `cuda/12.8.0` instead of `cuda/12.1.1`.
- Use PyTorch `2.8.0+cu128` and torchvision `0.23.0+cu128`.
- Set `TORCH_CUDA_ARCH_LIST="12.0"` before building CUDA extensions.
- Use CUDA 12.8 matching wheel indexes for Kaolin and PyG packages such as `torch-cluster`.
- Avoid letting `pip install -e '.[all]'` reinstall the old `torch==2.4.0` pin from `pyproject.toml`.

## 1. Start an Interactive GPU Job

Log in to the Euler login node, then start an interactive job with one GPU:

```bash
srun --cpus-per-task=1 --mem-per-cpu=8G --time=4:00:00 --gpus=pro_6000:1 --pty bash
```

## 2. Move to the Repository

```bash
cd ~/AISCO_code
```

If you already have an old environment from the 4090 setup, remove it before rebuilding:

```bash
rm -rf aisco_env
```

## 3. Load the Blackwell-Compatible Modules

```bash
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.8.0 python_cuda/3.11.6 eth_proxy
```

Set the CUDA architecture for local CUDA extension builds:

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
```

## 4. Create and Activate the Virtual Environment

```bash
python3 -m venv aisco_env
source aisco_env/bin/activate
python -m pip install --upgrade pip setuptools wheel build ninja
```

## 5. Install PyTorch for CUDA 12.8

Install PyTorch and torchvision from the CUDA 12.8 PyTorch wheel index:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0+cu128 torchvision==0.23.0+cu128
```

Verify that PyTorch sees CUDA and the Blackwell GPU:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("arch list:", torch.cuda.get_arch_list())

x = torch.tensor([1.0], device="cuda")
print("cuda smoke test:", (x + 1).item())
PY
```

Expected values:

- `torch.version.cuda` should be `12.8`.
- GPU capability should be `(12, 0)`.
- `torch.cuda.get_arch_list()` should include `sm_120`.

## 6. Install the AISCO Package Without Downgrading PyTorch

The current `pyproject.toml` pins:

```toml
"torch==2.4.0",
"torchvision==0.19.0",
```

Those pins are for the old CUDA 12.1 setup and will conflict with the Blackwell environment. At minimum, do one of the following before installing all extras.

### Option A: Temporarily edit `pyproject.toml`

Change the pins to:

```toml
"torch==2.8.0",
"torchvision==0.23.0",
```

Then install the package:

```bash
python -m pip install -e '.[all]' --no-build-isolation
```

After setup, do not commit the temporary dependency-pin change unless the whole project is being migrated to CUDA 12.8.

### Option B: Install the project without dependencies first

This avoids reinstalling the old torch version from `pyproject.toml`:

```bash
python -m pip install --no-deps -e .
```

Then install the optional dependencies you need manually or by using a setup script that overrides the torch pins.

## 7. Install Kaolin

For the Blackwell/CUDA 12.8 PyTorch environment, use the Kaolin wheel index matching PyTorch 2.8.0 and CUDA 12.8:

```bash
python -m pip install -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html \
  kaolin==0.18.0
```

Verify:

```bash
python - <<'PY'
import kaolin
print("kaolin:", getattr(kaolin, "__version__", "unknown"))
PY
```

## 8. Install DISO

Build DISO with the Blackwell architecture exported:

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
python -m pip install diso --no-build-isolation
```

Verify:

```bash
python - <<'PY'
import diso
print("diso:", getattr(diso, "__file__", "unknown location"))
print("DiffMC available:", hasattr(diso, "DiffMC"))
print("DiffDMC available:", hasattr(diso, "DiffDMC"))
PY
```

## 9. WP1 `torch-cluster` Fix

The old WP1 fix used PyG wheels for `torch-2.4.0+cu121`. Replace it with the PyTorch 2.8.0 / CUDA 12.8 wheel index:

```bash
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.8.0 python_cuda/3.11.6 eth_proxy
source /cluster/home/$USER/AISCO_code/aisco_env/bin/activate
export TORCH_CUDA_ARCH_LIST="12.0"

python -m pip uninstall -y torch-cluster
python -m pip install --no-cache-dir torch-cluster \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

Verify:

```bash
python - <<'PY'
import torch
import torch_cluster

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("torch_cluster import OK")
PY
```

## 10. Optional: Build `fixed_renderer`

Only do this if the renderer extension needs to be rebuilt.

```bash
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.8.0 python_cuda/3.11.6 eth_proxy
source /cluster/home/$USER/AISCO_code/aisco_env/bin/activate
export TORCH_CUDA_ARCH_LIST="12.0"

cd /cluster/home/$USER/AISCO_code/subprojects/fixed_renderer
python setup.py build_ext --inplace
```

Return to the repository root afterwards:

```bash
cd /cluster/home/$USER/AISCO_code
```

## 11. Activation for Later Sessions

Every time you open a new Euler session for this environment:

```bash
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.8.0 python_cuda/3.11.6 eth_proxy
cd ~/AISCO_code
source aisco_env/bin/activate
export TORCH_CUDA_ARCH_LIST="12.0"
```

## 12. Final Smoke Test

Run this before starting longer jobs:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("arch list:", torch.cuda.get_arch_list())

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (12, 0)
assert "sm_120" in torch.cuda.get_arch_list()

x = torch.tensor([1.0], device="cuda")
assert (x + 1).item() == 2.0
print("Blackwell CUDA environment OK")
PY
```

If this passes, the environment is ready for Blackwell.

## Compact Command Sequence

Use this only after reading the notes above about the `pyproject.toml` torch pins.

```bash
srun --cpus-per-task=1 --mem-per-cpu=8G --time=4:00:00 --gpus=1 --pty bash

cd ~/AISCO_code
rm -rf aisco_env

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.8.0 python_cuda/3.11.6 eth_proxy
export TORCH_CUDA_ARCH_LIST="12.0"

python3 -m venv aisco_env
source aisco_env/bin/activate
python -m pip install --upgrade pip setuptools wheel build ninja

python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0+cu128 torchvision==0.23.0+cu128

# Make sure pyproject.toml no longer forces torch==2.4.0 / torchvision==0.19.0 before this.
python -m pip install -e '.[all]' --no-build-isolation

python -m pip install -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html \
  kaolin==0.18.0

python -m pip install diso --no-build-isolation

python -m pip uninstall -y torch-cluster
python -m pip install --no-cache-dir torch-cluster \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```
