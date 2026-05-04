# Euler Setup for RTX PRO 6000 Blackwell

This is the Blackwell version of the old RTX 4090 Euler setup.

Important changes:

- use `cuda/12.8.0` instead of `cuda/12.1.1`
- use PyTorch `2.8.0+cu128` instead of `2.4.0`
- use `TORCH_CUDA_ARCH_LIST="12.0"` for Blackwell / `sm_120`
- use the CUDA 12.8 wheel links for Kaolin and `torch-cluster`

The current `setup_euler.sh` still contains CUDA 12.1 / PyTorch 2.4 assumptions, so do not run it unmodified for Blackwell.

## Setup

Log in into the login node.

Start an `srun` job on an RTX PRO 6000 Blackwell GPU:

```bash
srun --cpus-per-task=1 --mem-per-cpu=8G --time=4:00:00 --gpus=pro_6000:1 --pty bash
```

Move to the `AISCO_code` folder:

```bash
cd ~/AISCO_code
```

If you already have an old 4090/CUDA 12.1 environment, delete it:

```bash
rm -rf aisco_env
```

Load the Blackwell modules:

```bash
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.8.0 python_cuda/3.11.6 eth_proxy
export TORCH_CUDA_ARCH_LIST="12.0"
```

Make sure the loaded CUDA compiler is really CUDA 12.8:

```bash
which nvcc
nvcc --version
```

If this still points to `cuda-12.1.1`, DISO and other CUDA extensions will fail with `Unsupported gpu architecture 'compute_120'`.

On Euler, `python_cuda/3.11.6` may put CUDA 12.1 back on `PATH`. If `which nvcc` still shows `cuda-12.1.1`, try loading CUDA 12.8 last:

```bash
module purge
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy
module load cuda/12.8.0
export TORCH_CUDA_ARCH_LIST="12.0"
hash -r

which nvcc
nvcc --version
module list
```

If this still shows CUDA 12.1, check whether `python_cuda/3.11.6` is hard-wired to CUDA 12.1 and use a normal Python 3.11.6 module instead:

```bash
module --show_hidden spider python/3.11.6
module --show_hidden spider python_cuda/3.11.6
module --show_hidden spider cuda/12.8.0
```

The environment must use Python 3.11.6 plus CUDA 12.8. The Python module does not need to be named `python_cuda` as long as the venv can be created with Python 3.11.6 and `nvcc --version` reports CUDA 12.8.

Create and activate the environment:

```bash
python3 -m venv aisco_env
source aisco_env/bin/activate
python -m pip install --upgrade pip setuptools wheel build ninja
```

Install PyTorch for CUDA 12.8:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0+cu128 torchvision==0.23.0+cu128
```

Before installing all AISCO dependencies, temporarily update the torch pins in `pyproject.toml`:

```toml
"torch==2.8.0",
"torchvision==0.23.0",
```

Do not commit this temporary change unless the whole project is being migrated to CUDA 12.8.

Install AISCO with all extras:

```bash
python -m pip install -e '.[all]' --no-build-isolation \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

Install Kaolin:

```bash
python -m pip install -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html \
  kaolin==0.18.0
```

Install DISO:

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
python -m pip install diso --no-build-isolation
```

To fix the setup problem in WP1, also run this at the end:

```bash
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.8.0 python_cuda/3.11.6 eth_proxy
source /cluster/home/$USER/AISCO_code/aisco_env/bin/activate
export TORCH_CUDA_ARCH_LIST="12.0"

python -m pip uninstall -y torch-cluster
python -m pip install --no-cache-dir torch-cluster \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

Verify the environment:

```bash
python - <<'PY'
import torch

print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("arch_list", torch.cuda.get_arch_list())

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (12, 0)
assert "sm_120" in torch.cuda.get_arch_list()
print("Blackwell setup OK")
PY
```

Congrats, you are all set.

## Activation Later

Every time you open a new terminal:

```bash
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.8.0 python_cuda/3.11.6 eth_proxy
cd ~/AISCO_code
source aisco_env/bin/activate
export TORCH_CUDA_ARCH_LIST="12.0"
```
