After you set up your account on the Beast, cloned AISCO_code onto it and got git running, you need to set up the env. 
This is not trivial, as the RTX5090 needs torch2.7 or newer, which cant be candeld by a RTX4090...
I tried to document my process of setting it up. Then I tried to boil it down to the essential Parts, so if you execute all of these comands in your terminal it should work. 
If it actually does or you needed another fix, please document this here.

You should start by editing the pyproject.toml and upgrade the following two lines:
  "torch>=2.4.0",
  "torchvision==0.23.0",
Do not commit these changes! Delete them after you did this setup.

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

Add NVIDIA CUDA repo and install CUDA 12.8 toolkit:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

Delete and recreate environment via setup script:

```bash
cd /home/clhuber/AISCO_code
rm -rf aisco_env
AUTO_INSTALL_SYSTEM_DEPS=0 PYTHON_BIN="$(which python)" ./setup_beast.sh
```


```bash
source aisco_env/bin/activate

python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128
python -m pip install --no-build-isolation -e ".[all]"
```
