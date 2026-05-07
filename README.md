#kdiehl, #eschaerer
# Welcome to AISCO_code
You only find technical infos to get you coding here. All conceptual information can be found in `01 Knowledge Base/00 IT Tutorials / AISCO_code` on Obsidian.

Lets clone the AISCO_code Repo via SSH:
`git clone git@github.com:IDEALLab/AISCO_code.git`

## VS Code Plugins
- install `python`
- install `Git graph`
	- click on `Git Graph` at the bottom left to open the plugin
- install `ruff`
- install `mypy type checker`
- install 'PLY Viewer'

## Important VS Code settings
- open settings `ctrl + ,` -> enter `git blame` -> enable
	- shows who edited what line of code => no tags necessary
- open settings `ctrl + ,` -> enter `format on save` -> enable
	- activates ruff, to make your code look nice and work well
	- reduces ruff errors later on

## hide unnecessary folders
- open VS Code
- `strg + , ` to open settings
- search for `hide folders`
- click on `add pattern`
- add folder name starting with `**/`

## Install entire aisco_env
log in into Login Node
start srun job
`srun --cpus-per-task=1 --mem-per-cpu=8G --time=4:00:00 --gpus=pro_6000:1 --pty bash`
Move to the AISCO_code folder
`cd ~/AISCO_code`
run
`chmod +x setup_euler_new.sh`
`~/AISCO_code/setup_euler_new.sh`
Wait for approximately 15 mins until the setup finishes, and then cancel your srun job
-> congrats, you are all set!

## Legacy Tutorial to install Setup whitout Blackwell-Architecture
Temporarily change in the pyproject.toml the following dependencies to the following older versions:
torch: 2.4.0,
torchvision: 0.19.0,

Install entire aisco_env
on Matthews PC use setup_wsl.sh
any outdated setup files can be found at commit 64f74931
6. log in into Login Node
7. start srun job

`srun --cpus-per-task=1 --mem-per-cpu=8G --time=4:00:00 --gpus=1 --pty bash`
Move to the AISCO_code folder
cd ~/AISCO_code
run
`chmod +x ~/AISCO_code/setup_euler_old.sh`
execute `~/AISCO_code/setup_euler_old.sh`
make sure to install both, Kaolin and DISO when prompted
after the script has finished run
`module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy`
`source aisco_env/bin/activate`
`python3 -m pip install -e '.[all]' --no-build-isolation`
To fix a the setup problem in WP1 also run this at the end:
`module purge`  
`module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy`  
`source /cluster/home/$USER/AISCO_code/aisco_env/bin/activate`  
`python -m pip uninstall -y torch-cluster`  
`python -m pip install --no-cache-dir torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html`
-> congrats, you are all set!

Add .so Files Manually
On fresh install of aisco_env on Euler, you must add four .so files manually. This is due to incompatible GPUs of Euler and AISCO-BEAST.
a) in /cluster/home/kdiehl/AISCO_code/subprojects/pydevelopability/pydevelopability add _devcore.cpython-311-x86_64-linux-gnu.so
b) in /cluster/home/kdiehl/AISCO_code/subprojects/fixed_renderer/cuda add create_texture_image.cpython-311-x86_64-linux-gnu.so, load_textures.cpython-311-x86_64-linux-gnu.so and rasterize.cpython-311-x86_64-linux-gnu.so
s
all these file are stored in cluster/work/fuge/AISCO


## Environment Activation Guide (do this every time you open a new terminal)
for AISCO, please activate myenv in the `cd ~/AISCO_code/` folder:
`source aisco_env/bin/activate`
