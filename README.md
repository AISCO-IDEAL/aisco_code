# Setup and Usage Instructions for the AISCO_code Repository

## Setup
Before starting, please make sure that you have set up Euler correctly and know the Euler workflows.

* **1. Open a Terminal and log in to an Euler Login Node:**
ssh username@euler.ethz.ch

The Terminal then shows something like: `username@eu-login-XX:~$`


* **2. Clone the AISCO_code repository:**
git clone git@github.com:IDEALLab/AISCO_code.git

* **3. Install the Virtual Environment `aisco_env`:**
1. Request an SRUN-Job with an RTX Pro 6000:
srun --cpus-per-task=1 --mem-per-cpu=8G --time=4:00:00 --gpus=pro_6000:1 --pty bash

2. Move to the previously cloned AISCO_code folder:
cd ~/AISCO_code

3. Make the script executable:
chmod +x setup_euler_working.sh

4. Install the Virtual Environment:
~/AISCO_code/setup_euler_working.sh

---

## Usage
The only file needed to change is `~/AISCO_code/user_overrides.yaml`.

**Required parameter changes:**
Running the AISCO_code repository requires the following changes to the `user_overrides.yaml` file.

* Set the **enabled** flag on top of the `user_overrides.yaml` to **true**
* Add your wandb setting for: `wp1.setup.wandb_entity`, `wp1.setup.wandb_project`, `wp4.general_config.wandb_general.entity_name`, and `wp4.general_config.wandb_general.project_name`
* Add your storage path: `wp4.general_config.storage_paths.storage_dir`, e.g. `/cluster/work/fuge/AISCO/report_temp`

**Optional parameter changes:**
With the following keys you can tune the outcome of your GAOT model or the optimisation loop.

* `wp1.optimizer.args.epoch`: Epoch number for the GAOT model training
* `wp4.general_config.general_info.latent_space`: Activate or deactivate the training of a latent space (true/ false)
* `wp4.losses_v0_config.aero`: Decide whether to use the (set `gaot_loss` to 1) or deactivate (set `null_loss_aero` to 1) the aero loss, exactly one of the two losses needs to be set to 1, the other one to 0
* `wp4.losses_v0_config.solar`: Decide which version of the solar loss to use or deactivate (set `null_loss_solar` to 1). Exactly one of these five loss versions needs to be set to 1, the others need to be 0: `solar_s_r` (one single sun ray without ray tracing), `solar_s_r_r_t` (single sun ray with active ray tracing), `solar_m_r` (multiple sun rays without ray tracing), and `solar_m_r_r_t` (multiple sun rays with active ray tracing).
* `wp4.training_config.training_params.num_epochs`: Epoch number for the optimisation loop

You can mainly do two things:
* **Train a GAOT model** (requires permisson to access `cluster/work/fuge/AISCO/subprojects/wp1_cfd_ai`):
sbatch ~/AISCO_codesubprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/slurm_scripts/wp1_main.sh
(Optional parameter changes in the `user_overrides.yaml` file under WP1)

* **Run an optimisation loop:**
sbatch ~/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/main_testing.sh
(Optional parameter changes in the `user_overrides.yaml` file under WP4)

If you want to change other settings, feel free to have a look at the configs yourself. The current active configs are:

* GAOT Training: `~AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/config/main_config.yaml`.
* Optimisation loop: all configs under `~AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/configs`

Make sure to still activate the `user_overrides.yaml` for the required parameter changes to take action.


----------------------------------------------------------------------------------------------------


# OLD README, DO NOT USE THIS ANYMORE!
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
`chmod +x setup_euler_working.sh`
`~/AISCO_code/setup_euler_working.sh`
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
All these files are stored in /cluster/work/fuge/AISCO.


## Environment Activation Guide (do this every time you open a new terminal)
for AISCO, please activate myenv in the `cd ~/AISCO_code/` folder:
`source aisco_env/bin/activate`
