<div align="center">
  <img src="https://aisco.dev/assets/69e0fea1c5dcb303f101cd6f_schweizer-fahne-kaufen.jpg.png" alt="AISCO logo" width="180" />
  <h3>AI for Solar Car Optimisation</h3>
  <p>
    <strong>Artificial Intelligence for Solar Car Optimisation</strong><br />
    A focus project of aCentauri Solar Racing and ETH Zürich's IDEAL Lab.
  </p>
  <p>
    <a href="https://aisco.dev/">Website</a> ·
    <a href="https://aisco.dev/team-aisco">Team</a> ·
    <a href="mailto:info@aisco.dev">Contact</a>
  </p>
</div>

<hr />

## About AISCO

<p align="center">
  AISCO revolutionises the design process of solar race cars by exploring a diverse range of aeroshells, estimating their drag and solar performance, and ranking generated concepts by minimal race time.
</p>

<p align="center">
  AISCO predicts aerodynamic drag using a self-built CFD-AI, dynamically balances aerodynamic performance with solar power input, and respects street-legal regulations.
</p>

<h2>Project components</h2>

<table>
  <tr>
    <td width="33%" valign="top">
      <h3>Aeroshell Optimiser</h3>
      <img src="https://aisco.dev/assets/6a310989b0aad0ed3ded69dd_single.gif" alt="Aeroshell optimiser" width="100%" />
      <p>Explores new aeroshell designs by minimising weighted aerodynamic and solar-performance losses. Designs are ranked by minimal race time.</p>
    </td>
    <td width="33%" valign="top">
      <h3>Assembler</h3>
      <img src="https://aisco.dev/assets/6a1464fb50fe2242b9237fe5_assembler3.png" alt="Assembler" width="100%" />
      <p>Transforms regulations, spatial boundaries, vision standards, and internal components into buildable, compliant geometry.</p>
    </td>
    <td width="33%" valign="top">
      <h3>CFD AI</h3>
      <img src="https://aisco.dev/assets/6a310987f03fad7253753d8f_drag.gif" alt="CFD AI drag prediction" width="100%" />
      <p>Predicts aerodynamic drag nearly instantaneously from vehicle geometry, replacing repeated high-fidelity CFD simulations during optimisation.</p>
    </td>
  </tr>
  <tr>
    <td width="33%" valign="top">
      <h3>GUI</h3>
      <img src="https://aisco.dev/assets/6a146d957b7d27d1026c4960_gui.png" alt="AISCO graphical user interface" width="100%" />
      <p>Gives engineers control over design constraints, weights, and race regulations through an intuitive graphical interface.</p>
    </td>
    <td width="33%" valign="top">
      <h3>Solar Estimator</h3>
      <img src="https://aisco.dev/assets/6a3109868b98ff02cfc9ad9b_solar.gif" alt="Solar estimator" width="100%" />
      <p>Models geometry-dependent solar effects and feeds estimated solar power back into the optimiser as a core loss function.</p>
    </td>
    <td width="33%" valign="top">
      <h3>LLM · aCeBot</h3>
      <img src="https://aisco.dev/assets/6a14648a76aae6da7ce5659f_llm2.png" alt="aCeBot" width="100%" />
      <p>Provides structured access to internal documentation, chats, and project history to support knowledge transfer.</p>
    </td>
  </tr>
</table>

<h2>The team</h2>

<p align="center">
  <img src="https://aisco.dev/assets/6a1362cbe7d0227e0f20060a_team.jpg" alt="AISCO team" width="65%" />
</p>

<p align="center"><small>
  9 ETH Focus Students and 9 Freelancers<br />
  <strong>Back row:</strong> Emanuel Schärer · Clemens Huber · Kolja Diehl · Robin Arnold · Federico Sanchez<br />
  <strong>Front row:</strong> Elias Huber · Tippi Engel · Alex Althaus · Victor Diekmann
</small></p>

<h2>Supporters</h2>

<table>
  <tr>
    <td width="33%" align="center"><strong>IDEAL Lab</strong><br /><small>Institute for Design, Engineering and Learning.</small></td>
    <td width="33%" align="center"><strong>Hasler Foundation</strong><br /><small>Supporting ICT for the benefit of Switzerland.</small></td>
    <td width="33%" align="center"><strong>D-MAVT</strong><br /><small>Department of Mechanical and Process Engineering, Sustainability Fund.</small></td>
  </tr>
</table>

<p align="center"><small>
  Supported by Prof. Dr. Mark Fuge · Dr. Soheyl Massoudi · Tim Aebersold · Shizheng Wen
</small></p>

<hr />

# AISCO_code setup and usage

## Setup

Before starting, make sure that you have set up Euler correctly and understand the Euler workflows.

### 1. Log in to an Euler login node

```bash
ssh username@euler.ethz.ch
```

### 2. Clone the repository

```bash
git clone git@github.com:IDEALLab/AISCO_code.git
cd AISCO_code
```

### 3. Install the virtual environment

Request an SRUN job with an RTX Pro 6000:

```bash
srun --cpus-per-task=1 --mem-per-cpu=8G --time=4:00:00 --gpus=pro_6000:1 --pty bash
```

Then run:

```bash
chmod +x setup_euler_working.sh
./setup_euler_working.sh
```

## Usage

The main file to customise is `~/AISCO_code/user_overrides.yaml`.

### Required parameters

- Set the top-level `enabled` flag to `true`.
- Configure `wp1.setup.wandb_entity` and `wp1.setup.wandb_project`.
- Configure `wp4.general_config.wandb_general.entity_name` and `wp4.general_config.wandb_general.project_name`.
- Set `wp4.general_config.storage_paths.storage_dir`, for example:

  ```text
  /cluster/work/fuge/AISCO/report_temp
  ```

### Optional parameters

- `wp1.optimizer.args.epoch`: number of epochs for GAOT model training.
- `wp4.general_config.general_info.latent_space`: enable or disable latent-space training.
- `wp4.losses_v0_config.aero`: select the aerodynamic loss implementation. Set exactly one of `gaot_loss` or `null_loss_aero` to `1`.
- `wp4.losses_v0_config.solar`: select the solar-loss implementation or disable it with `null_loss_solar`. Exactly one solar-loss option must be set to `1`.
- `wp4.training_config.training_params.num_epochs`: number of epochs for the optimisation loop.

### Run a workflow

Train a GAOT model:

```bash
sbatch ~/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/slurm_scripts/wp1_main.sh
```

Run an optimisation loop:

```bash
sbatch ~/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/main_testing.sh
```

The active configuration files are:

- GAOT training: `~/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/config/main_config.yaml`
- Optimisation loop: `~/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/configs`

Make sure that `user_overrides.yaml` is enabled for your changes to take effect.

<details>
<summary><strong>Developer tools and VS Code recommendations</strong></summary>

### Recommended VS Code extensions

- Python
- Git Graph
- Ruff
- Mypy Type Checker
- PLY Viewer

### Recommended settings

- Enable **Git: Blame** to see who edited each line.
- Enable **Format on Save** to activate Ruff and keep the code consistent.
- Use VS Code's **Files: Exclude** settings to hide unnecessary folders with patterns beginning with `**/`.

</details>

<details>
<summary><strong>Legacy installation notes</strong></summary>

For systems without a Blackwell-compatible GPU, the older setup used:

```text
torch: 2.4.0
torchvision: 0.19.0
```

On a legacy Euler setup, use `setup_euler_old.sh`, then load the required modules and install the project:

```bash
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy
source aisco_env/bin/activate
python3 -m pip install -e '.[all]' --no-build-isolation
```

If required for WP1:

```bash
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy
source /cluster/home/$USER/AISCO_code/aisco_env/bin/activate
python -m pip uninstall -y torch-cluster
python -m pip install --no-cache-dir torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

On a fresh Euler installation, some compiled `.so` files may need to be copied manually from `/cluster/work/fuge/AISCO` into the corresponding project directories.

</details>

## Activate the environment

Run this each time you open a new terminal:

```bash
cd ~/AISCO_code
source aisco_env/bin/activate
```

<hr />

<p align="center">
  <a href="https://aisco.dev/">aisco.dev</a> ·
  <a href="mailto:info@aisco.dev">info@aisco.dev</a> ·
  <a href="https://www.acentauri.ch/impressum">Impressum</a><br />
  © 2026 aCentauri Solar Racing · ETH Student Project
</p>
