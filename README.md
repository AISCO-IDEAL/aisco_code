# AISCO Code

Monorepo for AISCO workflows, including:
- WP1: GAOT model training
- WP4: aeroshell optimization loop
- WP7: GUI for launching and viewing runs
- supporting subprojects for geometry, meshing, and analysis

This README is a **first-time visitor guide** focused on getting a working run quickly.

## Repository map

- `/subprojects/wp1_cfd_ai/` — GAOT training components
- `/subprojects/wp2_assembler/` — no-build-zone/assembly geometry generation
- `/subprojects/wp3_solar_estimator/` — solar and meshing-related tooling
- `/subprojects/wp4_aeroshell_optimizer/` — optimization workflows (active)
- `/subprojects/wp5/` — additional tooling and experiments
- `/subprojects/wp7_gui/` — web GUI for run orchestration/viewing
- `/user_overrides.yaml` — your user-level runtime overrides (critical)

## Quickstart (Euler)

> Prerequisite: you can log in to Euler and run standard SLURM workflows.

1) Log in:

```bash
ssh <username>@euler.ethz.ch
```

2) Clone this repository:

```bash
git clone git@github.com:AISCO-IDEAL/aisco_code.git
cd aisco_code
```

3) Request an interactive GPU job for setup:

```bash
srun --cpus-per-task=1 --mem-per-cpu=8G --time=4:00:00 --gpus=pro_6000:1 --pty bash
```

4) Install the virtual environment:

```bash
chmod +x setup_euler_working.sh
./setup_euler_working.sh
```

5) Activate the environment (every new terminal/session):

```bash
source aisco_env/bin/activate
```

## First-run configuration (`user_overrides.yaml`)

Before launching workflows, update `/user_overrides.yaml`:

### Required
- Set top-level `enabled: true`
- Set wandb values:
  - `wp1.setup.wandb_entity`
  - `wp1.setup.wandb_project`
  - `wp4.general_config.wandb_general.entity_name`
  - `wp4.general_config.wandb_general.project_name`
- Set storage output path:
  - `wp4.general_config.storage_paths.storage_dir`
  - Example: `/cluster/work/fuge/AISCO/report_temp`

### Common optional tuning
- `wp1.optimizer.args.epoch` (GAOT training epochs)
- `wp4.training_config.training_params.num_epochs` (optimization epochs)
- `wp4.general_config.general_info.latent_space` (enable/disable latent-space training)
- `wp4.losses_v0_config.aero` (choose GAOT aero loss vs null aero loss)
- `wp4.losses_v0_config.solar` (choose active solar loss variant vs null solar loss)

## Main workflows

### 1) Train a GAOT model (WP1)

```bash
sbatch ~/aisco_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/slurm_scripts/wp1_main.sh
```

Notes:
- Requires access to project storage used by WP1 data/workflows.
- Tune WP1-related overrides in `/user_overrides.yaml` first.

### 2) Run aeroshell optimization loop (WP4)

```bash
sbatch ~/aisco_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/main_testing.sh
```

Notes:
- Tune WP4-related overrides in `/user_overrides.yaml` first.

## Active config locations

- WP1 GAOT training config:
  - `/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/config/main_config.yaml`
- WP4 optimization configs:
  - `/subprojects/wp4_aeroshell_optimizer/v2/configs/`

## Additional setup notes

- BEAST setup notes: `/BEAST_SETUP_WORKING.md`
- Alternative/legacy setup scripts exist in repository root:
  - `setup_beast.sh`
  - `setup_euler_new.sh`
  - `setup_euler_old.sh`
  - `setup_wsl.sh`

## New contributor checklist

1. Clone repository and create environment.
2. Activate `aisco_env`.
3. Update `/user_overrides.yaml` required fields.
4. Launch either WP1 training or WP4 optimization via SLURM.
5. Check logs/artifacts in your configured storage location.
