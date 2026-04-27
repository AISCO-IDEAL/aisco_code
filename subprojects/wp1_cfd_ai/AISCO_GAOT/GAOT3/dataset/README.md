# GAOT3 Dataset Preprocessing

Run preprocessing from the repository root using the unified runner:

```bash
python subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/dataset/preprocess_all_datasets.py --datasets all
```

The dataset-specific scripts are now flattened directly in this folder:

- `preprocessing_euler.py`
- `preprocessing_ahmed_drivaer.py`
- `preprocessing_blended_wing.py`

## Supported Datasets

- `euler`
- `drivaer`
- `ahmed`
- `blended_wing`

## How To Select Datasets

Use `--datasets` followed by one or more dataset names:

```bash
# One dataset
python subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/dataset/preprocess_all_datasets.py --datasets euler

# Multiple datasets
python subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/dataset/preprocess_all_datasets.py --datasets drivaer ahmed

# All datasets
python subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/dataset/preprocess_all_datasets.py --datasets all
```

The editable configuration block in
`preprocess_all_datasets.py` controls source folders, worker counts, and default output base path.

## Run On SLURM

A preprocessing SLURM script is provided at:

`../slurm_scripts/wp1_preprocess.sh`

In that script, edit:

- `DATASETS` to choose which datasets to process
- `OUTPUT_PATH` to override the default config output path
- worker counts and `TARGET_REDUCTION` as needed

Then submit:

```bash
sbatch subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/slurm_scripts/wp1_preprocess.sh
```

## Pipeline Overview

The preprocessing pipeline now follows one orchestrator flow for every dataset:

1. `preprocess_all_datasets.py` reads `PREPROCESSING_CONFIG`.
2. For each selected dataset, it dispatches to the corresponding dataset adapter:
	 - Euler: `preprocessing_euler.py`
	 - DrivAer/Ahmed: `preprocessing_ahmed_drivaer.py`
	 - Blended wing: `preprocessing_blended_wing.py`
3. Each adapter writes per-sample `data.pt` files and then writes dataset-level `order_.txt` and `manifest.json`.
4. Before each sample is saved, the canonical contract is applied and validated so all datasets share the same required keys.

## Output Structure

The runner writes one folder per dataset under a shared output base path:

1. Per-sample subfolders with `data.pt` (`<sample_id>/data.pt`)
2. `order_.txt` containing sample folder names
3. `manifest.json` with counts, source root, and contract version

Example:

```text
<output_path>/euler/
<output_path>/drivaer/
<output_path>/ahmed/
<output_path>/blended_wing/
```

Default `output_path` is set in `PREPROCESSING_CONFIG["defaults"]["output_path"]`.
You can override it per run with:

```bash
python subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/dataset/preprocess_all_datasets.py \
	--datasets euler drivaer \
	--output-path /cluster/work/fuge/AISCO/wp1_cfd_ai/processed_experiment
```

Blended wing sample folders are named `case_XXXX`.

## Canonical Contract

All saved samples are normalized to the same core keys before writing:

- `pos`, `c`, `face`
- `x`, `cp`, `pressure`, `wall_shear`
- `cd`, `force`, `v_inf`
- `geom_extent`, `geom_aspect`, `conditioning`
- `filename`, `source_dataset`, `source_case_id`

For Cp training, `x` and `cp` are enforced to be identical.
