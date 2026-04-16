# Mesh Benchmark Tool
Purpose: compare runtime and memory usage of mesh extraction methods with identical main-file execution flow.
## Files
1. tests/wp5/mesh_benchmark_runner.py
2. tests/wp5/mesh_benchmark_scenarios.yaml
## Supported Main Targets (preconfigured)
1. subprojects/wp4_aeroshell_optimizer/v1/src/main_files/main_v1_1_one_model.py
2. subprojects/wp4_aeroshell_optimizer/v2/src/main_files/main_v1_2_GINN_concat.py
## Supported GPU Profiles (preconfigured)
1. rtx2080
2. rtx4090
Note: add rtx5090 later by extending gpu_profiles in the scenario yaml.
## How It Works
1. Reads scenario yaml.
2. Applies yaml config patches per run and per extractor case.
3. Executes each main file as a subprocess.
4. Polls nvidia-smi to track process-level and total GPU memory usage.
5. Stores per-sample traces and aggregated CSV/JSON results.
6. Restores original config files after each case.
## Run Example
source /cluster/home/eschaerer/AISCO_code/aisco_env/bin/activate
PYTHONPATH=/cluster/home/eschaerer/AISCO_code python /cluster/home/eschaerer/AISCO_code/tests/wp5/mesh_benchmark_runner.py --scenario /cluster/home/eschaerer/AISCO_code/tests/wp5/mesh_benchmark_scenarios.yaml
## Useful Filters
1. --only-runs v1_main_v1_1_one_model
2. --only-cases mesh_sdf diso_mc
3. --only-gpus rtx2080
4. --dry-run
## SLURM Compatibility Flags
1. --force-cuda-visible-devices 0
2. --force-gpu-query-index 0
Use these when SLURM remaps visible devices so scenario profile ids do not match physical gpu indices.
## Worktree SLURM Launcher
Use script: subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/worktree_launch_mesh_benchmark.sh
Edit device selection at the top of that script:
1. GPU_TYPE
2. ONLY_GPUS
3. FORCE_CUDA_VISIBLE_DEVICES
4. FORCE_GPU_QUERY_INDEX
## Output
Output root: tests/wp5/benchmark_results
Per run stamp folder contains:
1. results.csv
2. results.json
3. per-case stdout log files
4. per-case memory sample json files
## Important Note About v2 solar_v3B_2
The v2 wrapper currently loads v1 solar config path for SolarLossConfigV3; scenario patches therefore target the v1 solar config file for extractor method changes.
