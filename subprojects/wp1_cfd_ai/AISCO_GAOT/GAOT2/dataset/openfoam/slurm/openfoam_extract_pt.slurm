#!/bin/bash
# =============================================================================
# OpenFOAM .pt Extraction — Python post-processing for all cases
#
# Runs process_openfoam_pressure.py to extract .pt files from
# OpenFOAM VTK output. Submit after solve array completes.
#
# Usage:
#   sbatch --dependency=afterok:<solve_job_id> openfoam_extract_pt.slurm
# =============================================================================

#SBATCH --job-name=OF_Extract
#SBATCH --partition=normal.4h
#SBATCH --account=public
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/scratch_logs/OF_Extract_%j.out
#SBATCH --error=/cluster/scratch/%u/scratch_logs/OF_Extract_%j.err

set -euo pipefail

echo "=== OpenFOAM .pt Extraction at $(date) ==="

# Activate Python environment
source /cluster/home/$USER/AISCO_code/aisco_env/bin/activate

SCRIPT_DIR="/cluster/home/$USER/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/dataset/openfoam"
DATASET_DIR="/cluster/home/$USER/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/dataset"
CASES_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM_federico/CASES_v3"
OUTPUT_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler_openfoam_cp_v050"

# common_preprocessing lives in dataset/, one level up from openfoam/
export PYTHONPATH="${DATASET_DIR}:${PYTHONPATH:-}"

python "${SCRIPT_DIR}/process_openfoam_pressure.py" \
    --cases-dir "${CASES_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --time-step 4500 \
    --workers 16 \
    --verbose

echo "=== Extraction complete at $(date) ==="

# ---- Convergence QA gate ----
echo "=== Running convergence QA at $(date) ==="
python "${SCRIPT_DIR}/convergence_qa.py" \
    --dataset-dir "${OUTPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}/qa" \
    --verbose

echo "=== Convergence QA complete at $(date) ==="
