#!/bin/bash
# =============================================================================
# OpenFOAM Full Pipeline — One-button end-to-end submission
#
# Runs PLY→STL conversion + case generation on a single node, then submits
# the mesh → solve → extract chain as SLURM array jobs with dependencies.
#
# Result: finished .pt files in the dataset output directory.
#
# Usage:
#   sbatch openfoam_full_pipeline.slurm
#   sbatch openfoam_full_pipeline.slurm --export=ALL,MAX_PARALLEL=50
# =============================================================================

#SBATCH --job-name=OF_Pipeline
#SBATCH --partition=normal.4h
#SBATCH --account=public
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/elihuber/scratch_logs/OF_Pipeline_%j.out
#SBATCH --error=/cluster/scratch/elihuber/scratch_logs/OF_Pipeline_%j.err

set -euo pipefail

echo "=== OpenFOAM Full Pipeline at $(date) ==="
echo "Job ID: ${SLURM_JOB_ID}"

# --- Configuration -----------------------------------------------------------
MAX_PARALLEL="${MAX_PARALLEL:-25}"
V_INF="${V_INF:-25.0}"

SCRIPT_DIR="/cluster/home/elihuber/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/dataset/openfoam"
SLURM_SCRIPTS="${SCRIPT_DIR}/slurm"

PLY_INPUT="/cluster/work/fuge/AISCO/wp1_cfd_ai/openfoam_geometry"
STL_OUTPUT="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/GEOMETRY"
GEO_MANIFEST="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/geometry_manifest.json"
CASES_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/CASES_v3"
CASES_MANIFEST="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/cases_manifest.json"

# Ensure scratch logs directory exists
mkdir -p /cluster/scratch/elihuber/scratch_logs

# --- Activate Python environment ---------------------------------------------
source /cluster/home/elihuber/AISCO_code/aisco_env/bin/activate

# --- Step 1: PLY → STL conversion --------------------------------------------
echo ""
echo "=== Step 1: PLY → STL conversion ==="
python "${SCRIPT_DIR}/ply_to_stl.py" \
    --input-dir "${PLY_INPUT}" \
    --output-dir "${STL_OUTPUT}" \
    --manifest "${GEO_MANIFEST}" \
    --verbose

echo "PLY→STL complete at $(date)"

# --- Step 2: Generate OpenFOAM cases -----------------------------------------
echo ""
echo "=== Step 2: Generate OpenFOAM cases ==="
python "${SCRIPT_DIR}/generate_cases.py" \
    --manifest "${GEO_MANIFEST}" \
    --output "${CASES_DIR}" \
    --cases-manifest "${CASES_MANIFEST}" \
    --vinf "${V_INF}" \
    --verbose

echo "Case generation complete at $(date)"

# --- Step 3: Determine run IDs from manifest ---------------------------------
echo ""
echo "=== Step 3: Building run list ==="

# Extract all converted run IDs from geometry manifest
RUN_LIST="${CASES_DIR}/run_list_full_$(date -u +%Y%m%dT%H%M%S).txt"
python -c "
import json, sys
with open('${GEO_MANIFEST}') as f:
    geo = json.load(f)
ids = sorted(int(e['run_id'].split('_')[1]) for e in geo if e['status'] == 'converted')
if not ids:
    print('ERROR: No converted runs found in geometry manifest', file=sys.stderr)
    sys.exit(1)
with open('${RUN_LIST}', 'w') as f:
    for i in ids:
        f.write(f'{i:03d}\n')
print(f'Run list: {len(ids)} cases (IDs {ids[0]:03d}-{ids[-1]:03d})')
"

N=$(wc -l < "${RUN_LIST}")
echo "Run list written: ${RUN_LIST} (${N} cases)"

# --- Step 4: Submit mesh → solve → extract chain ----------------------------
echo ""
echo "=== Step 4: Submitting SLURM array jobs (max_parallel=${MAX_PARALLEL}) ==="

bash "${SLURM_SCRIPTS}/submit_pipeline.sh" "${RUN_LIST}" "${MAX_PARALLEL}"

echo ""
echo "=== Full pipeline submitted at $(date) ==="
echo "This prep job (${SLURM_JOB_ID}) will now exit."
echo "The mesh/solve/extract jobs are queued with dependencies."
echo "You can safely disconnect — results will appear in:"
echo "  ${CASES_DIR}/ (OpenFOAM cases)"
echo "  /cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler_openfoam_cp_v050/ (.pt files)"
