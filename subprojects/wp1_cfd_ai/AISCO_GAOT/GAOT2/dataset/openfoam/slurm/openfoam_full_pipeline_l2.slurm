#!/bin/bash
# =============================================================================
# OpenFOAM Full Pipeline (L2) — One-button end-to-end for subdivided geometries
#
# Runs PLY→STL conversion + case generation on a single node, then submits
# the mesh → solve → extract chain as SLURM array jobs with dependencies.
#
# ISOLATED from production:
#   - Input:  of_geom_l2/ (subdivided PLYs)
#   - STL:    OPENFOAM/GEOMETRY_L2/
#   - Cases:  OPENFOAM/CASES_L2/
#   - Output: processed_euler_openfoam_cp_v050_l2/
#
# Usage:
#   sbatch openfoam_full_pipeline_l2.slurm
# =============================================================================

#SBATCH --job-name=OF_Pipeline_L2
#SBATCH --partition=normal.4h
#SBATCH --account=es_fuge
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/elihuber/scratch_logs/OF_Pipeline_L2_%j.out
#SBATCH --error=/cluster/scratch/elihuber/scratch_logs/OF_Pipeline_L2_%j.err

set -euo pipefail

echo "=== OpenFOAM Full Pipeline (L2 — Subdivided Geometry) at $(date) ==="
echo "Job ID: ${SLURM_JOB_ID}"

# --- Configuration -----------------------------------------------------------
MAX_PARALLEL="${MAX_PARALLEL:-25}"
V_INF="${V_INF:-25.0}"

SCRIPT_DIR="/cluster/home/elihuber/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/dataset/openfoam"
SLURM_SCRIPTS="${SCRIPT_DIR}/slurm"

PLY_INPUT="/cluster/work/fuge/AISCO/wp1_cfd_ai/of_geom_l2"
STL_OUTPUT="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/GEOMETRY_L2"
GEO_MANIFEST="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/geometry_manifest_l2.json"
CASES_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/CASES_L2"
CASES_MANIFEST="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/cases_manifest_l2.json"

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

RUN_LIST="${CASES_DIR}/run_list_l2_$(date -u +%Y%m%dT%H%M%S).txt"
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
echo "=== Step 4: Submitting L2 SLURM array jobs (max_parallel=${MAX_PARALLEL}) ==="

bash "${SLURM_SCRIPTS}/submit_pipeline_l2.sh" "${RUN_LIST}" "${MAX_PARALLEL}"
