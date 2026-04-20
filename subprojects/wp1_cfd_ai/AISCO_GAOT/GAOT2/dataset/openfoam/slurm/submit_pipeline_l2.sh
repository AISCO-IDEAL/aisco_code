#!/bin/bash
# =============================================================================
# OpenFOAM L2 Pipeline Submission — submit_pipeline_l2.sh
#
# Submits the mesh → solve → extract chain for L2 (subdivided) geometries.
# Uses CASES_L2 and outputs to processed_euler_openfoam_cp_v050_l2.
# Completely isolated from production CASES_v3.
#
# Usage:
#   bash submit_pipeline_l2.sh /path/to/run_ids.txt              # default max 25 parallel
#   bash submit_pipeline_l2.sh /path/to/run_ids.txt 50           # max 50 parallel
#   bash submit_pipeline_l2.sh /path/to/run_ids.txt --no-extract # skip extraction
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Argument parsing --------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <run_list_file> [max_parallel] [--no-extract]" >&2
    exit 1
fi

RUN_LIST="$(realpath "$1")"
SKIP_EXTRACT=0

for arg in "$@"; do
    if [[ "$arg" == "--no-extract" ]]; then
        SKIP_EXTRACT=1
    fi
done

if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
    MAX_PARALLEL="$2"
else
    MAX_PARALLEL=25
fi

# --- Validation --------------------------------------------------------------
if [[ ! -f "$RUN_LIST" ]]; then
    echo "ERROR: Run-list file not found: $RUN_LIST" >&2
    exit 1
fi

N=$(wc -l < "$RUN_LIST")
if [[ "$N" -eq 0 ]]; then
    echo "ERROR: Run-list file is empty: $RUN_LIST" >&2
    exit 1
fi

echo "=== OpenFOAM L2 Pipeline Submission ==="
echo "Run-list: ${RUN_LIST} (${N} cases)"
echo "Max parallel: ${MAX_PARALLEL}"
echo "Cases dir: CASES_L2 (isolated from production)"
echo ""

mkdir -p /cluster/scratch/elihuber/scratch_logs

# --- Submit mesh array -------------------------------------------------------
echo "Submitting L2 mesh jobs (${N} tasks)..."
MESH_JOBID=$(sbatch --parsable \
    --array="1-${N}%${MAX_PARALLEL}" \
    --export="ALL,RUN_LIST=${RUN_LIST}" \
    "${SCRIPT_DIR}/openfoam_mesh_array_l2.slurm")
echo "Mesh job submitted: ${MESH_JOBID}"

# --- Submit solve array with per-task dependency (aftercorr) -----------------
echo "Submitting L2 solve jobs (dependency: aftercorr:${MESH_JOBID})..."
SOLVE_JOBID=$(sbatch --parsable \
    --array="1-${N}%${MAX_PARALLEL}" \
    --dependency="aftercorr:${MESH_JOBID}" \
    --export="ALL,RUN_LIST=${RUN_LIST}" \
    "${SCRIPT_DIR}/openfoam_solve_array_l2.slurm")
echo "Solve job submitted: ${SOLVE_JOBID}"

# --- Submit extraction -------------------------------------------------------
if [[ "$SKIP_EXTRACT" -eq 0 ]]; then
    echo "Submitting L2 .pt extraction (dependency: afterany:${SOLVE_JOBID})..."
    EXTRACT_JOBID=$(sbatch --parsable \
        --dependency="afterany:${SOLVE_JOBID}" \
        "${SCRIPT_DIR}/openfoam_extract_pt_l2.slurm")
    echo "Extract job submitted: ${EXTRACT_JOBID}"
fi

echo ""
echo "=== L2 Pipeline submitted ==="
echo "Mesh:    ${MESH_JOBID} (array 1-${N}%${MAX_PARALLEL})"
echo "Solve:   ${SOLVE_JOBID} (array 1-${N}%${MAX_PARALLEL}, aftercorr:mesh)"
if [[ "$SKIP_EXTRACT" -eq 0 ]]; then
    echo "Extract: ${EXTRACT_JOBID} (afterany:solve)"
fi
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j ${MESH_JOBID} --format=JobID,State,ExitCode,Elapsed"
