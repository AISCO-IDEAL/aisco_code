#!/bin/bash
# =============================================================================
# OpenFOAM Pipeline Submission — submit_pipeline.sh
#
# Submits the mesh → solve → extract chain using a run-list file.
# Each array task reads its RUN_ID from the file, enabling incremental
# and selective pipeline modes (full, append, rerun).
#
# Uses aftercorr dependency: mesh[N] → solve[N] (per-task dependency).
# If mesh task N fails, only solve task N is cancelled.
#
# Usage:
#   bash submit_pipeline.sh /path/to/run_ids.txt              # default max 25 parallel
#   bash submit_pipeline.sh /path/to/run_ids.txt 50           # max 50 parallel
#   bash submit_pipeline.sh /path/to/run_ids.txt 20 --no-extract  # skip extraction
#   bash submit_pipeline.sh /path/to/run_ids.txt --no-extract     # default parallel, no extract
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Argument parsing --------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <run_list_file> [max_parallel] [--no-extract]" >&2
    echo "" >&2
    echo "  run_list_file   Text file with one RUN_ID per line (e.g., 001, 010, 200)" >&2
    echo "  max_parallel    Max simultaneous SLURM array tasks (default: 25)" >&2
    echo "  --no-extract    Skip the extraction + QA step" >&2
    exit 1
fi

RUN_LIST="$(realpath "$1")"
SKIP_EXTRACT=0

# Detect --no-extract BEFORE reading MAX_PARALLEL — $2 could be --no-extract
# if the caller omits max_parallel, which would corrupt the sbatch --array flag.
for arg in "$@"; do
    if [[ "$arg" == "--no-extract" ]]; then
        SKIP_EXTRACT=1
    fi
done

# Only use $2 as MAX_PARALLEL if it's a number, not a flag
if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
    MAX_PARALLEL="$2"
else
    MAX_PARALLEL=25
fi

# --- Validation --------------------------------------------------------------
if [[ ! -f "$RUN_LIST" ]]; then
    echo "ERROR: Run-list file not found: $RUN_LIST" >&2
    echo "Create a text file with one RUN_ID per line (e.g., 001, 010, 200)." >&2
    exit 1
fi

N=$(wc -l < "$RUN_LIST")
if [[ "$N" -eq 0 ]]; then
    echo "ERROR: Run-list file is empty: $RUN_LIST" >&2
    exit 1
fi

echo "=== OpenFOAM Pipeline Submission ==="
echo "Run-list: ${RUN_LIST} (${N} cases)"
echo "Max parallel: ${MAX_PARALLEL}"
echo "Batch ID: $(date +%Y%m%dT%H%M%S)"
echo ""

# Ensure scratch log directory exists
mkdir -p /cluster/scratch/elihuber/scratch_logs

# --- Submit mesh array -------------------------------------------------------
echo "Submitting mesh jobs (${N} tasks)..."
MESH_JOBID=$(sbatch --parsable \
    --array="1-${N}%${MAX_PARALLEL}" \
    --export="ALL,RUN_LIST=${RUN_LIST}" \
    "${SCRIPT_DIR}/openfoam_mesh_array.slurm")
echo "Mesh job submitted: ${MESH_JOBID}"

# --- Submit solve array with per-task dependency (aftercorr) -----------------
echo "Submitting solve jobs (dependency: aftercorr:${MESH_JOBID})..."
SOLVE_JOBID=$(sbatch --parsable \
    --array="1-${N}%${MAX_PARALLEL}" \
    --dependency="aftercorr:${MESH_JOBID}" \
    --export="ALL,RUN_LIST=${RUN_LIST}" \
    "${SCRIPT_DIR}/openfoam_solve_array.slurm")
echo "Solve job submitted: ${SOLVE_JOBID}"

# --- Submit extraction (optional) --------------------------------------------
if [[ "$SKIP_EXTRACT" -eq 0 ]]; then
    echo "Submitting .pt extraction (dependency: afterany:${SOLVE_JOBID})..."
    EXTRACT_JOBID=$(sbatch --parsable \
        --dependency="afterany:${SOLVE_JOBID}" \
        "${SCRIPT_DIR}/openfoam_extract_pt.slurm")
    echo "Extract job submitted: ${EXTRACT_JOBID}"
fi

echo ""
echo "=== Pipeline submitted ==="
echo "Mesh:    ${MESH_JOBID} (array 1-${N}%${MAX_PARALLEL})"
echo "Solve:   ${SOLVE_JOBID} (array 1-${N}%${MAX_PARALLEL}, aftercorr:mesh)"
if [[ "$SKIP_EXTRACT" -eq 0 ]]; then
    echo "Extract: ${EXTRACT_JOBID} (afterany:solve)"
fi
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j ${MESH_JOBID} --format=JobID,State,ExitCode,Elapsed"
echo "  sacct -j ${SOLVE_JOBID} --format=JobID,State,ExitCode,Elapsed"
if [[ "$SKIP_EXTRACT" -eq 0 ]]; then
    echo "  sacct -j ${EXTRACT_JOBID} --format=JobID,State,ExitCode,Elapsed"
fi
