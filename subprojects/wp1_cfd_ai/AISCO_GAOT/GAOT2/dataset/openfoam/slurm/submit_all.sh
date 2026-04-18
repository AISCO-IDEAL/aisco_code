#!/bin/bash
# =============================================================================
# OpenFOAM Pipeline Orchestrator — submit_all.sh
#
# Submits the mesh array job, then the solve array job with
# --afterok dependency so each solve task only starts if its
# corresponding mesh task succeeded.
#
# Usage:
#   bash submit_all.sh              # submit all 223
#   bash submit_all.sh 1-10         # submit array range 1-10
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARRAY_RANGE="${1:-1-223}"
MAX_PARALLEL="${2:-20}"

echo "=== OpenFOAM Pipeline Submission ==="
echo "Array range: ${ARRAY_RANGE}"
echo "Max parallel: ${MAX_PARALLEL}"
echo ""

# Ensure scratch log directory exists
mkdir -p /cluster/scratch/elihuber/scratch_logs

# Submit mesh array
MESH_ARGS="--array=${ARRAY_RANGE}%${MAX_PARALLEL}"
echo "Submitting mesh jobs..."
MESH_JOBID=$(sbatch --parsable ${MESH_ARGS} "${SCRIPT_DIR}/openfoam_mesh_array.slurm")
echo "Mesh job submitted: ${MESH_JOBID}"

# Submit solve array with dependency on mesh
echo "Submitting solve jobs (dependency: afterok:${MESH_JOBID})..."
SOLVE_JOBID=$(sbatch --parsable ${MESH_ARGS} --dependency=afterok:${MESH_JOBID} "${SCRIPT_DIR}/openfoam_solve_array.slurm")
echo "Solve job submitted: ${SOLVE_JOBID}"

# Submit .pt extraction with dependency on solve
echo "Submitting .pt extraction (dependency: afterok:${SOLVE_JOBID})..."
EXTRACT_JOBID=$(sbatch --parsable --dependency=afterok:${SOLVE_JOBID} "${SCRIPT_DIR}/openfoam_extract_pt.slurm")
echo "Extract job submitted: ${EXTRACT_JOBID}"

echo ""
echo "=== Pipeline submitted ==="
echo "Mesh:    ${MESH_JOBID} (array ${ARRAY_RANGE})"
echo "Solve:   ${SOLVE_JOBID} (array ${ARRAY_RANGE}, after mesh)"
echo "Extract: ${EXTRACT_JOBID} (after all solves)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j ${MESH_JOBID} --format=JobID,State,ExitCode,Elapsed"
echo "  sacct -j ${SOLVE_JOBID} --format=JobID,State,ExitCode,Elapsed"
echo "  sacct -j ${EXTRACT_JOBID} --format=JobID,State,ExitCode,Elapsed"
