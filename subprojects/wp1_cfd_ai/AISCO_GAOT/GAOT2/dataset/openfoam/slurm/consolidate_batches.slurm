#!/bin/bash
# =============================================================================
# OpenFOAM Federico — Post-QA Consolidation Job
#
# Reads convergence_report.json from each dirty/{batch}/qa/ directory and
# populates:
#   clean/  — PASS + WARN samples (flat, batch-prefixed names)
#   strict/ — PASS-only samples
#
# Submit AFTER the extraction array job completes:
#   EXTRACT_JOB=$(sbatch --parsable extract_qa_all_batches.slurm)
#   sbatch --dependency=afterok:${EXTRACT_JOB} consolidate_batches.slurm
# =============================================================================

#SBATCH --job-name=OF_Consolidate
#SBATCH --partition=normal.4h
#SBATCH --account=es_fuge
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/scratch_logs/OF_Consolidate_%j.out
#SBATCH --error=/cluster/scratch/%u/scratch_logs/OF_Consolidate_%j.err

set -euo pipefail

echo "=== OF_Consolidate at $(date) ==="
echo "Node: ${SLURMD_NODENAME}"

source "/cluster/home/${USER}/AISCO_code/aisco_env/bin/activate"

REPO="/cluster/home/${USER}/AISCO_code"
SCRIPT="${REPO}/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/dataset/openfoam/consolidate_openfoam.py"
MERGED_ROOT="/cluster/work/fuge/AISCO/wp1_cfd_ai/openfoam_merged"

python3 "${SCRIPT}" \
    --dirty-dir  "${MERGED_ROOT}/dirty" \
    --clean-dir  "${MERGED_ROOT}/clean" \
    --strict-dir "${MERGED_ROOT}/strict" \
    --verbose

echo "=== Consolidation complete at $(date) ==="
echo ""
echo "Results:"
echo "  dirty/  : $(find "${MERGED_ROOT}/dirty"  -name 'data.pt' | wc -l) .pt files"
echo "  clean/  : $(find "${MERGED_ROOT}/clean"  -name 'data.pt' | wc -l) .pt files"
echo "  strict/ : $(find "${MERGED_ROOT}/strict" -name 'data.pt' | wc -l) .pt files"
