#!/usr/bin/env bash
# =============================================================================
# run_urans_batch_array.slurm — URANS overnight array job (all prepared cases)
#
# Maps each SLURM array task ID to a line in urans_ready_cases.txt.
# Each task: decomposePar -force → pimpleFoam -parallel → reconstructPar → cleanup
#
# Usage:
#   1. Run prepare_urans_batch.sh first (creates urans_ready_cases.txt)
#   2. N=$(wc -l < urans_ready_cases.txt)
#   3. sbatch --array=1-${N}%10 run_urans_batch_array.slurm
#
# For just the 23 existing cases:
#   sbatch --array=1-23%10 run_urans_batch_array.slurm
# =============================================================================
#SBATCH --job-name=OF_URANS_BATCH
#SBATCH --partition=normal.24h
#SBATCH --account=es_fuge
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/scratch/elihuber/scratch_logs/OF_URANS_BATCH_%A_%a.out
#SBATCH --error=/cluster/scratch/elihuber/scratch_logs/OF_URANS_BATCH_%A_%a.err

set -euo pipefail

module load stack/2024-06 gcc/12.2.0 openmpi/4.1.6
module load openfoam/2306

CASES_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/CASES_V2"
READY_LIST="/cluster/home/elihuber/AISCO_code/urans_ready_cases.txt"

# Map task ID to run_id
RUN_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${READY_LIST}")
if [[ -z "${RUN_ID}" ]]; then
    echo "ERROR: No run_id at line ${SLURM_ARRAY_TASK_ID} in ${READY_LIST}" >&2
    exit 1
fi

URANS_DIR="${CASES_DIR}/${RUN_ID}_urans"

echo "=== [${RUN_ID}] URANS batch task ${SLURM_ARRAY_TASK_ID} started at $(date) ==="
echo "Job: ${SLURM_JOB_ID}  Array: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Case dir: ${URANS_DIR}"

if [[ ! -d "${URANS_DIR}" ]]; then
    echo "ERROR: ${URANS_DIR} does not exist. Run prepare_urans_batch.sh first." >&2
    exit 1
fi

if ! grep -q "pimpleFoam" "${URANS_DIR}/system/controlDict"; then
    echo "ERROR: ${URANS_DIR} not patched for URANS (no pimpleFoam in controlDict)." >&2
    exit 1
fi

cd "${URANS_DIR}"

# Check if already fully complete (postProcessing written beyond start time)
LATEST_FORCEDIR=$(ls -d postProcessing/forceCoeffs/[0-9]*/ 2>/dev/null | sort -V | tail -1 || true)
END_TIME=$(grep -E '^endTime' system/controlDict | awk '{print $2}' | tr -d ';')
if [[ -n "${LATEST_FORCEDIR}" ]]; then
    LATEST_T=$(basename "${LATEST_FORCEDIR%/}")
    if python3 -c "exit(0 if float('${LATEST_T}') >= float('${END_TIME}') - 0.01 else 1)" 2>/dev/null; then
        echo "SKIP: ${RUN_ID}_urans already complete (forceCoeffs at t=${LATEST_T} >= endTime=${END_TIME})"
        exit 0
    fi
fi

# Remove stale processor dirs from any previous partial run
rm -rf processor*

echo "[${RUN_ID}] Running decomposePar ..."
decomposePar -force > log.decomposePar_urans 2>&1
N_PROC=$(ls -d processor* 2>/dev/null | wc -l)
echo "[${RUN_ID}] decomposePar: ${N_PROC} processor dirs created"
if [[ "${N_PROC}" -ne 128 ]]; then
    echo "ERROR: Expected 128 processor dirs, got ${N_PROC}" >&2
    exit 1
fi

echo "[${RUN_ID}] Running pimpleFoam (endTime=${END_TIME}) ..."
mpirun -np 128 pimpleFoam -parallel > log.pimpleFoam 2>&1
FOAM_EXIT=$?
echo "[${RUN_ID}] pimpleFoam exit code: ${FOAM_EXIT}"
if [[ "${FOAM_EXIT}" -ne 0 ]]; then
    echo "ERROR: pimpleFoam failed with exit code ${FOAM_EXIT}" >&2
    exit "${FOAM_EXIT}"
fi

echo "[${RUN_ID}] Reconstructing latest time ..."
reconstructPar -latestTime > log.reconstructPar_urans 2>&1

echo "[${RUN_ID}] Cleaning processor dirs ..."
rm -rf processor*

echo "=== [${RUN_ID}] URANS batch task ${SLURM_ARRAY_TASK_ID} complete at $(date) ==="
