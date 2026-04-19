#!/bin/bash
# =============================================================================
# OpenFOAM Mesh Array Job — snappyHexMesh for all cases
#
# Runs blockMesh + surfaceFeatureExtract + snappyHexMesh + checkMesh
# for each case in parallel via SLURM array.
#
# Features:
#   - reconstructParMesh crash tolerance (Bug 4 + Bug 8)
#   - Cell-count sanity check against STL face count heuristic
#   - Lustre stale-file-handle tolerant cleanup
#   - Structured JSON error logging to stderr
#
# v051: 8 BL layers, expansionRatio 1.15, input geometry from openfoam_geometry.
# Cases live in CASES_v3 (CASES_V2 preserved as v0.4.0 baseline).
# =============================================================================

#SBATCH --job-name=OF_Mesh
#SBATCH --partition=normal.4h
#SBATCH --account=es_fuge
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1900M
#SBATCH --array=1-234%25
#SBATCH --output=/cluster/scratch/elihuber/scratch_logs/OF_Mesh_%A_%a.out
#SBATCH --error=/cluster/scratch/elihuber/scratch_logs/OF_Mesh_%A_%a.err

set -euo pipefail

# Load OpenFOAM
# Note: Euler's spack-installed module sets the full environment at load time.
# No bashrc sourcing needed.
module purge
module load stack/2024-06 gcc/12.2.0 openmpi/4.1.6
module load openfoam/2306

# --- Helper functions --------------------------------------------------------

# Lustre-safe processor directory cleanup (tolerates stale file handles)
lustre_safe_cleanup() {
    rm -rf processor* 2>/dev/null || true
    if ls -d processor* >/dev/null 2>&1; then
        sleep 5
        rm -rf processor* 2>/dev/null || true
    fi
}

# Structured JSON event logging
log_event() {
    local level="$1" stage="$2" message="$3"
    echo "{\"run_id\":\"RUN_${RUN_ID}\",\"job_id\":\"${SLURM_JOB_ID:-unknown}\",\"timestamp\":\"$(date -Iseconds)\",\"level\":\"${level}\",\"stage\":\"${stage}\",\"message\":\"${message}\"}"
}

# Structured JSON error logging (to stderr)
log_error() {
    local error_class="$1" stage="$2" exit_code="$3" message="$4" retry="$5"
    echo "{\"run_id\":\"RUN_${RUN_ID}\",\"job_id\":\"${SLURM_JOB_ID:-unknown}\",\"timestamp\":\"$(date -Iseconds)\",\"level\":\"error\",\"error_class\":\"${error_class}\",\"stage\":\"${stage}\",\"exit_code\":${exit_code},\"error_message\":\"${message}\",\"retry_action\":\"${retry}\"}" >&2
}

# --- Determine case directory ------------------------------------------------

if [[ -n "${RUN_LIST:-}" && -f "${RUN_LIST}" ]]; then
    # Run-list mode: read RUN_ID from file by line number
    RUN_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUN_LIST")
    if [[ -z "$RUN_ID" ]]; then
        echo "ERROR: No RUN_ID at line ${SLURM_ARRAY_TASK_ID} in ${RUN_LIST}" >&2
        exit 1
    fi
    # Ensure 3-digit zero-padded format
    # Use 10# prefix to force decimal — bash printf treats leading-zero strings as octal
    # (e.g. "010" octal = 8 decimal, producing RUN_008 instead of RUN_010)
    RUN_ID=$(printf "%03d" "$((10#${RUN_ID}))")
else
    # Legacy mode: use array task ID directly as run number
    RUN_ID=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
fi
CASE_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/CASES_v3/RUN_${RUN_ID}"

if [[ ! -d "$CASE_DIR" ]]; then
    echo "ERROR: Case directory $CASE_DIR does not exist. Skipping."
    exit 0
fi

cd "$CASE_DIR"

echo "=== Meshing RUN_${RUN_ID} at $(date) ==="
log_event "info" "mesh" "mesh_started"

# Read mesh-specific proc count (32 procs — optimal ~47K cells/rank for snappyHexMesh).
# Solve uses system/decomposeParDict (128 procs) separately.
MESH_NPROCS=$(grep "numberOfSubdomains" system/decomposeParDict.mesh | awk '{print $2}' | tr -d ';')
if [[ -z "$MESH_NPROCS" || "$MESH_NPROCS" -lt 1 ]]; then
    log_error "CONFIG_ERROR" "mesh" "1" "Cannot read numberOfSubdomains from decomposeParDict.mesh" "none"
    exit 1
fi
echo "Mesh proc count: ${MESH_NPROCS}"

# 1. Background mesh
echo "Running blockMesh..."
blockMesh > log.blockMesh 2>&1

# 2. Surface feature extraction
echo "Running surfaceFeatureExtract..."
surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1

# 3. Decompose for parallel snappyHexMesh using mesh-specific dict (32 procs)
echo "Decomposing for ${MESH_NPROCS} processes (mesh only)..."
lustre_safe_cleanup
decomposePar -decomposeParDict system/decomposeParDict.mesh -force > log.decomposePar_mesh 2>&1

# 4. Parallel snappyHexMesh
# Must pass -decomposeParDict so snappyHexMesh validates against 32 procs, not the 128-proc solve dict.
echo "Running snappyHexMesh (${MESH_NPROCS} MPI ranks)..."
mpirun -np "$MESH_NPROCS" snappyHexMesh -parallel -overwrite -decomposeParDict system/decomposeParDict.mesh > log.snappyHexMesh 2>&1

# 5. Reconstruct mesh (tolerate spurious SIGSEGV on exit — OpenFOAM 2306 known issue)
#    Also tolerate FOAM FATAL ERROR when mesh files were written before the crash
#    (Bug 8: polyMeshAdder fails with high proc counts but mesh is already valid)
echo "Reconstructing mesh..."
reconstructParMesh -constant > log.reconstructParMesh 2>&1 || {
    if tail -5 log.reconstructParMesh | grep -q "^End"; then
        echo "WARNING: reconstructParMesh exited non-zero but log shows End. Continuing."
        log_event "warning" "reconstructParMesh" "Non-zero exit but End marker found"
    elif [[ -f constant/polyMesh/owner ]]; then
        echo "WARNING: reconstructParMesh crashed but mesh files exist. Verifying..."
        log_event "warning" "reconstructParMesh" "Crash with mesh-exists fallback"
        checkMesh > log.checkMesh_verify 2>&1
        if grep -q "Mesh OK" log.checkMesh_verify; then
            echo "WARNING: Mesh verified OK despite reconstructParMesh failure. Continuing."
            log_event "warning" "reconstructParMesh" "Mesh verified OK via checkMesh fallback"
        else
            log_error "FOAM_FATAL" "reconstructParMesh" "1" "reconstructParMesh failed AND mesh verification failed" "none"
            echo "ERROR: reconstructParMesh failed AND mesh verification failed for RUN_${RUN_ID}" >&2
            exit 1
        fi
    else
        log_error "FOAM_FATAL" "reconstructParMesh" "1" "reconstructParMesh failed, no mesh files found" "none"
        echo "ERROR: reconstructParMesh failed for RUN_${RUN_ID}" >&2
        exit 1
    fi
}

# 6. Clean processor directories from meshing (Lustre-safe)
lustre_safe_cleanup

# 7. Check mesh quality
echo "Running checkMesh..."
checkMesh > log.checkMesh 2>&1

# Exit non-zero if mesh has fatal errors (triggers --aftercorr dependency)
if grep -q "FAILED" log.checkMesh; then
    log_error "MESH_QUALITY" "checkMesh" "1" "checkMesh reported FAILED" "none"
    echo "ERROR: checkMesh FAILED for RUN_${RUN_ID}"
    exit 1
fi

# 8. Cell-count sanity check against STL face count heuristic
#    Expected: cells ≈ 100–200× STL faces. Flag if outside 50–300× range.
STL_FILE="constant/triSurface/car_surface.stl"
CELL_COUNT=$(grep "cells:" log.checkMesh | awk '{print $2}' || echo "0")
STL_FACES=0
if [[ -f "$STL_FILE" ]]; then
    # Detect ASCII vs binary STL and count faces
    if head -1 "$STL_FILE" | grep -q "^solid"; then
        STL_FACES=$(grep -c "facet normal" "$STL_FILE" 2>/dev/null || echo "0")
    else
        # Binary STL: face count at bytes 80-83 (little-endian uint32)
        STL_FACES=$(python3 -c "import struct; f=open('${STL_FILE}','rb'); f.seek(80); print(struct.unpack('<I',f.read(4))[0])" 2>/dev/null || echo "0")
    fi
fi

if [[ "$STL_FACES" -gt 0 && "$CELL_COUNT" -gt 0 ]]; then
    RATIO=$((CELL_COUNT / STL_FACES))
    echo "Cell count: ${CELL_COUNT}, STL faces: ${STL_FACES}, ratio: ${RATIO}x"
    if [[ "$RATIO" -lt 50 || "$RATIO" -gt 300 ]]; then
        log_event "warning" "checkMesh" "Cell-to-face ratio ${RATIO}x outside expected 50-300x range (cells=${CELL_COUNT}, faces=${STL_FACES})"
        echo "WARNING: Cell-to-face ratio ${RATIO}x outside expected 50-300x range for RUN_${RUN_ID}"
    fi
else
    echo "Cell count: ${CELL_COUNT}, STL faces: ${STL_FACES} (sanity check skipped — missing data)"
fi

log_event "info" "mesh" "mesh_completed (cells=${CELL_COUNT})"
echo "=== Meshing RUN_${RUN_ID} complete at $(date) ==="
exit 0
