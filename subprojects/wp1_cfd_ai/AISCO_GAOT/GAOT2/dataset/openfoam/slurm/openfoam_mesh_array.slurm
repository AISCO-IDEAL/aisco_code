#!/bin/bash
# =============================================================================
# OpenFOAM Mesh Array Job — snappyHexMesh for all cases
#
# Runs blockMesh + surfaceFeatureExtract + snappyHexMesh + checkMesh
# for each case in parallel via SLURM array.
#
# v050: 8 BL layers, expansionRatio 1.15, input geometry from openfoam_geometry.
# Cases live in CASES_v3 (CASES_V2 preserved as v0.4.0 baseline).
# =============================================================================

#SBATCH --job-name=OF_Mesh
#SBATCH --partition=normal.4h
#SBATCH --account=es_fuge
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1900M
#SBATCH --array=1-223%20
#SBATCH --output=/cluster/scratch/elihuber/scratch_logs/OF_Mesh_%A_%a.out
#SBATCH --error=/cluster/scratch/elihuber/scratch_logs/OF_Mesh_%A_%a.err

set -euo pipefail

# Load OpenFOAM
# Note: Euler's spack-installed module sets the full environment at load time.
# No bashrc sourcing needed.
module purge
module load stack/2024-06 gcc/12.2.0 openmpi/4.1.6
module load openfoam/2306

# Determine case directory from run-list file or array task ID
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

# 1. Background mesh
echo "Running blockMesh..."
blockMesh > log.blockMesh 2>&1

# 2. Surface feature extraction
echo "Running surfaceFeatureExtract..."
surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1

# 3. Decompose for parallel snappyHexMesh (128 procs, same as solve)
echo "Decomposing for 128 processes..."
rm -rf processor*
decomposePar -force > log.decomposePar_mesh 2>&1

# 4. Parallel snappyHexMesh
echo "Running snappyHexMesh (128 MPI ranks)..."
mpirun -np 128 snappyHexMesh -parallel -overwrite > log.snappyHexMesh 2>&1

# 5. Reconstruct mesh (tolerate spurious SIGSEGV on exit — OpenFOAM 2306 known issue)
#    Also tolerate FOAM FATAL ERROR when mesh files were written before the crash
#    (Bug 8: polyMeshAdder fails with high proc counts but mesh is already valid)
echo "Reconstructing mesh..."
reconstructParMesh -constant > log.reconstructParMesh 2>&1 || {
    if tail -5 log.reconstructParMesh | grep -q "^End"; then
        echo "WARNING: reconstructParMesh exited non-zero but log shows End. Continuing."
    elif [[ -f constant/polyMesh/owner ]]; then
        echo "WARNING: reconstructParMesh crashed but mesh files exist. Verifying..."
        checkMesh > log.checkMesh_verify 2>&1
        if grep -q "Mesh OK" log.checkMesh_verify; then
            echo "WARNING: Mesh verified OK despite reconstructParMesh failure. Continuing."
        else
            echo "ERROR: reconstructParMesh failed AND mesh verification failed for RUN_${RUN_ID}" >&2
            exit 1
        fi
    else
        echo "ERROR: reconstructParMesh failed for RUN_${RUN_ID}" >&2
        exit 1
    fi
}

# 6. Clean processor directories from meshing
# Tolerate "Stale file handle" errors on parallel filesystems (Lustre/GPFS)
rm -rf processor* 2>/dev/null || true
# Retry once after a brief pause if stale handles linger
if ls -d processor* >/dev/null 2>&1; then
    sleep 5
    rm -rf processor* 2>/dev/null || true
fi

# 7. Check mesh quality
echo "Running checkMesh..."
checkMesh > log.checkMesh 2>&1

# Exit non-zero if mesh has fatal errors (triggers --aftercorr dependency)
if grep -q "FAILED" log.checkMesh; then
    echo "ERROR: checkMesh FAILED for RUN_${RUN_ID}"
    exit 1
fi

echo "=== Meshing RUN_${RUN_ID} complete at $(date) ==="
exit 0
