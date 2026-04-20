#!/bin/bash
# =============================================================================
# URANS Benchmark Job — pimpleFoam for RUN_001_urans
#
# Plan 036: Single-case URANS vs RANS comparison experiment.
# Restarts from converged RANS solution at t=3000, runs to t=3000.48 (0.48 s
# physical time = 2 flow-through times at 25 m/s over 6 m car).
# Produces time-averaged Cd/Cl for comparison against RANS baseline.
# =============================================================================

#SBATCH --job-name=OF_URANS_001
#SBATCH --partition=normal.4h
#SBATCH --account=public
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/cluster/scratch/elihuber/scratch_logs/OF_URANS_001_%j.out
#SBATCH --error=/cluster/scratch/elihuber/scratch_logs/OF_URANS_001_%j.err

set -euo pipefail

CASE_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/CASES_v3/RUN_001_urans"

echo "=== URANS RUN_001 started on $(hostname) at $(date) ==="
echo "Case: $CASE_DIR"

# Load OpenFOAM
module purge
module load stack/2024-06 gcc/12.2.0 openmpi/4.1.6
module load openfoam/2306

cd "$CASE_DIR"

# Sanity check: processor dirs must exist (decomposePar was run in M1)
N_PROC=$(ls -d processor* 2>/dev/null | wc -l)
if [[ "$N_PROC" -ne 128 ]]; then
    echo "ERROR: Expected 128 processor dirs, found $N_PROC. Run decomposePar -force first."
    exit 1
fi
echo "Processor directories: $N_PROC (OK)"

# Run pimpleFoam
echo "Starting pimpleFoam at $(date)..."
mpirun -np 128 pimpleFoam -parallel > log.pimpleFoam 2>&1
FOAM_EXIT=$?

echo "pimpleFoam finished at $(date) with exit code $FOAM_EXIT"

if [[ $FOAM_EXIT -ne 0 ]]; then
    echo "ERROR: pimpleFoam exited with code $FOAM_EXIT. Check log.pimpleFoam."
    exit $FOAM_EXIT
fi

# Quick summary of final timestep
echo "--- Last 20 lines of log.pimpleFoam ---"
tail -20 log.pimpleFoam

echo "=== URANS RUN_001 complete at $(date) ==="
exit 0
