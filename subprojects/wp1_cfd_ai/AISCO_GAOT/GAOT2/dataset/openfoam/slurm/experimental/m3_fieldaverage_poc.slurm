#!/bin/bash
# =============================================================================
# M3 Time-Averaging POC — SLURM Solve Job
#
# Re-runs RUN_010, RUN_021, RUN_058 with fieldAverage enabled.
# Extracts both instantaneous AND time-averaged fields via foamToVTK.
#
# Submit: sbatch --array=10,21,58 m3_fieldaverage_poc.slurm
# =============================================================================

#SBATCH --job-name=M3_FA_POC
#SBATCH --partition=normal.4h
#SBATCH --account=public
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/cluster/scratch/elihuber/scratch_logs/M3_FA_POC_%A_%a.out
#SBATCH --error=/cluster/scratch/elihuber/scratch_logs/M3_FA_POC_%A_%a.err

set -euo pipefail

module purge
module load stack/2024-06 gcc/12.2.0 openmpi/4.1.6
module load openfoam/2306

RUN_ID=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
CASE_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/CASES_v3/RUN_${RUN_ID}"

if [[ ! -d "$CASE_DIR" ]]; then
    echo "ERROR: Case directory $CASE_DIR does not exist."
    exit 1
fi

cd "$CASE_DIR"
echo "=== M3 POC: Solving RUN_${RUN_ID} with fieldAverage ($(date)) ==="

# Verify fieldAverage is present
if ! grep -q 'fieldAverage1' system/controlDict; then
    echo "ERROR: fieldAverage1 not found in controlDict. Run m3_fieldaverage_poc_prepare.sh first."
    exit 1
fi

# Verify no stale solution dirs
for tdir in 1200 3000; do
    if [[ -d "$tdir" ]]; then
        echo "WARNING: stale solution dir $tdir/ found — removing to ensure fresh solve"
        rm -rf "$tdir"
    fi
done

echo "1. Decomposing for 128 MPI processes..."
decomposePar -force > log.decomposePar_m3 2>&1

echo "2. Running simpleFoam (3000 iterations, fieldAverage active from t=2700)..."
mpirun -np 128 simpleFoam -parallel > log.simpleFoam_m3 2>&1

echo "3. Reconstructing latest time step..."
reconstructPar -latestTime > log.reconstructPar_m3 2>&1

echo "4. Reconstructing fieldAverage output (postProcessing/fieldAverage1)..."
# fieldAverage writes pMean and wallShearStressMean to the time directory.
# reconstructPar with -latestTime should include these, but verify.
ls -la "$(ls -1d [0-9]* | sort -n | tail -1)/" | grep -E 'pMean|wallShearStressMean' || \
    echo "  WARNING: pMean/wallShearStressMean not found in latest time dir"

echo "5. Exporting car_surface to VTK (latestTime — includes mean fields if present)..."
foamToVTK -patches '(car_surface)' -latestTime > log.foamToVTK_m3 2>&1

echo "6. Cleaning processor directories..."
rm -rf processor*

echo "=== M3 POC: RUN_${RUN_ID} complete ($(date)) ==="
