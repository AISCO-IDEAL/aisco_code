#!/bin/bash
# =============================================================================
# OpenFOAM Solve Array Job — simpleFoam for all cases
#
# Runs decomposePar + simpleFoam (up to 3000 iterations with residual-based early stop) + reconstructPar
# + foamToVTK post-processing for each case.
#
# Submit with --dependency=aftercorr:<mesh_job_id> to only solve
# cases where meshing succeeded.
# v040: cases in CASES_V2, target 3000 iterations, 8 BL layers.
# =============================================================================

#SBATCH --job-name=OF_Solve
#SBATCH --partition=normal.4h
#SBATCH --account=es_fuge
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1900M
#SBATCH --array=1-223%20
#SBATCH --output=/cluster/scratch/elihuber/scratch_logs/OF_Solve_%A_%a.out
#SBATCH --error=/cluster/scratch/elihuber/scratch_logs/OF_Solve_%A_%a.err

set -euo pipefail

# Load OpenFOAM
# Note: Euler's spack-installed module sets the full environment at load time.
# No bashrc sourcing needed.
module purge
module load stack/2024-06 gcc/12.2.0 openmpi/4.1.6
module load openfoam/2306

# Optional live progress stream in the SLURM stdout log.
# Enable with: sbatch --export=ALL,SHOW_ITERATIONS=1[,ITERATION_LOG_INTERVAL=10] openfoam_solve_array.slurm
SHOW_ITERATIONS=${SHOW_ITERATIONS:-0}
ITERATION_LOG_INTERVAL=${ITERATION_LOG_INTERVAL:-1}

if ! [[ "$SHOW_ITERATIONS" =~ ^(0|1)$ ]]; then
    echo "ERROR: SHOW_ITERATIONS must be 0 or 1. Set --export=ALL,SHOW_ITERATIONS=1 to enable progress logging."
    exit 1
fi

if ! [[ "$ITERATION_LOG_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ITERATION_LOG_INTERVAL must be a positive integer. Example: --export=ALL,SHOW_ITERATIONS=1,ITERATION_LOG_INTERVAL=10"
    exit 1
fi

# Determine case directory from array task ID
RUN_ID=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
CASE_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/CASES_V2/RUN_${RUN_ID}"

if [[ ! -d "$CASE_DIR" ]]; then
    echo "ERROR: Case directory $CASE_DIR does not exist. Skipping."
    exit 0
fi

if [[ ! -d "$CASE_DIR/constant/polyMesh" ]]; then
    echo "ERROR: No mesh found for $RUN_ID (mesh likely failed). Skipping."
    exit 0
fi

cd "$CASE_DIR"

echo "=== Solving RUN_${RUN_ID} at $(date) ==="

# 1. Regenerate decomposeParDict for 128 processes (mesh only had 16)
# Remove old processor dirs if they exist
rm -rf processor* > /dev/null 2>&1

cat > system/decomposeParDict << 'EOF'
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      decomposeParDict;
}

numberOfSubdomains 128;

method          scotch;
EOF

# 2. Decompose for parallel (solve uses 128 ranks)
echo "Decomposing for 128 processes..."
decomposePar -force > log.decomposePar 2>&1

# 2. Run simpleFoam (up to 3000 iterations, steady RANS k-omega SST, 8 BL layers)
echo "Running simpleFoam (up to 3000 iterations, 128 MPI ranks)..."
if [[ "$SHOW_ITERATIONS" == "1" ]]; then
    echo "Streaming 'Time =' progress lines to SLURM stdout every ${ITERATION_LOG_INTERVAL} iteration(s)."
    mpirun -np 128 simpleFoam -parallel 2>&1 \
        | tee log.simpleFoam \
        | stdbuf -oL awk -v interval="$ITERATION_LOG_INTERVAL" '
            /^Time = / {
                iteration_count += 1
                if (iteration_count % interval == 0) {
                    print strftime("[%Y-%m-%d %H:%M:%S]"), $0
                    fflush()
                }
            }
        '
else
    mpirun -np 128 simpleFoam -parallel > log.simpleFoam 2>&1
fi

# 3. Reconstruct latest time step
echo "Reconstructing solution..."
reconstructPar -latestTime > log.reconstructPar 2>&1

# 4. Export car_surface patch to VTK for Python post-processing
echo "Running foamToVTK for car_surface patch..."
foamToVTK -patches '(car_surface)' -latestTime > log.foamToVTK 2>&1

# 5. Clean up processor directories (save storage)
rm -rf processor*

echo "=== Solving RUN_${RUN_ID} complete at $(date) ==="
exit 0
