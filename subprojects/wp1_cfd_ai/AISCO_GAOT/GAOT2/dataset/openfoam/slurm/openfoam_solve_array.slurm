#!/bin/bash
# =============================================================================
# OpenFOAM Solve Array Job — simpleFoam for all cases
#
# Runs decomposePar + simpleFoam (up to 4500 iterations with residual-based
# early stop) + reconstructPar + foamToVTK post-processing for each case.
#
# Features:
#   - Reads proc count from system/decomposeParDict (not hardcoded)
#   - 3-tier MPI crash retry cascade: NPROCS → NPROCS/2 → 32
#   - Lustre stale-file-handle tolerant cleanup
#   - Structured JSON error logging to stderr
#
# Submit with --dependency=aftercorr:<mesh_job_id> to only solve
# cases where meshing succeeded.
# v051: cases in CASES_v3, target 4500 iterations, 8 BL layers.
# =============================================================================

#SBATCH --job-name=OF_Solve
#SBATCH --partition=normal.4h
#SBATCH --account=es_fuge
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1900M
#SBATCH --array=1-234%2
#SBATCH --output=/cluster/scratch/%u/scratch_logs/OF_Solve_%A_%a.out
#SBATCH --error=/cluster/scratch/%u/scratch_logs/OF_Solve_%A_%a.err

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
    # Use 10# prefix to force decimal — bash printf treats leading-zero strings as octal
    RUN_ID=$(printf "%03d" "$((10#${RUN_ID}))")
else
    RUN_ID=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
fi
CASE_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM_federico/CASES_v3/RUN_${RUN_ID}"

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
log_event "info" "solve" "solve_started"

# --- Read processor count from decomposeParDict ------------------------------

NPROCS=$(grep "numberOfSubdomains" system/decomposeParDict | awk '{print $2}' | tr -d ';')
if [[ -z "$NPROCS" || "$NPROCS" -lt 1 ]]; then
    log_error "CONFIG_ERROR" "solve" "1" "Cannot read numberOfSubdomains from decomposeParDict" "none"
    echo "ERROR: Cannot read numberOfSubdomains from system/decomposeParDict for RUN_${RUN_ID}" >&2
    exit 1
fi
echo "Proc count from decomposeParDict: ${NPROCS}"

# --- Build retry tiers: NPROCS → NPROCS/2 → 32 (deduplicated) ---------------

RETRY_TIERS=("$NPROCS")
HALF=$((NPROCS / 2))
if [[ "$HALF" -ge 32 && "$HALF" -ne "$NPROCS" ]]; then
    RETRY_TIERS+=("$HALF")
fi
if [[ 32 -lt "${RETRY_TIERS[-1]}" ]]; then
    RETRY_TIERS+=(32)
fi

# --- Solve with retry cascade ------------------------------------------------

SOLVE_SUCCESS=0
ACTUAL_PROCS="$NPROCS"
SOLVE_EXIT=0

for ATTEMPT_PROCS in "${RETRY_TIERS[@]}"; do
    echo "--- Solve attempt with ${ATTEMPT_PROCS} MPI ranks ---"
    log_event "info" "solve" "attempting solve at ${ATTEMPT_PROCS} procs"

    # Update decomposeParDict if proc count changed from original
    if [[ "$ATTEMPT_PROCS" -ne "$NPROCS" ]]; then
        sed -i "s/numberOfSubdomains.*/numberOfSubdomains ${ATTEMPT_PROCS};/" system/decomposeParDict
        log_event "warning" "solve" "solve_retry_reduced_procs to ${ATTEMPT_PROCS}"
    fi

    # Clean old processor dirs (Lustre-safe)
    lustre_safe_cleanup

    # Decompose for parallel
    echo "Decomposing for ${ATTEMPT_PROCS} processes..."
    decomposePar -force > log.decomposePar 2>&1

    # Run simpleFoam — capture exit code without triggering errexit
    echo "Running simpleFoam (up to 4500 iterations, ${ATTEMPT_PROCS} MPI ranks)..."
    SOLVE_EXIT=0
    if [[ "$SHOW_ITERATIONS" == "1" ]]; then
        echo "Streaming 'Time =' progress lines to SLURM stdout every ${ITERATION_LOG_INTERVAL} iteration(s)."
        set +o pipefail
        mpirun -np "$ATTEMPT_PROCS" simpleFoam -parallel 2>&1 \
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
        SOLVE_EXIT=${PIPESTATUS[0]}
        set -o pipefail
    else
        mpirun -np "$ATTEMPT_PROCS" simpleFoam -parallel > log.simpleFoam 2>&1 || SOLVE_EXIT=$?
    fi

    if [[ "$SOLVE_EXIT" -eq 0 ]]; then
        SOLVE_SUCCESS=1
        ACTUAL_PROCS="$ATTEMPT_PROCS"
        log_event "info" "solve" "solve_completed at ${ATTEMPT_PROCS} procs"
        break
    fi

    # Check if this is a retryable MPI crash (common exit codes: 1, 11, 134, 139)
    LAST_ERROR=$(tail -5 log.simpleFoam 2>/dev/null | tr '\n' ' ' | tr '"' "'" || echo "unknown")
    case "$SOLVE_EXIT" in
        1|11|134|139)
            log_error "MPI_CRASH" "solve" "$SOLVE_EXIT" "$LAST_ERROR" "reduced_procs"
            echo "WARNING: Solve failed at ${ATTEMPT_PROCS} procs (exit ${SOLVE_EXIT}). Will retry at reduced count."
            ;;
        *)
            log_error "SOLVE_FAIL" "solve" "$SOLVE_EXIT" "$LAST_ERROR" "none"
            echo "ERROR: Solve failed at ${ATTEMPT_PROCS} procs with non-retryable exit code ${SOLVE_EXIT}" >&2
            exit 1
            ;;
    esac
done

if [[ "$SOLVE_SUCCESS" -ne 1 ]]; then
    log_error "MPI_CRASH" "solve" "$SOLVE_EXIT" "All retry tiers exhausted (${RETRY_TIERS[*]})" "none"
    echo "ERROR: Solve failed for RUN_${RUN_ID} after all retry tiers (${RETRY_TIERS[*]})" >&2
    exit 1
fi

if [[ "$ACTUAL_PROCS" -ne "$NPROCS" ]]; then
    echo "NOTE: Solve succeeded at ${ACTUAL_PROCS} procs (original: ${NPROCS})"
fi

# --- Post-processing ---------------------------------------------------------

# Reconstruct latest time step
echo "Reconstructing solution..."
reconstructPar -latestTime > log.reconstructPar 2>&1

# Export car_surface patch to VTK for Python post-processing
echo "Running foamToVTK for car_surface patch..."
foamToVTK -patches '(car_surface)' -latestTime > log.foamToVTK 2>&1

# Clean up processor directories (save storage, Lustre-safe)
lustre_safe_cleanup

log_event "info" "solve" "solve_pipeline_complete at ${ACTUAL_PROCS} procs"
echo "=== Solving RUN_${RUN_ID} complete at $(date) ==="
exit 0
