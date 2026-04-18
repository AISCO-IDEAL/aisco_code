#!/usr/bin/env bash
# =============================================================================
# prepare_urans_batch.sh — Copy RANS cases to _urans and apply 3 config patches
#
# For each requested run ID:
#   1. Copy CASES_V2/RUN_XXX → CASES_V2/RUN_XXX_urans  (skip if already done)
#   2. Patch system/controlDict  (simpleFoam→pimpleFoam, deltaT, endTime, writes)
#   3. Patch system/fvSchemes    (steadyState→Euler)
#   4. Replace system/fvSolution (SIMPLE→PIMPLE block)
#
# Does NOT run decomposePar — that happens inside the SLURM array task.
#
# Usage:
#   bash prepare_urans_batch.sh                # prepare all solved cases
#   bash prepare_urans_batch.sh --run-ids 10 19 21
#   bash prepare_urans_batch.sh --run-ids 10   # single case (test)
# =============================================================================
set -euo pipefail

CASES_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/CASES_V2"
URANS_END_OFFSET=5.0   # seconds of physical time beyond RANS endTime

# ---- argument parsing -------------------------------------------------------
RUN_IDS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-ids)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                RUN_IDS+=("$1")
                shift
            done
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            echo "Usage: $0 [--run-ids ID1 ID2 ...]" >&2
            exit 1
            ;;
    esac
done

# If no --run-ids specified, collect all cases that have a 3000/ directory
if [[ ${#RUN_IDS[@]} -eq 0 ]]; then
    echo "No --run-ids given; discovering all solved RANS cases in ${CASES_DIR}..."
    for case_dir in "${CASES_DIR}"/RUN_[0-9][0-9][0-9]/; do
        case_name="$(basename "${case_dir}")"
        # Skip _urans dirs (exact 3-digit glob above already excludes them, but be safe)
        [[ "${case_name}" =~ _urans ]] && continue
        # Only include cases that have a latestTime (3000/) directory
        if ls "${case_dir}"[0-9]*/ &>/dev/null 2>&1; then
            latest=$(ls -d "${case_dir}"[0-9]*/ 2>/dev/null | sort -t/ -k1 -V | tail -1)
            if [[ -d "${latest}" ]]; then
                id_str="${case_name#RUN_}"
                id_num=$(( 10#${id_str} ))     # base-10 explicit; strips leading zeros
                RUN_IDS+=("${id_num}")
            fi
        fi
    done
    echo "Found ${#RUN_IDS[@]} solved cases."
fi

# ---- helpers ----------------------------------------------------------------
patch_control_dict() {
    local ctrl="$1"
    local rans_end
    rans_end=$(grep -E '^[[:space:]]*endTime' "${ctrl}" | awk '{print $2}' | tr -d ';')
    local urans_end
    urans_end=$(python3 -c "print(int(float('${rans_end}') + ${URANS_END_OFFSET}))")

    sed -i "s|application[[:space:]]*simpleFoam;|application     pimpleFoam;|" "${ctrl}"
    sed -i "s|startFrom[[:space:]]*startTime;|startFrom       latestTime;|"    "${ctrl}"
    sed -i "s|^endTime[[:space:]]*[0-9][0-9]*;|endTime         ${urans_end};|" "${ctrl}"
    sed -i "s|^deltaT[[:space:]]*[0-9][^;]*;|deltaT          4e-4;|"          "${ctrl}"
    sed -i "s|^writeInterval[[:space:]]*[0-9][^;]*;|writeInterval   125;|"    "${ctrl}"
    sed -i "s|^purgeWrite[[:space:]]*[0-9][^;]*;|purgeWrite      3;|"         "${ctrl}"
}

patch_fv_schemes() {
    local fvs="$1"
    sed -i "s|default[[:space:]]*steadyState;|default         Euler;|" "${fvs}"
}

write_fv_solution() {
    local fvs="$1"
    cat > "${fvs}" << 'FVSOL'
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-7;
        relTol          0.01;
        smoother        GaussSeidel;
        nPreSweeps      0;
        nPostSweeps     2;
        cacheAgglomeration on;
        agglomerator    faceAreaPair;
        nCellsInCoarsestLevel 10;
        mergeLevels     1;
    }

    pFinal
    {
        $p;
        relTol          0;
    }

    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }

    "(U|k|omega)Final"
    {
        $U;
        relTol          0;
    }
}

PIMPLE
{
    nOuterCorrectors    2;
    nCorrectors         2;
    nNonOrthogonalCorrectors 1;
}
FVSOL
}

# ---- main loop --------------------------------------------------------------
n_copied=0
n_skipped=0
n_failed=0

for raw_id in "${RUN_IDS[@]}"; do
    run_id=$(printf "RUN_%03d" "${raw_id}")
    rans_dir="${CASES_DIR}/${run_id}"
    urans_dir="${CASES_DIR}/${run_id}_urans"

    echo "--- Processing ${run_id} ---"

    # Sanity: RANS source must exist
    if [[ ! -d "${rans_dir}" ]]; then
        echo "  SKIP: RANS source ${rans_dir} not found"
        (( n_skipped++ )) || true
        continue
    fi

    # Skip if _urans dir already exists and is patched (contains pimpleFoam)
    if [[ -d "${urans_dir}" ]]; then
        if grep -q "pimpleFoam" "${urans_dir}/system/controlDict" 2>/dev/null; then
            echo "  SKIP: ${run_id}_urans already prepared"
            (( n_skipped++ )) || true
            continue
        else
            echo "  WARN: ${run_id}_urans exists but not patched; re-patching in place"
        fi
    else
        echo "  Copying ${run_id} → ${run_id}_urans ..."
        if ! cp -r "${rans_dir}" "${urans_dir}"; then
            echo "  FAIL: cp failed for ${run_id}" >&2
            (( n_failed++ )) || true
            continue
        fi
    fi

    # Apply the 3 patches
    if ! patch_control_dict "${urans_dir}/system/controlDict"; then
        echo "  FAIL: controlDict patch failed for ${run_id}" >&2
        (( n_failed++ )) || true
        continue
    fi
    patch_fv_schemes  "${urans_dir}/system/fvSchemes"
    write_fv_solution "${urans_dir}/system/fvSolution"

    # Verification
    if ! grep -q "pimpleFoam"  "${urans_dir}/system/controlDict" || \
       ! grep -q "Euler"       "${urans_dir}/system/fvSchemes"   || \
       ! grep -q "PIMPLE"      "${urans_dir}/system/fvSolution"; then
        echo "  FAIL: post-patch verification failed for ${run_id}" >&2
        (( n_failed++ )) || true
        continue
    fi

    echo "  OK: ${run_id}_urans prepared"
    (( n_copied++ )) || true
done

echo ""
echo "=== prepare_urans_batch.sh complete: ${n_copied} prepared, ${n_skipped} skipped, ${n_failed} failed ==="

# Write list of all ready _urans cases for the SLURM array job
READY_LIST="/cluster/home/elihuber/AISCO_code/urans_ready_cases.txt"
echo "Writing ready-case list to ${READY_LIST} ..."
> "${READY_LIST}"
for urans_dir in "${CASES_DIR}"/RUN_[0-9]*_urans/; do
    case_name="$(basename "${urans_dir%/}")"
    run_id="${case_name%_urans}"
    if grep -q "pimpleFoam" "${urans_dir}/system/controlDict" 2>/dev/null; then
        echo "${run_id}" >> "${READY_LIST}"
    fi
done
sort -t_ -k2 -n "${READY_LIST}" -o "${READY_LIST}"
echo "Ready cases: $(wc -l < "${READY_LIST}")"

[[ "${n_failed}" -eq 0 ]]
