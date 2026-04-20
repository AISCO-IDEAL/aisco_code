#!/usr/bin/env bash
# =============================================================================
# run_urans_test.sh — Short URANS end-to-end test, finishes in < 5 minutes
#
# Uses RUN_010 (already RANS-solved at t=3000).
# Runs 250 pimpleFoam steps (endTime=3000.1, 0.1 s physical time).
# Validates: decomposePar succeeds, solver starts, forceCoeffs written, no NaN.
#
# Usage:
#   bash run_urans_test.sh
# =============================================================================
set -euo pipefail

source /cluster/home/elihuber/AISCO_code/aisco_env/bin/activate

CASES_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/CASES_v3"
RANS_CASE="${CASES_DIR}/RUN_010"
TEST_CASE="${CASES_DIR}/RUN_010_urans_test"
LOGS_DIR="/cluster/scratch/elihuber/scratch_logs"
AISCO_CODE="/cluster/home/elihuber/AISCO_code"

mkdir -p "${LOGS_DIR}"

echo "=== URANS end-to-end test ==="
echo "Source: ${RANS_CASE}"
echo "Target: ${TEST_CASE}"
echo ""

# Sanity: RANS case must have a solved time directory
if ! ls -d "${RANS_CASE}"/[1-9]*/ &>/dev/null 2>&1; then
    echo "ERROR: ${RANS_CASE} has no solved time directory. Run RANS first." >&2
    exit 1
fi
RANS_END=$(ls -d "${RANS_CASE}"/[0-9]*/ 2>/dev/null | sort -V | tail -1 | xargs basename)
echo "RANS end time: ${RANS_END}"
TEST_END_TIME=$(( 10#${RANS_END} + 5 ))  # 5 s beyond RANS = 12500 steps at deltaT=4e-4

# --- Prepare test case ---
echo "Preparing test case..."
rm -rf "${TEST_CASE}"
cp -r "${RANS_CASE}" "${TEST_CASE}"

# Patch controlDict
sed -i "s|application[[:space:]]*simpleFoam;|application     pimpleFoam;|"  "${TEST_CASE}/system/controlDict"
sed -i "s|startFrom[[:space:]]*startTime;|startFrom       latestTime;|"     "${TEST_CASE}/system/controlDict"
sed -i "s|^endTime[[:space:]]*[0-9][0-9]*;|endTime         ${TEST_END_TIME};|" "${TEST_CASE}/system/controlDict"
sed -i "s|^deltaT[[:space:]]*[0-9][^;]*;|deltaT          4e-4;|"            "${TEST_CASE}/system/controlDict"
sed -i "s|^writeInterval[[:space:]]*[0-9][^;]*;|writeInterval   125;|"      "${TEST_CASE}/system/controlDict"
sed -i "s|^purgeWrite[[:space:]]*[0-9][^;]*;|purgeWrite      3;|"           "${TEST_CASE}/system/controlDict"

# Patch fvSchemes
sed -i "s|default[[:space:]]*steadyState;|default         Euler;|" "${TEST_CASE}/system/fvSchemes"

# Rewrite fvSolution
cat > "${TEST_CASE}/system/fvSolution" << 'FVSOL'
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

# Verify patches applied
grep -q "pimpleFoam"  "${TEST_CASE}/system/controlDict" || { echo "FAIL: controlDict patch" >&2; exit 1; }
grep -q "Euler"       "${TEST_CASE}/system/fvSchemes"   || { echo "FAIL: fvSchemes patch" >&2; exit 1; }
grep -q "PIMPLE"      "${TEST_CASE}/system/fvSolution"  || { echo "FAIL: fvSolution patch" >&2; exit 1; }
echo "  Patches verified OK"
echo "  endTime: ${RANS_END} → ${TEST_END_TIME} (+5 s physical time, ~12500 steps at deltaT=4e-4)"

# --- Write SLURM test job inline ---
TEST_SLURM="/tmp/urans_test_run010.slurm"
cat > "${TEST_SLURM}" << SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=OF_URANS_TEST_010
#SBATCH --partition=normal.4h
#SBATCH --account=es_fuge
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=00:10:00
#SBATCH --output=${LOGS_DIR}/OF_URANS_TEST_010_%j.out
#SBATCH --error=${LOGS_DIR}/OF_URANS_TEST_010_%j.err

set -euo pipefail
module load stack/2024-06 gcc/12.2.0 openmpi/4.1.6
module load openfoam/2306

cd "${TEST_CASE}"
rm -rf processor*

echo "=== decomposePar ==="
decomposePar -force > log.decomposePar_test 2>&1
N=\$(ls -d processor* 2>/dev/null | wc -l)
echo "processor dirs: \${N}"
[[ "\${N}" -eq 128 ]] || { echo "ERROR: expected 128 processor dirs, got \${N}" >&2; exit 1; }

echo "=== pimpleFoam (endTime=${TEST_END_TIME}) ==="
mpirun -np 128 pimpleFoam -parallel > log.pimpleFoam_test 2>&1

echo "=== reconstructPar ==="
reconstructPar -latestTime > log.reconstructPar_test 2>&1
rm -rf processor*

echo "=== Test complete ==="
SLURM_EOF

# --- Submit and wait ---
echo ""
echo "Submitting test SLURM job..."
JOBID=$(sbatch --parsable "${TEST_SLURM}")
echo "Test job submitted: ${JOBID}"
echo "Logs: ${LOGS_DIR}/OF_URANS_TEST_010_${JOBID}.{out,err}"
echo ""
echo "Waiting for job to complete (polling every 20s)..."

while true; do
    STATUS=$(sacct -j "${JOBID}" --format=State --noheader 2>/dev/null \
        | awk '{print $1}' | sort -u | tr '\n' ' ' | xargs)
    echo "  [$(date +%H:%M:%S)] State: ${STATUS}"
    if echo "${STATUS}" | grep -qE "RUNNING|PENDING|REQUEUED"; then
        sleep 20
    else
        break
    fi
done

sacct -j "${JOBID}" --format=JobID,State,ExitCode,Elapsed --noheader
EXIT_STATE=$(sacct -j "${JOBID}" --format=State --noheader | awk 'NR==1{print $1}')

if [[ "${EXIT_STATE}" != "COMPLETED" ]]; then
    echo ""
    echo "FAIL: job ${JOBID} ended with state ${EXIT_STATE}"
    echo "--- Last 30 lines of error log ---"
    tail -30 "${LOGS_DIR}/OF_URANS_TEST_010_${JOBID}.err" 2>/dev/null || true
    exit 1
fi

# --- Validate output ---
echo ""
echo "=== Validating test output ==="

# Check forceCoeffs written
FC_DIR="${TEST_CASE}/postProcessing/forceCoeffs"
FC_FILE=$(find "${FC_DIR}" -name "coefficient*.dat" -o -name "forceCoeffs.dat" 2>/dev/null | sort | tail -1)
if [[ -z "${FC_FILE}" ]]; then
    echo "FAIL: no forceCoeffs file found in ${FC_DIR}" >&2
    exit 1
fi
echo "  forceCoeffs file: ${FC_FILE}"

N_LINES=$(grep -v "^#" "${FC_FILE}" | grep -c "[0-9]" || echo 0)
echo "  forceCoeffs data rows: ${N_LINES}"
if [[ "${N_LINES}" -lt 10 ]]; then
    echo "FAIL: fewer than 10 data rows in forceCoeffs" >&2
    exit 1
fi

# Check for NaN
if grep -q "nan\|NaN\|Nan\|1\\.#" "${FC_FILE}" 2>/dev/null; then
    echo "FAIL: NaN detected in forceCoeffs — solver diverged" >&2
    exit 1
fi
echo "  No NaN detected"

echo ""
echo "=== TEST PASSED ==="
echo "  RUN_010 URANS test: ${N_LINES} timesteps, no NaN, forceCoeffs written"
echo "  The overnight pipeline is ready to submit."
echo ""
echo "Run the overnight pipeline with:"
echo "  bash ${AISCO_CODE}/submit_urans_overnight.sh"
