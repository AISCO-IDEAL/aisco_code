#!/usr/bin/env bash
# =============================================================================
# submit_urans_overnight.sh — Full PLY-to-URANS pipeline for all 234 geometries
#
# Pipeline (all 234 from scratch):
#   1. ply_to_stl.py        — convert all 234 PLYs → STL, rebuild manifest
#   2. generate_cases.py    — create ALL 234 case dirs (overwrites existing)
#   3. submit_all.sh        — mesh + RANS for all 234 cases (1-241, gaps ok)
#   4. [after RANS done]    — prepare_urans_batch.sh + run_urans_batch_array
#
# Usage:
#   bash submit_urans_overnight.sh            # submit everything now
#   bash submit_urans_overnight.sh --dry-run  # print commands without running
#
# Prerequisites:
#   source AISCO_code/aisco_env/bin/activate
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AISCO_CODE="${SCRIPT_DIR}"
CASES_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/CASES_V2"
GEOMETRY_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/openfoam_geometry"
STL_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/GEOMETRY_V2"
MANIFEST_V2="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/geometry_manifest_v2.json"
CASES_MANIFEST="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/cases_manifest_v2.json"
PLY_TO_STL="${AISCO_CODE}/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/dataset/openfoam/ply_to_stl.py"
GENERATE_CASES="${AISCO_CODE}/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/dataset/openfoam/generate_cases.py"
SUBMIT_ALL="${AISCO_CODE}/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/dataset/openfoam/slurm/submit_all.sh"
PREPARE_URANS="${AISCO_CODE}/prepare_urans_batch.sh"
URANS_SLURM="${AISCO_CODE}/run_urans_batch_array.slurm"
LOGS_DIR="/cluster/scratch/elihuber/scratch_logs"

# IDs 1-241 cover all PLY files; mesh script exits 0 for missing dirs (gap 224-230)
ARRAY_RANGE="1-241"
MAX_PARALLEL=50

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "=== DRY RUN MODE ==="
fi

mkdir -p "${LOGS_DIR}"

echo ""
echo "=== URANS Overnight Pipeline — All 234 Geometries ==="
echo "$(date)"
echo ""

# ---------------------------------------------------------------------------
# Step 1: PLY → STL conversion for all 234 geometries
# ---------------------------------------------------------------------------
echo "--- Step 1: ply_to_stl.py (all 234 geometries) ---"
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "  [DRY] python3 ${PLY_TO_STL} \\"
    echo "        --input-dir ${GEOMETRY_DIR} \\"
    echo "        --output-dir ${STL_DIR} \\"
    echo "        --manifest ${MANIFEST_V2}"
else
    python3 "${PLY_TO_STL}" \
        --input-dir "${GEOMETRY_DIR}" \
        --output-dir "${STL_DIR}" \
        --manifest "${MANIFEST_V2}"
    N_CONV=$(python3 -c "
import json, pathlib
d = json.load(open('${MANIFEST_V2}'))
print(len([e for e in d if e.get('status') == 'converted']))
")
    echo "  Manifest has ${N_CONV} converted entries."
fi

# ---------------------------------------------------------------------------
# Step 2: generate_cases.py for ALL 234 geometries
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: generate_cases.py (all 234 cases) ---"
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "  [DRY] python3 ${GENERATE_CASES} \\"
    echo "        --manifest ${MANIFEST_V2} \\"
    echo "        --output ${CASES_DIR} \\"
    echo "        --cases-manifest ${CASES_MANIFEST}"
else
    python3 "${GENERATE_CASES}" \
        --manifest "${MANIFEST_V2}" \
        --output "${CASES_DIR}" \
        --cases-manifest "${CASES_MANIFEST}"
    N_CASES=$(ls -d "${CASES_DIR}"/RUN_[0-9]*/ 2>/dev/null | grep -v "_urans" | wc -l)
    echo "  Case dirs created: ${N_CASES}"
fi

# ---------------------------------------------------------------------------
# Step 3: Submit mesh + RANS for all cases (1-241, gaps handled by exit 0)
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: Submit mesh + RANS (array ${ARRAY_RANGE}, %${MAX_PARALLEL} parallel) ---"
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "  [DRY] bash ${SUBMIT_ALL} ${ARRAY_RANGE} ${MAX_PARALLEL}"
    RANS_JOBID="<RANS_JOBID>"
else
    SUBMIT_OUTPUT=$(bash "${SUBMIT_ALL}" "${ARRAY_RANGE}" "${MAX_PARALLEL}")
    echo "${SUBMIT_OUTPUT}"
    RANS_JOBID=$(echo "${SUBMIT_OUTPUT}" | grep "Solve job submitted:" | awk '{print $NF}')
    echo ""
    echo "  RANS solve job ID: ${RANS_JOBID}"
fi

# ---------------------------------------------------------------------------
# Step 4: After RANS finishes — prepare + submit URANS via trigger job
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 4: Queue URANS trigger (dependency: afterok:${RANS_JOBID}) ---"

# Write the trigger script that will run after RANS completes
TRIGGER_SCRIPT="${AISCO_CODE}/urans_trigger.sh"
cat > "${TRIGGER_SCRIPT}" << 'TRIGGER_EOF'
#!/usr/bin/env bash
# Called by SLURM after RANS array job completes
set -euo pipefail
source /cluster/home/elihuber/AISCO_code/aisco_env/bin/activate
cd /cluster/home/elihuber/AISCO_code

echo "=== URANS trigger: preparing all solved cases at $(date) ==="
bash prepare_urans_batch.sh

N=$(wc -l < urans_ready_cases.txt)
echo "Submitting URANS for ${N} cases..."
URANS_JOBID=$(sbatch --parsable --array="1-${N}%50" run_urans_batch_array.slurm)
echo "URANS array job submitted: ${URANS_JOBID}"

echo "=== URANS trigger done at $(date) ==="
TRIGGER_EOF
chmod +x "${TRIGGER_SCRIPT}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "  [DRY] sbatch --dependency=afterok:${RANS_JOBID} \\"
    echo "        --wrap=\"bash ${TRIGGER_SCRIPT}\""
else
    TRIGGER_JOBID=$(sbatch --parsable \
        --dependency="afterok:${RANS_JOBID}" \
        --job-name="OF_URANS_TRIGGER" \
        --partition=normal.4h \
        --account=es_fuge \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem-per-cpu=4000M \
        --time=00:30:00 \
        --output="${LOGS_DIR}/OF_URANS_TRIGGER_%j.out" \
        --error="${LOGS_DIR}/OF_URANS_TRIGGER_%j.err" \
        --wrap="bash ${TRIGGER_SCRIPT}")
    echo "  URANS trigger job submitted: ${TRIGGER_JOBID}"
    echo "  (runs automatically after RANS job ${RANS_JOBID} completes)"
fi

echo ""
echo "=== Pipeline submitted at $(date) ==="
echo ""
echo "Dependency chain:"
echo "  Mesh array  → Solve array (afterok:mesh)"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    echo "  Solve array (${RANS_JOBID}) → URANS trigger (${TRIGGER_JOBID:-dry}) (afterok:rans)"
fi
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Post-process after URANS completes:"
echo "  source aisco_env/bin/activate"
echo "  python3 compare_rans_urans.py --all-cases"
