#!/bin/bash
# =============================================================================
# OpenFOAM Federico — Full Re-extraction + QA Array Job
#
# Runs process_openfoam_pressure.py + convergence_qa.py for each of the
# 17 extractable batches (7 new + 10 old), outputting to:
#   /cluster/work/fuge/AISCO/wp1_cfd_ai/openfoam_merged/dirty/{batch_short}/
#
# Submit:
#   sbatch extract_qa_all_batches.slurm
#
# Check job:
#   sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Elapsed
#
# After all tasks complete, run:
#   sbatch --dependency=afterok:<JOBID> consolidate_batches.slurm
# =============================================================================

#SBATCH --job-name=OF_Extract_QA
#SBATCH --partition=normal.4h
#SBATCH --account=es_fuge
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-16
#SBATCH --output=/cluster/scratch/%u/scratch_logs/OF_Extract_QA_%A_%a.out
#SBATCH --error=/cluster/scratch/%u/scratch_logs/OF_Extract_QA_%A_%a.err

set -euo pipefail

echo "=== OF_Extract_QA task ${SLURM_ARRAY_TASK_ID} at $(date) ==="
echo "Node: ${SLURMD_NODENAME}  CPUs: ${SLURM_CPUS_PER_TASK}"

# ---------------------------------------------------------------------------
#  Environment
# ---------------------------------------------------------------------------
source "/cluster/home/${USER}/AISCO_code/aisco_env/bin/activate"

REPO="/cluster/home/${USER}/AISCO_code"
OF_BASE="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM_federico"
OLD_BASE="${OF_BASE}/old_simulations"
DIRTY_ROOT="/cluster/work/fuge/AISCO/wp1_cfd_ai/openfoam_merged/dirty"
SCRIPT_DIR="${REPO}/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/dataset/openfoam"

# Ensure log directory exists
mkdir -p "/cluster/scratch/${USER}/scratch_logs"

# ---------------------------------------------------------------------------
#  Batch lookup table (index → batch_short, cases_dir, source_dataset)
#  Must stay in sync with openfoam_batch_config.py
# ---------------------------------------------------------------------------
declare -A BATCH_SHORT
declare -A CASES_DIR
declare -A SOURCE_DATASET

BATCH_SHORT[0]="of021";  CASES_DIR[0]="${OF_BASE}/OPENFOAM_021/CASES_v3";   SOURCE_DATASET[0]="openfoam_federico_021"
BATCH_SHORT[1]="of022";  CASES_DIR[1]="${OF_BASE}/OPENFOAM_022/CASES_v3";   SOURCE_DATASET[1]="openfoam_federico_022"
BATCH_SHORT[2]="of023";  CASES_DIR[2]="${OF_BASE}/OPENFOAM_023/CASES_v3";   SOURCE_DATASET[2]="openfoam_federico_023"
BATCH_SHORT[3]="of024";  CASES_DIR[3]="${OF_BASE}/OPENFOAM_024/CASES_v3";   SOURCE_DATASET[3]="openfoam_federico_024"
BATCH_SHORT[4]="of025";  CASES_DIR[4]="${OF_BASE}/OPENFOAM_025/CASES_v3";   SOURCE_DATASET[4]="openfoam_federico_025"
BATCH_SHORT[5]="of026";  CASES_DIR[5]="${OF_BASE}/OPENFOAM_026/CASES_v3";   SOURCE_DATASET[5]="openfoam_federico_026"
BATCH_SHORT[6]="of027";  CASES_DIR[6]="${OF_BASE}/OPENFOAM_027/CASES_v3";   SOURCE_DATASET[6]="openfoam_federico_027"
BATCH_SHORT[7]="old001"; CASES_DIR[7]="${OLD_BASE}/OPENFOAM_001/CASES_v3";  SOURCE_DATASET[7]="openfoam_federico_old_001"
BATCH_SHORT[8]="old002"; CASES_DIR[8]="${OLD_BASE}/OPENFOAM_002/CASES_v3";  SOURCE_DATASET[8]="openfoam_federico_old_002"
BATCH_SHORT[9]="old003"; CASES_DIR[9]="${OLD_BASE}/OPENFOAM_003/CASES_v3";  SOURCE_DATASET[9]="openfoam_federico_old_003"
BATCH_SHORT[10]="old004"; CASES_DIR[10]="${OLD_BASE}/OPENFOAM_004/CASES_v3"; SOURCE_DATASET[10]="openfoam_federico_old_004"
BATCH_SHORT[11]="old005"; CASES_DIR[11]="${OLD_BASE}/OPENFOAM_005/CASES_v3"; SOURCE_DATASET[11]="openfoam_federico_old_005"
BATCH_SHORT[12]="old006"; CASES_DIR[12]="${OLD_BASE}/OPENFOAM_006/CASES_v3"; SOURCE_DATASET[12]="openfoam_federico_old_006"
BATCH_SHORT[13]="old008"; CASES_DIR[13]="${OLD_BASE}/OPENFOAM_008/CASES_v3"; SOURCE_DATASET[13]="openfoam_federico_old_008"
BATCH_SHORT[14]="old009"; CASES_DIR[14]="${OLD_BASE}/OPENFOAM_009/CASES_v3"; SOURCE_DATASET[14]="openfoam_federico_old_009"
BATCH_SHORT[15]="old012"; CASES_DIR[15]="${OLD_BASE}/OPENFOAM_012/CASES_v3"; SOURCE_DATASET[15]="openfoam_federico_old_012"
BATCH_SHORT[16]="old013"; CASES_DIR[16]="${OLD_BASE}/OPENFOAM_013/CASES_v3"; SOURCE_DATASET[16]="openfoam_federico_old_013"

IDX=${SLURM_ARRAY_TASK_ID}
BSHORT="${BATCH_SHORT[$IDX]}"
BCASES="${CASES_DIR[$IDX]}"
BSRC="${SOURCE_DATASET[$IDX]}"
OUTPUT_DIR="${DIRTY_ROOT}/${BSHORT}"

echo "Batch: ${BSHORT}  |  cases: ${BCASES}  |  source: ${BSRC}"
echo "Output: ${OUTPUT_DIR}"

if [[ ! -d "${BCASES}" ]]; then
    echo "ERROR: cases directory not found: ${BCASES}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
#  Step 1: Extraction
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: Extraction at $(date) ---"
python3 "${SCRIPT_DIR}/process_openfoam_pressure.py" \
    --cases-dir "${BCASES}" \
    --output-dir "${OUTPUT_DIR}" \
    --source-dataset "${BSRC}" \
    --time-step 4500 \
    --workers "${SLURM_CPUS_PER_TASK}"

echo "Extraction done at $(date)"

# ---------------------------------------------------------------------------
#  Step 2: Convergence QA
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: Convergence QA at $(date) ---"
python3 "${SCRIPT_DIR}/convergence_qa.py" \
    --dataset-dir "${OUTPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}/qa" \
    --verbose

echo "QA done at $(date)"
echo "=== Task ${SLURM_ARRAY_TASK_ID} (${BSHORT}) complete at $(date) ==="
