#!/bin/bash
# =============================================================================
# OpenFOAM Full Single-Node Pipeline
#
# Runs the complete pipeline on ONE node (max 128 cores):
#   1. Loop subdivision        (~180k verts, 16 workers)  [skippable]
#   2. PLY → STL + case gen    (sequential, lightweight)  [skippable]
#   3. Mesh  snappyHexMesh     (3 cases parallel → 3 × 32 = 96 cores)
#   4. Solve simpleFoam        (3 cases parallel → 3 × 32 = 96 cores)
#   5. Extract .pt + QA        (16 workers)               [skippable]
#
#   Look at the flags!!! you might want to skip or not something
#
# Usage:
#   sbatch slurm/full_process.slurm
#   sbatch subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/dataset/openfoam/slurm/full_process.slurm
#
# watch with cpu core count:
#   watch -n 1 'squeue -u $USER -o "%.12i %.16j %.8T %.6D %.6C %R"'
# =============================================================================

#SBATCH --job-name=OF_FullProcess
#SBATCH --account=es_fuge
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1900M
#SBATCH --output=/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM_federico/OPENFOAM_001/OF_FullProcess_%j.out
#SBATCH --error=/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM_federico/OPENFOAM_001/OF_FullProcess_%j.err

set -euo pipefail

echo "=== OF Full Process: $(date) | Job ${SLURM_JOB_ID} | $(hostname) ==="

# =============================================================================
# Flags — set to 1 to skip that step
# =============================================================================
SKIP_SUBDIVISION=0   # 1 = PLYs already subdivided, skip step 1
SKIP_PREP=0          # 1 = PLY→STL + case gen already done, skip step 2
SKIP_MESH=0          # 1 = cases already meshed, skip step 3
SKIP_SOLVE=0         # 1 = cases already solved, skip step 4
SKIP_EXTRACT=0       # 1 = skip .pt extraction + QA, skip step 5

# =============================================================================
# Paths — edit these before submitting
# =============================================================================
WORK_ROOT="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM_federico/OPENFOAM_001"

PLY_INPUT="$WORK_ROOT/ply_to_simulate"
PLY_SUBDIVIDED="$WORK_ROOT/ply_to_simulate_subdivided"
STL_OUTPUT="$WORK_ROOT/GEOMETRY"
GEO_MANIFEST="$WORK_ROOT/geometry_manifest.json"
CASES_DIR="$WORK_ROOT/CASES_v3"
CASES_MANIFEST="$WORK_ROOT/cases_manifest.json"
OUTPUT_DIR="$WORK_ROOT/processed_euler_openfoam_cp_v050"

PIPE="/cluster/home/$USER/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/dataset/openfoam"
DATASET_DIR="/cluster/home/$USER/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/dataset"

MESH_BATCH=3   # parallel mesh jobs  (3 × 32 = 96 cores)
SOLVE_BATCH=3  # parallel solve jobs (3 × 32 = 96 cores)
# =============================================================================

module purge
module load stack/2024-06 gcc/12.2.0 openmpi/4.1.6
module load openfoam/2306
source /cluster/home/$USER/AISCO_code/aisco_env/bin/activate
export PYTHONPATH="${DATASET_DIR}:${PYTHONPATH:-}"

mkdir -p "$WORK_ROOT/logs"
mkdir -p "$STL_OUTPUT" "$CASES_DIR"

# -----------------------------------------------------------------------
# Step 1: Subdivide
# -----------------------------------------------------------------------
echo ""
if [[ "$SKIP_SUBDIVISION" -eq 1 ]]; then
    echo "=== Step 1: Subdivision SKIPPED ==="
else
    echo "=== Step 1: Subdivision at $(date) ==="
    python "$PIPE/subdivide_all_geoms.py" \
        --input-dir  "$PLY_INPUT" \
        --output-dir "$PLY_SUBDIVIDED" \
        --target-verts 180000 \
        --workers 16
    echo "Subdivision done at $(date)"
fi

# -----------------------------------------------------------------------
# Step 2: Rename PLY → RUN_XXX.ply, PLY→STL, generate cases
# -----------------------------------------------------------------------
echo ""
if [[ "$SKIP_PREP" -eq 1 ]]; then
    echo "=== Step 2: PLY→STL + case generation SKIPPED ==="
else
    echo "=== Step 2: Rename + PLY→STL + case generation at $(date) ==="

    export GEO_MANIFEST PLY_SUBDIVIDED
    python3 - << 'PYEOF'
import json, os
from pathlib import Path

manifest_path = Path(os.environ["GEO_MANIFEST"])
ply_dir = Path(os.environ["PLY_SUBDIVIDED"])

if manifest_path.exists():
    entries = json.loads(manifest_path.read_text())
    max_id = max((int(e["run_id"].split("_")[1]) for e in entries), default=0)
else:
    max_id = 0

new_plys = sorted(p for p in ply_dir.glob("*.ply") if not p.stem.startswith("RUN_"))
for i, ply in enumerate(new_plys):
    new_id = max_id + 1 + i
    target = ply_dir / f"RUN_{new_id:03d}.ply"
    ply.rename(target)
    print(f"Renamed {ply.name} → {target.name}")
PYEOF

    python "$PIPE/ply_to_stl.py" \
        --input-dir  "$PLY_SUBDIVIDED" \
        --output-dir "$STL_OUTPUT" \
        --manifest   "$GEO_MANIFEST" \
        --verbose

    python "$PIPE/generate_cases.py" \
        --manifest       "$GEO_MANIFEST" \
        --output         "$CASES_DIR" \
        --cases-manifest "$CASES_MANIFEST" \
        --vinf 25 \
        --verbose

    echo "Prep done at $(date)"
fi

# Build run list from manifest
RUN_LIST=$(mktemp)
python3 - << PYEOF > "$RUN_LIST"
import json
from pathlib import Path
entries = json.loads(Path("$GEO_MANIFEST").read_text())
ids = sorted(int(e["run_id"].split("_")[1]) for e in entries if e["status"] == "converted")
print("\n".join(f"{i:03d}" for i in ids))
PYEOF

N=$(wc -l < "$RUN_LIST")
echo "Run list: $N cases"
mapfile -t RUN_IDS < "$RUN_LIST"

# -----------------------------------------------------------------------
# Mesh function — 32 MPI ranks, runs as subshell (&)
# -----------------------------------------------------------------------
mesh_case() {
    local RUN_ID
    RUN_ID=$(printf "%03d" "$((10#$1))")
    local CASE="$CASES_DIR/RUN_${RUN_ID}"
    [[ ! -d "$CASE" ]] && echo "[mesh] SKIP $RUN_ID: case dir missing" && return 0
    echo "[mesh] START $RUN_ID at $(date)"
    cd "$CASE"
    blockMesh > log.blockMesh 2>&1
    surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1
    rm -rf processor* 2>/dev/null || true
    decomposePar -decomposeParDict system/decomposeParDict.mesh -force > log.decomposePar_mesh 2>&1
    mpirun --bind-to none -np 32 snappyHexMesh -parallel -overwrite \
        -decomposeParDict system/decomposeParDict.mesh > log.snappyHexMesh 2>&1
    reconstructParMesh -constant > log.reconstructParMesh 2>&1 || {
        if [[ -f constant/polyMesh/owner ]]; then
            checkMesh > log.checkMesh_verify 2>&1 || { echo "[mesh] FAIL $RUN_ID: mesh broken"; return 1; }
            echo "[mesh] $RUN_ID: reconstructParMesh crash but mesh verified OK"
        else
            echo "[mesh] FAIL $RUN_ID: no mesh files"; return 1
        fi
    }
    rm -rf processor* 2>/dev/null || true
    checkMesh > log.checkMesh 2>&1
    echo "[mesh] DONE $RUN_ID at $(date)"
}

# -----------------------------------------------------------------------
# Solve function — 64 MPI ranks with 32-proc retry, runs as subshell (&)
# -----------------------------------------------------------------------
solve_case() {
    local RUN_ID
    RUN_ID=$(printf "%03d" "$((10#$1))")
    local CASE="$CASES_DIR/RUN_${RUN_ID}"
    [[ ! -d "$CASE/constant/polyMesh" ]] && echo "[solve] SKIP $RUN_ID: no mesh" && return 0
    echo "[solve] START $RUN_ID at $(date)"
    cd "$CASE"
    local solved=0
    for NPROCS in 32 16; do
        sed -i "s/numberOfSubdomains.*/numberOfSubdomains ${NPROCS};/" system/decomposeParDict
        rm -rf processor* 2>/dev/null || true
        decomposePar -force > log.decomposePar 2>&1
        mpirun --bind-to none -np "$NPROCS" simpleFoam -parallel > log.simpleFoam 2>&1 && solved=1 && break
        echo "[solve] $RUN_ID: failed at $NPROCS procs, retrying..."
    done
    if [[ $solved -eq 0 ]]; then
        echo "[solve] FAIL $RUN_ID: all proc counts failed"; return 1
    fi
    reconstructPar -latestTime > log.reconstructPar 2>&1
    foamToVTK -patches '(car_surface)' -latestTime > log.foamToVTK 2>&1
    rm -rf processor* 2>/dev/null || true
    echo "[solve] DONE $RUN_ID at $(date)"
}

# -----------------------------------------------------------------------
# Step 3: Mesh (MESH_BATCH cases at a time)
# -----------------------------------------------------------------------
echo ""
if [[ "$SKIP_MESH" -eq 1 ]]; then
    echo "=== Step 3: Meshing SKIPPED ==="
else
    echo "=== Step 3: Meshing ($N cases, $MESH_BATCH parallel) at $(date) ==="
    for (( i=0; i<${#RUN_IDS[@]}; i+=MESH_BATCH )); do
        pids=()
        for (( j=i; j<i+MESH_BATCH && j<${#RUN_IDS[@]}; j++ )); do
            mesh_case "${RUN_IDS[j]}" &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do
            wait "$pid" || echo "WARNING: a mesh job failed (pid $pid)"
        done
    done
    echo "Meshing done at $(date)"
fi

# -----------------------------------------------------------------------
# Step 4: Solve (SOLVE_BATCH cases at a time)
# -----------------------------------------------------------------------
echo ""
if [[ "$SKIP_SOLVE" -eq 1 ]]; then
    echo "=== Step 4: Solving SKIPPED ==="
else
    echo "=== Step 4: Solving ($N cases, $SOLVE_BATCH parallel) at $(date) ==="
    for (( i=0; i<${#RUN_IDS[@]}; i+=SOLVE_BATCH )); do
        pids=()
        for (( j=i; j<i+SOLVE_BATCH && j<${#RUN_IDS[@]}; j++ )); do
            solve_case "${RUN_IDS[j]}" &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do
            wait "$pid" || echo "WARNING: a solve job failed (pid $pid)"
        done
    done
    echo "Solving done at $(date)"
fi

# -----------------------------------------------------------------------
# Step 5: Extract + QA
# -----------------------------------------------------------------------
echo ""
if [[ "$SKIP_EXTRACT" -eq 1 ]]; then
    echo "=== Step 5: Extract + QA SKIPPED ==="
else
    echo "=== Step 5: Extract + QA at $(date) ==="
    python "$PIPE/process_openfoam_pressure.py" \
        --cases-dir  "$CASES_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --time-step 4500 \
        --workers 16 \
        --verbose

    python "$PIPE/convergence_qa.py" \
        --dataset-dir "$OUTPUT_DIR" \
        --output-dir  "$OUTPUT_DIR/qa" \
        --verbose
fi

rm -f "$RUN_LIST"

echo ""
echo "=== Pipeline complete at $(date) ==="
echo "Results: $OUTPUT_DIR/qa/convergence_summary.txt"
