#!/bin/bash
# =============================================================================
# M3 Time-Averaging POC — Prepare 3 test cases for fieldAverage re-run
#
# For each of RUN_010, RUN_021, RUN_058:
#   1. Add fieldAverage function object (timeStart=2700, averages p + wallShearStress)
#   2. Remove existing solution time dirs (1200, 3000) so solver runs fresh from t=0
#
# Run this once from the AISCO_code directory BEFORE submitting the SLURM array.
# =============================================================================

set -euo pipefail

CASES_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/CASES_v3"
POC_RUNS=(010 021 058)

# fieldAverage block to append inside the functions{} section
FIELD_AVG_BLOCK='
    fieldAverage1
    {
        type            fieldAverage;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        timeStart       2700;

        fields
        (
            p
            {
                mean        on;
                prime2Mean  off;
                base        time;
            }
            wallShearStress
            {
                mean        on;
                prime2Mean  off;
                base        time;
            }
        );
    }'

for run_id in "${POC_RUNS[@]}"; do
    CASE="$CASES_DIR/RUN_${run_id}"
    CONTROL_DICT="$CASE/system/controlDict"

    echo "=== Preparing RUN_${run_id} ==="

    if [[ ! -d "$CASE" ]]; then
        echo "  ERROR: $CASE not found — skipping"
        continue
    fi

    # --- 1. Check fieldAverage not already present ---
    if grep -q 'fieldAverage1' "$CONTROL_DICT"; then
        echo "  fieldAverage1 already present — skipping block injection"
    else
        # Insert fieldAverage block before the closing } of the functions block.
        # The closing } of functions{} is the last } in the file.
        # Use Python for reliable multi-line insertion.
        python3 - "$CONTROL_DICT" "$FIELD_AVG_BLOCK" << 'PY'
import sys
from pathlib import Path

ctrl = Path(sys.argv[1])
block = sys.argv[2]
text = ctrl.read_text()

# Find the last } in the file — that's the closing brace of functions{}
last_brace = text.rfind('}')
if last_brace == -1:
    print(f"  ERROR: no closing }} found in {ctrl}")
    sys.exit(1)

new_text = text[:last_brace] + block + "\n" + text[last_brace:]
ctrl.write_text(new_text)
print(f"  Injected fieldAverage1 block into {ctrl}")
PY
    fi

    # --- 2. Verify endTime is 3000 ---
    end_time=$(grep 'endTime' "$CONTROL_DICT" | head -1 | awk '{print $2}' | tr -d ';')
    if [[ "$end_time" != "3000" ]]; then
        echo "  WARNING: endTime=$end_time (expected 3000) — please check controlDict"
    else
        echo "  endTime confirmed: 3000"
    fi

    # --- 3. Remove old solution time dirs so solver runs fresh from t=0 ---
    for tdir in 1200 3000; do
        tpath="$CASE/$tdir"
        if [[ -d "$tpath" ]]; then
            rm -rf "$tpath"
            echo "  Removed $tdir/ (stale solution)"
        fi
    done

    # --- 4. Clean any stale processor dirs ---
    n_proc=$(ls "$CASE" | grep -c 'processor' 2>/dev/null || true)
    if [[ "$n_proc" -gt 0 ]]; then
        rm -rf "$CASE"/processor*
        echo "  Removed $n_proc processor dirs"
    fi

    # --- 5. Verify postProcessing dir cleaned (fieldAverage writes here too) ---
    if [[ -d "$CASE/postProcessing/fieldAverage1" ]]; then
        rm -rf "$CASE/postProcessing/fieldAverage1"
        echo "  Cleaned stale postProcessing/fieldAverage1/"
    fi

    echo "  RUN_${run_id}: ready for SLURM submission"
done

echo ""
echo "Preparation complete. Submit with:"
echo "  sbatch --array=10,21,58 m3_fieldaverage_poc.slurm"
