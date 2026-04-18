#!/bin/bash
#SBATCH --job-name=OF_Resubmit
#SBATCH --partition=normal.4h
#SBATCH --account=es_fuge
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/elihuber/scratch_logs/OF_Resubmit_%j.out
#SBATCH --error=/cluster/scratch/elihuber/scratch_logs/OF_Resubmit_%j.err

set -euo pipefail

CASES_V2="/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/CASES_V2"
CODE_DIR="/cluster/home/elihuber/AISCO_code"

cd "$CODE_DIR"
source aisco_env/bin/activate

echo "=== Step 1: Fix 0/U syntax in all CASES_V2 (in-place) at $(date) ==="
# Replace: uniform(-V 0 0)  →  uniform (-V 0 0)
# Replace: dimensions[...]  →  dimensions      [...]
# Replace: - 1              →  -1  (in dimensions line only)
fixed=0
for U_FILE in "$CASES_V2"/RUN_*/0/U; do
    [[ -f "$U_FILE" ]] || continue
    # Only patch files that still have the broken syntax
    if grep -q 'uniform(-' "$U_FILE" || grep -qP 'dimensions\[' "$U_FILE"; then
        sed -i \
            -e 's/uniform(-\([0-9.]*\) /uniform (-\1 /g' \
            -e 's/dimensions\[/dimensions      [/g' \
            -e '/dimensions/s/- 1/-1/g' \
            "$U_FILE"
        ((fixed++)) || true
    fi
done
echo "Patched $fixed files."

echo "=== Step 2: Verify fix on sample cases ==="
grep "internalField" "$CASES_V2/RUN_001/0/U"
grep "internalField" "$CASES_V2/RUN_100/0/U"
grep "internalField" "$CASES_V2/RUN_241/0/U"

echo "=== Step 3: Submit mesh → RANS → extract → URANS pipeline at $(date) ==="
cd "$CODE_DIR/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/dataset/openfoam/slurm"
bash submit_all.sh 1-241 50

echo "=== Done — pipeline submitted at $(date) ==="
