#!/usr/bin/env bash
# One-button consolidation: copy QA-passing files from dirty/ → clean/ and strict/
# Run this after QA re-run is complete to update clean/ (PASS+WARN) and strict/ (PASS-only).
#
# Usage: bash consolidate_now.sh [--dry-run]

set -euo pipefail

source /cluster/home/elihuber/AISCO_code/aisco_env/bin/activate

python3 /cluster/home/elihuber/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/dataset/openfoam/consolidate_openfoam.py \
    --dirty-dir /cluster/work/fuge/AISCO/wp1_cfd_ai/openfoam_merged/dirty \
    --clean-dir  /cluster/work/fuge/AISCO/wp1_cfd_ai/openfoam_merged/clean \
    --strict-dir /cluster/work/fuge/AISCO/wp1_cfd_ai/openfoam_merged/strict \
    --verbose \
    "$@"
