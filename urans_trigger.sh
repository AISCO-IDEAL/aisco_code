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
