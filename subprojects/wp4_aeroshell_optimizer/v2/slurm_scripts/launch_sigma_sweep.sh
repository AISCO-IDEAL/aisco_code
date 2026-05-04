#!/bin/bash
# Launch a sweep over surface noise sigma configurations for latent pretraining.
#
# Each sbatch job uses pretrain_latent_sigma_sweep.slurm with different sigma values.
# All jobs share the same config (pretraining_latent_sweep_v1_config.yaml) and shapes.
#
# Sigma combos:
#   1. tight=0.001              — single-band, very tight (baseline)
#   2. tight=0.001, wide=0.01   — dual-band: tight + moderate  (DeepSDF-style)
#   3. tight=0.005, wide=0.05   — dual-band: wider both bands
#
# Usage:
#   cd ~/AISCO_code
#   bash subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/launch_sigma_sweep.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/pretrain_latent_sigma_sweep.slurm"

echo "Submitting sigma sweep jobs …"

# --- Combo 1: single tight band (baseline) ---
JID1=$(sbatch --export=ALL,TIGHT_SIGMA=0.001 \
    --job-name="sigma_0.001" \
    "$SLURM_SCRIPT" | awk '{print $NF}')
echo "  [1] tight=0.001              wide=none     → job $JID1"

# --- Combo 2: dual-band tight + moderate ---
JID2=$(sbatch --export=ALL,TIGHT_SIGMA=0.001,WIDE_SIGMA=0.01 \
    --job-name="sigma_0.001_0.01" \
    "$SLURM_SCRIPT" | awk '{print $NF}')
echo "  [2] tight=0.001  wide=0.01               → job $JID2"

# --- Combo 3: dual-band both wider ---
JID3=$(sbatch --export=ALL,TIGHT_SIGMA=0.005,WIDE_SIGMA=0.05 \
    --job-name="sigma_0.005_0.05" \
    "$SLURM_SCRIPT" | awk '{print $NF}')
echo "  [3] tight=0.005  wide=0.05               → job $JID3"

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f /cluster/scratch/\$USER/logs/sigma_0.001_<JOBID>.out"
