#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=HashStorageSweep
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=%u@ethz.ch
## sbatch ~/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/hash_storage_sweep.sh
#
# Proof-of-correctness sweep for the Plan-031 unified hash-based storage.
# Runs 9 scenarios (6 latent, 3 solitary) and prints a structured report.
# Uses a temporary directory — no production storage is touched.
# Exit code 0 = all pass.
# ---------------------------------------------------------------------------
set -euo pipefail

umask 0002

export no_proxy=$(echo "${no_proxy:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "${NO_PROXY:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')

module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

USERNAME=$(whoami)

echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Working directory: $PWD"
echo "=========================================="

VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"
ts "Checkpoint: virtualenv activated"
"$PYTHON_BIN" --version

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:${PYTHONPATH:-}"
cd "/cluster/home/$USERNAME/AISCO_code"
ts "Checkpoint: changed directory to $(pwd)"

ts "Checkpoint: launching hash_storage_sweep.py"
"$PYTHON_BIN" -u \
    subprojects/wp4_aeroshell_optimizer/v2/scripts/hash_storage_sweep.py \
    2>&1
EXIT_CODE=$?

echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "Sweep exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
