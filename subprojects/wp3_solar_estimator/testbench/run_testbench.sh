#!/bin/bash
#SBATCH --job-name=solar_testbench
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --output=/cluster/scratch/%u/logs/solar_testbench_%j.out
## run: sbatch ~/AISCO_code/subprojects/wp3_solar_estimator/testbench/run_testbench.sh

set -euo pipefail

ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# W&B proxy fix
export no_proxy=$(echo "${no_proxy:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "${NO_PROXY:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')

module purge
module load stack/2024-06 gcc/12.2.0 python/3.11.6 eth_proxy

USERNAME=$(whoami)
mkdir -p "/cluster/scratch/$USERNAME/logs"

echo "=========================================="
echo "Job started : $(date)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "CPUs        : $SLURM_CPUS_PER_TASK"
echo "=========================================="

VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"
ts "virtualenv activated"
"$PYTHON_BIN" --version

export PYTHONUNBUFFERED=TRUE
export WANDB_MODE=disabled
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:${PYTHONPATH:-}"

cd "/cluster/home/$USERNAME/AISCO_code"
ts "launching testbench_main.py"

EXIT_CODE=0
if ! "$PYTHON_BIN" -u subprojects/wp3_solar_estimator/testbench/testbench_main.py --all 2>&1; then
    EXIT_CODE=$?
    ts "ERROR: testbench_main.py exited with code $EXIT_CODE"
else
    ts "testbench_main.py finished successfully"
fi

echo "=========================================="
echo "Job ended: $(date)"
echo "Elapsed  : $SECONDS seconds"
echo "=========================================="
exit "$EXIT_CODE"
