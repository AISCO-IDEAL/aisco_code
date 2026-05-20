#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=Main_v2_local
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out
## ---------------------------------------------------------------------------
## How to run on AISCO-Beast (shared Task Spooler GPU queue)
##
##   Submit (queued, one heavy job at a time, Slack DM on finish):
##     beast-submit <slack-user> bash ~/path/to/beast_testing.sh
##
##   With a label:
##     beast-submit -L wp4_test eschaerer bash ~/path/to/beast_testing.sh
##
##   Inspect (works while a job is running, from any ssh session):
##     tsp                       # list queued / running / finished
##     tsp -i <ID>               # info on one job
##     tail -f $(tsp -o <ID>)    # follow stdout file of a job
##     tsp -t <ID>               # tail finished job's output
##     tsp -r <ID>               # remove / kill a job
##
##   Run directly (bypasses queue, blocks your shell):
##     bash ~/path/to/beast_testing.sh [slack-user]
##
## Configurable persistent log directory:
##   export BEAST_LOG_DIR=/path/to/dir   # default: $HOME/beast_logs
##   Full stdout/stderr is teed to:
##       $BEAST_LOG_DIR/<jobname>_<UTC-timestamp>_<pid>.log
##   plus $BEAST_LOG_DIR/latest.log -> newest run.
##
## Slack recipient resolution order inside the script:
##   1. AISCO_NOTIFY_USERNAME env var (set automatically by beast-submit)
##   2. $1 positional arg
##   3. $USER
## ---------------------------------------------------------------------------

set -uo pipefail

umask 0002

# Slack recipient: env var (set by beast-submit) > $1 > $USER
ETH_USER="${AISCO_NOTIFY_USERNAME:-${1:-${USER:-unknown}}}"
export AISCO_NOTIFY_USERNAME="$ETH_USER"
export AISCO_SUBMITTER="${AISCO_SUBMITTER:-$(id -un)}"

# ---------------------------------------------------------------------------
# Tee all output to a configurable log file (in addition to tsp's /tmp/ts-out.*)
# ---------------------------------------------------------------------------
BEAST_LOG_DIR="${BEAST_LOG_DIR:-$HOME/beast_logs}"
mkdir -p "$BEAST_LOG_DIR"
JOB_TAG="${SLURM_JOB_NAME:-beast_testing}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
BEAST_LOG_FILE="$BEAST_LOG_DIR/${JOB_TAG}.log"
# Symlink "latest" for convenience: tail -f $BEAST_LOG_DIR/latest.log
ln -sfn "$BEAST_LOG_FILE" "$BEAST_LOG_DIR/latest.log"
exec > >(tee -a "$BEAST_LOG_FILE") 2>&1
echo "[beast_testing] full log: $BEAST_LOG_FILE"

ts() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

export no_proxy=$(echo "${no_proxy:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "${NO_PROXY:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../../../" && pwd)"

if command -v module >/dev/null 2>&1; then
    module purge
    module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy
    ts "Checkpoint: environment modules loaded"
else
    ts "Checkpoint: 'module' command not found; running local mode"
fi

echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "tsp job ID: ${TS_SLOT:-${TS_SOCKET:+tsp}-N/A}"
echo "ETH user (Slack target): $ETH_USER"
echo "Running on node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Number of CPUs: ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Allocated memory per CPU: ${SLURM_MEM_PER_CPU:-N/A} MB"
echo "Number of GPUs: ${SLURM_GPUS:-$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null || echo N/A)}"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU type: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "Allocated memory per GPU: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
else
    echo "GPU type: nvidia-smi not found"
    echo "Allocated memory per GPU: N/A"
fi
echo "Working directory: $PWD"
echo "=========================================="

# Activate python environment (deterministic selection)
if [ -n "${AISCO_VENV_DIR:-}" ]; then
    VENV_DIR="$AISCO_VENV_DIR"
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        ts "ERROR: AISCO_VENV_DIR is set but invalid: $VENV_DIR"
        exit 1
    fi
    ts "Checkpoint: using explicitly requested venv from AISCO_VENV_DIR"
else
    VENV_DIR="$PROJECT_ROOT/aisco_env"
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        ts "ERROR: required local env not found: $VENV_DIR"
        ts "Hint: run ./setup_beast.sh or export AISCO_VENV_DIR, for example: /home/aisco/Documents/AISCO_code/aisco_env"
        exit 1
    fi
    ts "Checkpoint: using default local env: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"
ts "Checkpoint: virtualenv activated ($VENV_DIR)"
ts "Checkpoint: python executable = $PYTHON_BIN"
"$PYTHON_BIN" --version

echo "Verifying GPU access..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not-set>}"
ts "Checkpoint: GPU visibility check completed"

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="$PROJECT_ROOT/subprojects:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"
ts "Checkpoint: changed directory to $(pwd)"

ts "Checkpoint: starting import preflight"
if "$PYTHON_BIN" -u - <<'PY'
print('preflight: interpreter ok', flush=True)
import torch
print(f'preflight: torch {torch.__version__}', flush=True)
import subprojects.wp4_aeroshell_optimizer.v2.src.main_files.main_v2 as _m
print('preflight: main_v2 import ok', flush=True)
PY
then
    ts "Checkpoint: import preflight passed"
else
    rc=$?
    ts "ERROR: import preflight failed (rc=$rc)"
    exit "$rc"
fi

ts "Checkpoint: launching main_v2.py"
if "$PYTHON_BIN" -u subprojects/wp4_aeroshell_optimizer/v2/src/main_files/main_v2.py \
    --save_model \
    --problem_id="AISCO_WP4_Test1" 2>&1 #aalthaus, very unhappy with the arguments, don't do anything
then
    ts "Checkpoint: main_v2.py finished successfully"
else
    rc=$?
    ts "ERROR: main_v2.py exited non-zero (rc=$rc)"
    exit "$rc"
fi

echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="
