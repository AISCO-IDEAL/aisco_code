#!/bin/bash
## ---------------------------------------------------------------------------
## AISCO-Beast sweep launcher (no SLURM allocation).
##
## Run directly on the local Beast machine or via the Beast task-spooler queue:
##
##   Direct:
##     bash ~/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/beast_sweeping.sh [slack-user]
##
##   Queued:
##     beast-submit <slack-user> bash ~/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/beast_sweeping.sh
##
##   With label:
##     beast-submit -L wp4_sweep <slack-user> bash ~/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/beast_sweeping.sh
##
## Notes:
##   - This is the Beast analogue of launch_sweep.sh.
##   - There is NO resource request here. GPU/CPU selection is whatever the local
##     Beast machine exposes to the process.
##   - Parallelism is controlled only by MAX_AGENTS and available GPUs.
## ---------------------------------------------------------------------------

set -euo pipefail

umask 0002

# Slack recipient: env var (set by beast-submit) > $1 > $USER
ETH_USER="${AISCO_NOTIFY_USERNAME:-${1:-${USER:-unknown}}}"
export AISCO_NOTIFY_USERNAME="$ETH_USER"
export AISCO_SUBMITTER="${AISCO_SUBMITTER:-$(id -un)}"

# ---------------------------------------------------------------------------
# Tee all output to a configurable log file
# ---------------------------------------------------------------------------
BEAST_LOG_DIR="${BEAST_LOG_DIR:-$HOME/beast_logs}"
mkdir -p "$BEAST_LOG_DIR"
JOB_TAG="beast_sweeping_$(date -u +%Y%m%dT%H%M%SZ)_$$"
BEAST_LOG_FILE="$BEAST_LOG_DIR/${JOB_TAG}.log"
ln -sfn "$BEAST_LOG_FILE" "$BEAST_LOG_DIR/latest.log"
exec > >(tee -a "$BEAST_LOG_FILE") 2>&1
echo "[beast_sweeping] full log: $BEAST_LOG_FILE"

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
    ts "Checkpoint: 'module' command not found; running local Beast mode"
fi

echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "tsp job ID: ${TS_SLOT:-${TS_SOCKET:+tsp}-N/A}"
echo "ETH user (Slack target): $ETH_USER"
echo "Running on node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Number of CPUs: ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Allocated memory per CPU: ${SLURM_MEM_PER_CPU:-N/A} MB"
echo "Working directory: $PWD"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Number of GPUs: $(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null || echo N/A)"
    echo "GPU type: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "Allocated memory per GPU: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
else
    echo "Number of GPUs: N/A"
    echo "GPU type: nvidia-smi not found"
    echo "Allocated memory per GPU: N/A"
fi
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
        ts "Hint: run ./setup_beast.sh or export AISCO_VENV_DIR"
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

# ---- Sweep configuration ---------------------------------------------------
SWEEP_ID="<EDIT_ME>"                      # bare id (abc123xy) or full path (entity/project/id) -- set after `wandb sweep ...`
SWEEP_ENTITY="aisco"                    # used when SWEEP_ID is bare
SWEEP_PROJECT="AISCOModel_Pretraining_Single"  # used when SWEEP_ID is bare; must match wandb sweep output
COUNT="10"                              # runs per agent
MAX_AGENTS="1"                          # for Beast RTX5090 this should usually stay 1
AGENT_STARTUP_STAGGER_SECONDS="10"
SWEEP_TIMEOUT_SECONDS="0"               # 0 = no timeout

# Never let shell-level WANDB_PROJECT hijack where the agent looks for the sweep.
unset WANDB_PROJECT

SWEEP_ID_CLEANED="$(echo "$SWEEP_ID" | xargs)"
if [[ -z "$SWEEP_ID_CLEANED" || "$SWEEP_ID_CLEANED" == "REPLACE_WITH_SWEEP_ID" ]]; then
    ts "ERROR: SWEEP_ID is empty / placeholder. Register the sweep first with wandb sweep ..."
    exit 1
fi
if [[ "$SWEEP_ID_CLEANED" == */*/* ]]; then
    SWEEP_TARGET="$SWEEP_ID_CLEANED"
else
    SWEEP_TARGET="${SWEEP_ENTITY}/${SWEEP_PROJECT}/${SWEEP_ID_CLEANED}"
fi
ts "Checkpoint: resolved sweep target = $SWEEP_TARGET"

PRECHECK_TIMEOUT_SECONDS="${PRECHECK_TIMEOUT_SECONDS:-60}"
ts "Checkpoint: starting sweep import preflight (${PRECHECK_TIMEOUT_SECONDS}s timeout)"
if timeout "${PRECHECK_TIMEOUT_SECONDS}s" "$PYTHON_BIN" -u - <<'PY'
print('preflight: interpreter ok', flush=True)
import torch
print(f'preflight: torch {torch.__version__}', flush=True)
import wandb
print(f'preflight: wandb {wandb.__version__}', flush=True)
import subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_single as _m
print('preflight: pretraining_single import ok', flush=True)
PY
then
    ts "Checkpoint: import preflight passed"
else
    rc=$?
    ts "ERROR: import preflight failed or timed out (rc=$rc)"
    exit "$rc"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    ts "ERROR: nvidia-smi not found — cannot determine GPU availability on Beast"
    exit 1
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NUM_GPUS" -gt "$MAX_AGENTS" ]; then
    NUM_GPUS=$MAX_AGENTS
fi
if [ "$NUM_GPUS" -le 0 ]; then
    ts "ERROR: no visible GPUs found"
    exit 1
fi
ts "Checkpoint: launching $NUM_GPUS wandb agent(s), count=$COUNT"

if [ "$SWEEP_TIMEOUT_SECONDS" -gt 0 ]; then
    ts "Checkpoint: applying hard timeout ${SWEEP_TIMEOUT_SECONDS}s to each agent process"
fi

for ((i=0; i<NUM_GPUS; i++)); do
    AGENT_LOG="${BEAST_LOG_DIR}/agent_gpu${i}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
    ts "GPU $i -> agent starting. Log: $AGENT_LOG"
    if [ "$SWEEP_TIMEOUT_SECONDS" -gt 0 ]; then
        CUDA_VISIBLE_DEVICES=$i timeout --signal=TERM --kill-after=30s "${SWEEP_TIMEOUT_SECONDS}s" \
            wandb agent --count "$COUNT" "$SWEEP_TARGET" > "$AGENT_LOG" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=$i wandb agent --count "$COUNT" "$SWEEP_TARGET" > "$AGENT_LOG" 2>&1 &
    fi
    sleep "$AGENT_STARTUP_STAGGER_SECONDS"
done
wait

echo "=========================================="
echo "Job ended on: $(date)"
echo "Total elapsed time: $SECONDS seconds"
echo "=========================================="
