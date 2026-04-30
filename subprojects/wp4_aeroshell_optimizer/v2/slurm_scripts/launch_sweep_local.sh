#!/bin/bash
# =============================================================================
# Local W&B sweep launcher — Prefocus Rollout (single-GPU local machine)
# -----------------------------------------------------------------------------
# How to use:
#   1) Create the sweep on W&B once:
#        wandb sweep subprojects/wp4_aeroshell_optimizer/v2/configs/sweeps/prefocus_rollout_sweep.yaml
#      Copy the printed sweep id (e.g. "aisco/prefocus-rollout-sweep/abc12345").
#   2) Edit SWEEP_ID below, then run this script:
#        bash subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/launch_sweep_local.sh
#
# Mirrors the local-mode behaviour of beast_testing.sh (PROJECT_ROOT
# autodetection, deterministic venv activation, full log tee). No SLURM,
# no module system, no multi-agent fan-out — one wandb agent on the local GPU.
#
# Configurable env vars:
#   AISCO_PROJECT_ROOT     override project root detection
#   AISCO_VENV_DIR         override venv (default: $PROJECT_ROOT//aisco_env)
#   BEAST_LOG_DIR          log directory (default: $HOME/beast_logs)
#   CUDA_VISIBLE_DEVICES   GPU selection (default: 0)
#   SWEEP_ID               override the SWEEP_ID set inside this file
#   COUNT                  override the per-agent run count
# =============================================================================

set -euo pipefail
umask 0002

# ---- Configurable variables ------------------------------------------------
SWEEP_ID="${SWEEP_ID:-aisco/prefocus-rollout-sweep/vbyrbheh}"   # <-- EDIT THIS after `wandb sweep ...`
COUNT="${COUNT:-56}"                                              # 7 scenarios x 4 epochs x 2 schedulers
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Allow W&B to route normally (drop api.wandb.ai from no_proxy if present)
export no_proxy=$(echo "${no_proxy:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "${NO_PROXY:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')

ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---- Locate project root ---------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

_find_project_root() {
    local dir="$1"
    while [ -n "$dir" ] && [ "$dir" != "/" ]; do
        if [ -f "$dir/pyproject.toml" ] || [ -d "$dir//aisco_env" ] || [ "$(basename "$dir")" = "AISCO_code" ]; then
            printf '%s\n' "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

if [ -n "${AISCO_PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="$AISCO_PROJECT_ROOT"
elif PROJECT_ROOT="$(_find_project_root "$SCRIPT_DIR")"; then
    :
elif PROJECT_ROOT="$(_find_project_root "$PWD")"; then
    :
else
    ts "ERROR: could not locate AISCO_code project root (SCRIPT_DIR=$SCRIPT_DIR, PWD=$PWD)"
    ts "Hint: export AISCO_PROJECT_ROOT=/home/aisco/Documents/AISCO_code"
    exit 1
fi

# ---- Tee output to a persistent log ---------------------------------------
BEAST_LOG_DIR="${BEAST_LOG_DIR:-$HOME/beast_logs}"
mkdir -p "$BEAST_LOG_DIR"
JOB_TAG="prefocus_sweep_local_$(date -u +%Y%m%dT%H%M%SZ)_$$"
BEAST_LOG_FILE="$BEAST_LOG_DIR/${JOB_TAG}.log"
ln -sfn "$BEAST_LOG_FILE" "$BEAST_LOG_DIR/latest.log"
exec > >(tee -a "$BEAST_LOG_FILE") 2>&1
ts "Full log: $BEAST_LOG_FILE"

# ---- Banner ---------------------------------------------------------------
echo "=========================================="
echo "Prefocus Rollout sweep — local agent"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "User: ${USER:-$(id -un)}"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "SWEEP_ID: $SWEEP_ID"
echo "COUNT (max runs this agent will pull): $COUNT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

if [[ "$SWEEP_ID" == *"REPLACE_ME"* ]]; then
    ts "ERROR: SWEEP_ID is not set. Run 'wandb sweep <yaml>' first, then edit SWEEP_ID."
    exit 1
fi

# ---- Activate venv --------------------------------------------------------
if [ -n "${AISCO_VENV_DIR:-}" ]; then
    VENV_DIR="$AISCO_VENV_DIR"
else
    VENV_DIR="$PROJECT_ROOT//aisco_env"
fi
if [ ! -x "$VENV_DIR/bin/python" ]; then
    ts "ERROR: required venv not found at $VENV_DIR"
    ts "Hint: run ./setup_beast.sh or set AISCO_VENV_DIR"
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ts "venv activated: $VENV_DIR"
"$VENV_DIR/bin/python" --version

# ---- GPU sanity ----------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    ts "GPU info:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
    ts "WARNING: nvidia-smi not found; sweep will run on whatever device main_v2 selects"
fi

# ---- Python / wandb env ---------------------------------------------------
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="$PROJECT_ROOT/subprojects:${PYTHONPATH:-}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH}"
cd "$PROJECT_ROOT"
ts "cwd: $(pwd)"

if ! command -v wandb >/dev/null 2>&1; then
    ts "ERROR: 'wandb' CLI not on PATH after activating venv"
    exit 1
fi

# ---- Launch single agent --------------------------------------------------
ts "Launching wandb agent (count=$COUNT) for sweep $SWEEP_ID ..."
wandb agent --count "$COUNT" "$SWEEP_ID"

echo "=========================================="
ts "Agent finished. Total elapsed: ${SECONDS}s"
echo "=========================================="
