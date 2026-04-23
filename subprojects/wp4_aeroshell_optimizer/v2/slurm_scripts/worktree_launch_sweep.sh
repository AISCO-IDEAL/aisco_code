#!/bin/bash


set -uo pipefail
trap 'echo "[WARN] Script error at line $LINENO (exit $?)" >&2' ERR

# ---- Config, EDIT HERE ---------------------------------------------------------------
JOB_NAME="Fitting_Sweep" # Name for the SLURM job
TIME_LIMIT="03:00:00"  # Translates to --time below
SWEEP_ID="aisco/WIRE-Fitting/h6bw3iwq"   # <-- EDIT THIS
NUMBER_OF_GPUS=1  # Keep low to avoid QOSGrpGRES contention; increase when quota allows
GPU_TYPE=""  # Matches the known-working benchmark profile
RUNS_PER_GPU=160 # Adjust based on your needs and cluster limits --> Translates to COUNT below
CPUS_PER_TASK=2  # Translates to --cpus-per-task below
MEM_PER_CPU="3G"  # Translates to --mem-per-cpu below
SLURM_PARTITION="gpu"  # Translates to --partition below
SLURM_ACCOUNT="es_fuge"  # Translates to --account below
SLURM_QOS=""  # Optional, translates to --qos below when non-empty
FREEZE_DIR_REL=""
BRANCH_PREFIX="fit-sweep-branch"
WORKTREE_SUFFIX="_worktree"


# ---- Setup ---------------------------------------------------------------
# Allow W&B to route through the cluster proxy (eth_proxy sets no_proxy which blocks api.wandb.ai)
export no_proxy=$(echo "${no_proxy:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "${NO_PROXY:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')

fail_script() {
    trap - ERR
    return 1 2>/dev/null || exit 1
}

print_sbatch_hint() {
    local scheduler_output="$1"

    if echo "$scheduler_output" | grep -q "QOSGrpMemLimit"; then
        echo "[HINT] SLURM rejected the job due to group memory policy (QOSGrpMemLimit)." >&2
        echo "[HINT] Try reducing NUMBER_OF_GPUS, CPUS_PER_TASK, or MEM_PER_CPU, or wait for other jobs in the same account/QOS to finish." >&2
    fi

    if echo "$scheduler_output" | grep -q "QOSGrpGRES"; then
        echo "[HINT] SLURM rejected the job due to group GPU/GRES quota (QOSGrpGRES)." >&2
        echo "[HINT] Try NUMBER_OF_GPUS=1 and a specific GPU type (for example rtx_4090), or wait for group GPU usage to drop." >&2
    fi

    if echo "$scheduler_output" | grep -qi "invalid account\|invalid account or account\/partition combination"; then
        echo "[HINT] Check SLURM_ACCOUNT and partition permissions for this cluster." >&2
        if [ "$SLURM_ACCOUNT" = "public" ] && echo "$SLURM_PARTITION" | grep -q '^gpu'; then
            echo "[HINT] public is typically not enabled on GPU partitions for this cluster." >&2
            echo "[HINT] Use a GPU account (for example es_fuge) or switch to a CPU partition when using public." >&2
        fi
    fi

    if echo "$scheduler_output" | grep -qi "invalid qos\|invalid qos specification"; then
        echo "[HINT] Check SLURM_QOS or leave it empty to use the default qos for the account." >&2
    fi
}

USERNAME="$(whoami)"
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

### Move unstaged changes to new worktree
PATCH_FILE=""
UNTRACKED_LIST=""
UNTRACKED_TAR=""
if [ -n "$(git status --porcelain)" ]; then
    echo "[INFO] Working tree dirty — capturing changes for worktree copy"
    PATCH_FILE="/tmp/worktree-sweep-$(date +%Y%m%d_%H%M%S).patch"
    git diff > "$PATCH_FILE"

    UNTRACKED_LIST="/tmp/worktree-sweep-untracked-$(date +%Y%m%d_%H%M%S).txt"
    git ls-files --others --exclude-standard > "$UNTRACKED_LIST"

    if [ -s "$UNTRACKED_LIST" ]; then
        UNTRACKED_TAR="/tmp/worktree-sweep-untracked-$(date +%Y%m%d_%H%M%S).tar"
        tar -cf "$UNTRACKED_TAR" -T "$UNTRACKED_LIST"
    fi
fi

TS=$(date +%Y%m%d_%H%M%S)
BRANCH_NAME="${BRANCH_PREFIX}-${TS}"
WORKTREE_DIR="$(dirname "$REPO_ROOT")/${BRANCH_NAME}${WORKTREE_SUFFIX}"

echo "=== Creating worktree ==="
git worktree add -b "$BRANCH_NAME" "$WORKTREE_DIR" HEAD

cd "$WORKTREE_DIR"
if [ -n "$PATCH_FILE" ]; then
    git apply "$PATCH_FILE"
    if [ -n "$UNTRACKED_TAR" ]; then
        tar -xf "$UNTRACKED_TAR"
    fi
    git add -A
    git commit --no-verify -m "snapshot: $BRANCH_NAME (worktree)"
    rm -f "$PATCH_FILE" "$UNTRACKED_LIST" "$UNTRACKED_TAR"
else
    git commit --allow-empty --no-verify -m "snapshot: $BRANCH_NAME"
fi

# Freeze config (optional)
freeze_reply="${FREEZE_CONFIG:-}"
if [ -z "$freeze_reply" ]; then
    if [ -t 0 ] && [ -r /dev/tty ]; then
        read -r -p "Freeze config directory? [y/N] " freeze_reply </dev/tty || freeze_reply="N"
    else
        freeze_reply="N"
        echo "[INFO] Non-interactive session detected; skipping config freeze prompt (set FREEZE_CONFIG=y to force)."
    fi
fi

case "$freeze_reply" in
    y|Y)
        if [ ! -d "$WORKTREE_DIR/$FREEZE_DIR_REL" ]; then
            echo "[ERROR] Freeze directory not found: $WORKTREE_DIR/$FREEZE_DIR_REL" >&2
            fail_script
        fi
        chmod -R a-w "$WORKTREE_DIR/$FREEZE_DIR_REL" || true
        ;;
    *)
        echo "[INFO] Skipping config freeze."
        ;;
esac

echo "[INFO] Worktree ready at: $WORKTREE_DIR"

if [ -n "$GPU_TYPE" ]; then
    GPU_SBATCH="${GPU_TYPE}:${NUMBER_OF_GPUS}"
else
    GPU_SBATCH="${NUMBER_OF_GPUS}"
fi

SBATCH_QOS_LINE=""
if [ -n "$SLURM_QOS" ]; then
    SBATCH_QOS_LINE="#SBATCH --qos=${SLURM_QOS}"
fi

# ---- Create modified launch script ---------------------------------------
LAUNCH_SCRIPT="$WORKTREE_DIR/launch_sweep_worktree.sh"

cat > "$LAUNCH_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
${SBATCH_QOS_LINE}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --ntasks=${NUMBER_OF_GPUS}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus=${GPU_SBATCH}
#SBATCH --mem-per-cpu=${MEM_PER_CPU}
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out
#SBATCH --mail-type=START,END,FAIL

# ---- Config --------------------------------------------------------------
SWEEP_ID="${SWEEP_ID}"
COUNT=${RUNS_PER_GPU}
WORKTREE_DIR="$WORKTREE_DIR"

# ---- Job info ------------------------------------------------------------
echo "=========================================="
echo "Job started on: \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Running on node: \$SLURMD_NODENAME"
echo "Number of CPUs: \$SLURM_CPUS_PER_TASK"
echo "Allocated memory per CPU: \$SLURM_MEM_PER_CPU MB"
echo "Number of GPUs: \$SLURM_GPUS"
echo "GPU type: \$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Allocated memory per GPU: \$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
echo "Working directory: \$PWD"
echo "=========================================="

# ---- Verify GPU access ---------------------------------------------------
echo "Verifying GPU access..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES"

# ---- Environment --------------------------------------------------------
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy
umask 0002
USERNAME="\$(whoami)"
LOG_DIR="/cluster/scratch/\${USERNAME}/logs/\${SLURM_JOB_NAME}_\${SLURM_JOB_ID}"
mkdir -p "\$LOG_DIR"

VENV_DIR="/cluster/home/\${USERNAME}/AISCO_code/aisco_env"
source "\$VENV_DIR/bin/activate"

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="\$WORKTREE_DIR/subprojects:\$PYTHONPATH"
export PYTHONPATH="\$WORKTREE_DIR:\$PYTHONPATH"

cd "\$WORKTREE_DIR"

echo "Git commit used:"
git rev-parse HEAD

# ---- Launch agents ------------------------------------------------------
NUM_GPUS=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "[\$(date)] Detected \$NUM_GPUS GPUs — launching 1 agent each, \$COUNT runs per agent..."

for ((i=0; i<NUM_GPUS; i++)); do
    AGENT_LOG="\${LOG_DIR}/agent_gpu\${i}_\${SLURM_JOB_ID}.log"
    echo "[\$(date)]   GPU \$i → agent starting. Log: \$AGENT_LOG"
    CUDA_VISIBLE_DEVICES=\$i wandb agent --count "\$COUNT" "\$SWEEP_ID" > "\$AGENT_LOG" 2>&1 &
    sleep 10
done

wait

echo "=========================================="
echo "[\$(date)] All agents finished."
echo "Total elapsed: \$SECONDS seconds"
echo "=========================================="
EOF

chmod +x "$LAUNCH_SCRIPT"

# ---- Submit job ----------------------------------------------------------
if ! command -v sbatch >/dev/null 2>&1; then
    echo "[ERROR] sbatch command not found. Please load the SLURM environment before running this script." >&2
    fail_script
fi

if [ -z "$SLURM_ACCOUNT" ]; then
    echo "[ERROR] SLURM_ACCOUNT is empty. Set it to a valid account (for example: public)." >&2
    fail_script
fi

if [ -z "$SLURM_PARTITION" ]; then
    echo "[ERROR] SLURM_PARTITION is empty. Set it to a valid partition (for example: gpu.4h)." >&2
    fail_script
fi

BASE_PARTITION="${SLURM_PARTITION%%.*}"
if [ "$SLURM_ACCOUNT" = "public" ] && [ "$BASE_PARTITION" = "gpu" ]; then
    echo "[ERROR] SLURM_ACCOUNT=public is not permitted with GPU partitions on this cluster." >&2
    echo "[HINT] For GPU runs, use a GPU-enabled account (for example: SLURM_ACCOUNT=es_fuge)." >&2
    echo "[HINT] For public runs, switch to a CPU partition (for example: SLURM_PARTITION=normal.4h)." >&2
    fail_script
fi

if [ ! -r "$LAUNCH_SCRIPT" ]; then
    echo "[ERROR] Launch script is missing or not readable: $LAUNCH_SCRIPT" >&2
    fail_script
fi

if sbatch --help 2>&1 | grep -q -- "--test-only"; then
    trap - ERR
    TEST_OUTPUT=$(sbatch --test-only "$LAUNCH_SCRIPT" 2>&1)
    TEST_STATUS=$?
    trap 'echo "[WARN] Script error at line $LINENO (exit $?)" >&2' ERR
    if [ "$TEST_STATUS" -ne 0 ]; then
        echo "[ERROR] SLURM preflight failed. Job was not submitted." >&2
        echo "$TEST_OUTPUT" >&2
        print_sbatch_hint "$TEST_OUTPUT"
        fail_script
    fi
fi

echo "[INFO] Submitting job..."
trap - ERR
SBATCH_OUTPUT=$(sbatch "$LAUNCH_SCRIPT" 2>&1)
SBATCH_STATUS=$?
trap 'echo "[WARN] Script error at line $LINENO (exit $?)" >&2' ERR
echo "$SBATCH_OUTPUT"

if [ "$SBATCH_STATUS" -ne 0 ]; then
    echo "[ERROR] sbatch submission failed. Job was not submitted." >&2
    print_sbatch_hint "$SBATCH_OUTPUT"
    fail_script
fi

JOB_ID=$(echo "$SBATCH_OUTPUT" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)
if [ -z "$JOB_ID" ]; then
    echo "[ERROR] Could not parse SLURM job ID from sbatch output." >&2
    echo "[ERROR] Raw sbatch output: $SBATCH_OUTPUT" >&2
    fail_script
fi

echo "[INFO] Job ID: $JOB_ID"

# ---- Wait for completion -------------------------------------------------
echo "[INFO] Waiting for job to finish..."

while squeue -j "$JOB_ID" 2>/dev/null | grep -q "$JOB_ID"; do
    sleep 30
done

echo "[INFO] Job finished"

# ---- Cleanup -------------------------------------------------------------
cd "$WORKTREE_DIR"
chmod -R u+w "$WORKTREE_DIR/$FREEZE_DIR_REL" || true
rm -f "$LAUNCH_SCRIPT" || true

if [ -z "$(git status --porcelain)" ]; then
    echo "[INFO] No changes detected — cleaning up"
    cd "$REPO_ROOT"
    git worktree remove "$WORKTREE_DIR" --force
    git branch -D "$BRANCH_NAME"
else
    echo "[WARN] Changes detected — keeping worktree"
    git status
fi

echo "=== Done ==="
