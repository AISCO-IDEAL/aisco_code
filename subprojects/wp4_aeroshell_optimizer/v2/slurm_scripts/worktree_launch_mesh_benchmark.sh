#!/bin/bash

set -uo pipefail
trap 'echo "[WARN] Script error at line $LINENO (exit $?)" >&2' ERR

# ---- Config, EDIT HERE ---------------------------------------------------------------
JOB_NAME="${JOB_NAME:-Mesh_Benchmark}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
NUMBER_OF_GPUS="${NUMBER_OF_GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-rtx_4090}"  # "" for generic, or e.g. rtx_2080 / rtx_4090
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_PER_CPU="${MEM_PER_CPU:-5G}"

# Benchmark selections
ONLY_RUNS="${ONLY_RUNS:-v1_main_v1_1_one_model v2_main_v1_2_GINN_concat}"
ONLY_CASES="${ONLY_CASES:-mesh_sdf diso_mc diso_dmc flexicubes}"
ONLY_GPUS="${ONLY_GPUS:-rtx4090}"  # choose profile(s) from tests/wp5/mesh_benchmark_scenarios.yaml

# SLURM compatibility overrides (recommended for single-GPU allocations)
FORCE_CUDA_VISIBLE_DEVICES="${FORCE_CUDA_VISIBLE_DEVICES:-0}"
FORCE_GPU_QUERY_INDEX="${FORCE_GPU_QUERY_INDEX:-0}"
SAMPLE_INTERVAL_SEC="${SAMPLE_INTERVAL_SEC:-1.0}"

# Freeze mode:
#   auto (default): prompt if TTY exists, otherwise skip
#   yes: force freeze
#   no: skip freeze without prompting
FREEZE_MODE="${FREEZE_MODE:-auto}"

BENCH_SCENARIO_REL="${BENCH_SCENARIO_REL:-tests/wp5/mesh_benchmark_scenarios.yaml}"
BENCH_RUNNER_REL="${BENCH_RUNNER_REL:-tests/wp5/mesh_benchmark_runner.py}"

FREEZE_DIR_REL="subprojects/wp4_aeroshell_optimizer/v2/configs"
BRANCH_PREFIX="temp-benchmark-branch"
WORKTREE_SUFFIX="_worktree"


# ---- Setup ---------------------------------------------------------------
export no_proxy=$(echo "${no_proxy:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "${NO_PROXY:-}" | sed 's/api\.wandb\.ai,\?//g; s/,$//')

USERNAME="$(whoami)"
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Move unstaged changes to new worktree
PATCH_FILE=""
UNTRACKED_LIST=""
UNTRACKED_TAR=""
if [ -n "$(git status --porcelain)" ]; then
    echo "[INFO] Working tree dirty — capturing changes for worktree copy"
    PATCH_FILE="/tmp/worktree-benchmark-$(date +%Y%m%d_%H%M%S).patch"
    git diff > "$PATCH_FILE"

    UNTRACKED_LIST="/tmp/worktree-benchmark-untracked-$(date +%Y%m%d_%H%M%S).txt"
    git ls-files --others --exclude-standard > "$UNTRACKED_LIST"

    if [ -s "$UNTRACKED_LIST" ]; then
        UNTRACKED_TAR="/tmp/worktree-benchmark-untracked-$(date +%Y%m%d_%H%M%S).tar"
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
freeze_reply="n"
case "$FREEZE_MODE" in
    yes|YES|y|Y)
        freeze_reply="y"
        ;;
    no|NO|n|N)
        freeze_reply="n"
        ;;
    auto|AUTO|"")
        if [ -t 0 ] && [ -r /dev/tty ]; then
            read -r -p "Freeze config directory? [y/N] " freeze_reply </dev/tty
        else
            freeze_reply="n"
            echo "[INFO] No TTY detected, skipping config freeze."
        fi
        ;;
    *)
        echo "[ERROR] Invalid FREEZE_MODE='$FREEZE_MODE'. Use: auto|yes|no." >&2
        exit 1
        ;;
esac
case "$freeze_reply" in
    y|Y)
        if [ ! -d "$WORKTREE_DIR/$FREEZE_DIR_REL" ]; then
            echo "[ERROR] Freeze directory not found: $WORKTREE_DIR/$FREEZE_DIR_REL" >&2
            exit 1
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

# ---- Create modified launch script ---------------------------------------
LAUNCH_SCRIPT="$WORKTREE_DIR/launch_mesh_benchmark_worktree.sh"

cat > "$LAUNCH_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus=${GPU_SBATCH}
#SBATCH --mem-per-cpu=${MEM_PER_CPU}
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out
#SBATCH --mail-type=START,END,FAIL

# ---- Config --------------------------------------------------------------
WORKTREE_DIR="$WORKTREE_DIR"
BENCH_SCENARIO_REL="${BENCH_SCENARIO_REL}"
BENCH_RUNNER_REL="${BENCH_RUNNER_REL}"
ONLY_RUNS="${ONLY_RUNS}"
ONLY_CASES="${ONLY_CASES}"
ONLY_GPUS="${ONLY_GPUS}"
FORCE_CUDA_VISIBLE_DEVICES="${FORCE_CUDA_VISIBLE_DEVICES}"
FORCE_GPU_QUERY_INDEX=${FORCE_GPU_QUERY_INDEX}
SAMPLE_INTERVAL_SEC=${SAMPLE_INTERVAL_SEC}

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
echo "CUDA_VISIBLE_DEVICES (pre-run): \$CUDA_VISIBLE_DEVICES"

# ---- Environment --------------------------------------------------------
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy
umask 0002
USERNAME="\$(whoami)"
VENV_DIR="/cluster/home/\${USERNAME}/AISCO_code/aisco_env"
source "\$VENV_DIR/bin/activate"

export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="\$WORKTREE_DIR/subprojects:\$PYTHONPATH"
export PYTHONPATH="\$WORKTREE_DIR:\$PYTHONPATH"

cd "\$WORKTREE_DIR"

echo "Git commit used:"
git rev-parse HEAD

# ---- Launch benchmark ----------------------------------------------------
CMD=(python "\$BENCH_RUNNER_REL" --scenario "\$BENCH_SCENARIO_REL" --sample-interval "\$SAMPLE_INTERVAL_SEC" --repo-root-override "\$WORKTREE_DIR")

if [ -n "\$ONLY_RUNS" ]; then
  CMD+=(--only-runs)
    for token in \$ONLY_RUNS; do CMD+=("\$token"); done
fi

if [ -n "\$ONLY_CASES" ]; then
  CMD+=(--only-cases)
    for token in \$ONLY_CASES; do CMD+=("\$token"); done
fi

if [ -n "\$ONLY_GPUS" ]; then
  CMD+=(--only-gpus)
    for token in \$ONLY_GPUS; do CMD+=("\$token"); done
fi

if [ -n "\$FORCE_CUDA_VISIBLE_DEVICES" ]; then
    CMD+=(--force-cuda-visible-devices "\$FORCE_CUDA_VISIBLE_DEVICES")
fi

CMD+=(--force-gpu-query-index "\$FORCE_GPU_QUERY_INDEX")

echo "Running benchmark command: \${CMD[*]}"
"\${CMD[@]}"

echo "=========================================="
echo "[\$(date)] Benchmark run finished."
echo "=========================================="
EOF

chmod +x "$LAUNCH_SCRIPT"

# ---- Submit job ----------------------------------------------------------
echo "[INFO] Submitting benchmark job..."
SBATCH_OUTPUT=$(sbatch "$LAUNCH_SCRIPT")
echo "$SBATCH_OUTPUT"

JOB_ID=$(echo "$SBATCH_OUTPUT" | awk '{print $NF}')
if [ -z "$JOB_ID" ]; then
    echo "[ERROR] Failed to parse SLURM job id from sbatch output." >&2
    exit 1
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
