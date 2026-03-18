#!/bin/bash

umask 0002
set -euo pipefail

# ---- Config, EDIT HERE ---------------------------------------------------------------
JOB_NAME="Worktree_Sweep" # Name for the SLURM job
TIME_LIMIT="04:00:00"  # Translates to --time below

BRANCH_PREFIX="temp-sweep-branch"
WORKTREE_SUFFIX="_worktree"
FREEZE_DIR_REL="subprojects/wp4_aeroshell_optimizer/v1/configs"
SWEEP_ID="aisco/Alex_Sweeps/3r4ab3is"   # <-- EDIT THIS
NUMBER_OF_GPUS=2  # Adjust based on your needs and cluster limits --> Translates to ntasks and gpus below
GPU_TYPE=""  # Set to "" for generic allocation, "rtx_4090" for specific gpu--> Translates to gpus below
RUNS_PER_GPU=2 # Adjust based on your needs and cluster limits --> Translates to COUNT below
CPUS_PER_TASK=8  # Translates to --cpus-per-task below
MEM_PER_CPU="5G"  # Translates to --mem-per-cpu below



# ---- Setup ---------------------------------------------------------------
USERNAME="$(whoami)"
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

TS=$(date +%Y%m%d_%H%M%S)
BRANCH_NAME="${BRANCH_PREFIX}-${TS}"
WORKTREE_DIR="$(dirname "$REPO_ROOT")/${BRANCH_NAME}${WORKTREE_SUFFIX}"

echo "=== Creating worktree ==="
git worktree add -b "$BRANCH_NAME" "$WORKTREE_DIR" HEAD

cd "$WORKTREE_DIR"
git commit --allow-empty -m "snapshot: $BRANCH_NAME"

# Freeze config
if [ ! -d "$WORKTREE_DIR/$FREEZE_DIR_REL" ]; then
    echo "[ERROR] Freeze directory not found: $WORKTREE_DIR/$FREEZE_DIR_REL" >&2
    exit 1
fi
chmod -R a-w "$WORKTREE_DIR/$FREEZE_DIR_REL" || true

echo "[INFO] Worktree ready at: $WORKTREE_DIR"

if [ -n "$GPU_TYPE" ]; then
    GPU_SBATCH="${GPU_TYPE}:${NUMBER_OF_GPUS}"
else
    GPU_SBATCH="${NUMBER_OF_GPUS}"
fi

# ---- Create modified launch script ---------------------------------------
LAUNCH_SCRIPT="$WORKTREE_DIR/launch_sweep_worktree.sh"

cat > "$LAUNCH_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --partition=gpu
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
echo "[INFO] Submitting job..."
SBATCH_OUTPUT=$(sbatch "$LAUNCH_SCRIPT")
echo "$SBATCH_OUTPUT"

JOB_ID=$(echo "$SBATCH_OUTPUT" | awk '{print $NF}')
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
