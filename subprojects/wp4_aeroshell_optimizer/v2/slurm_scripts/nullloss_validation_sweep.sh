#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=NullLoss_Validation
#SBATCH --time=02:00:00                   # 8 scenarios * ~10-15min each
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G                  # Total 256G memory
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/%u/logs/%x_%j.out
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=%u@ethz.ch
## run this command: sbatch ~/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/slurm_scripts/nullloss_validation_sweep.sh
#------------------------------------------------------------------------------
# Automated Nullloss Validation Runner (SLURM)
#------------------------------------------------------------------------------
# This script tests all nullloss configurations with 5-epoch runs on Euler cluster.
# It creates temporary configs for each scenario and runs them sequentially.
#
# Test scenarios (8 total):
#   1. Baseline (no nulllosses)
#   2-7. Individual nulllosses (aero, solar, bb, eikonal, smoothness, developability)
#   8. Two nulllosses simultaneously (bb + developability)
#
# Output:
#   - Logs: /cluster/scratch/$USER/logs/
#   - Configs: /cluster/scratch/$USER/nullloss_sweep_TIMESTAMP/
#   - wandb: project "nullloss_validation"
#------------------------------------------------------------------------------

set -e  # Exit on error
umask 0002  # Ensure group write permissions for logs and models

#------------------------------------------------------------------------------
# Allow W&B to route through cluster proxy
#------------------------------------------------------------------------------
export no_proxy=$(echo "$no_proxy" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "$NO_PROXY" | sed 's/api\.wandb\.ai,\?//g; s/,$//')

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 python_cuda/3.11.6 eth_proxy

#------------------------------------------------------------------------------
# Setup environment variables
#------------------------------------------------------------------------------
USERNAME=$(whoami)
PROJECT_ROOT="/cluster/home/$USERNAME/AISCO_code"
CONFIG_DIR="$PROJECT_ROOT/subprojects/wp4_aeroshell_optimizer/v2/configs"
MAIN_SCRIPT="$PROJECT_ROOT/subprojects/wp4_aeroshell_optimizer/v2/src/main_files/main_v2.py"
LOSSES_CONFIG="$CONFIG_DIR/losses_v0_config.yaml"
TRAINING_CONFIG="$CONFIG_DIR/training_v0_config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRATCH_DIR="/cluster/scratch/$USERNAME"
LOG_DIR="$SCRATCH_DIR/logs"
SWEEP_DIR="$SCRATCH_DIR/nullloss_sweep_$TIMESTAMP"

# Ensure log and sweep directories exist
mkdir -p "$LOG_DIR"
mkdir -p "$SWEEP_DIR"

#------------------------------------------------------------------------------
# Print job information
#------------------------------------------------------------------------------
echo "=========================================="
echo "Nullloss Validation Sweep"
echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated memory per CPU: $SLURM_MEM_PER_CPU MB"
echo "Number of GPUs: $SLURM_GPUS"
echo "GPU type: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Allocated memory per GPU: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
echo "Working directory: $PWD"
echo "Sweep directory: $SWEEP_DIR"
echo "=========================================="

#------------------------------------------------------------------------------
# Activate Python virtual environment
#------------------------------------------------------------------------------
VENV_DIR="/cluster/home/$USERNAME/AISCO_code/aisco_env"
source "$VENV_DIR/bin/activate"

#------------------------------------------------------------------------------
# Verify GPU access
#------------------------------------------------------------------------------
echo "Verifying GPU access..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

#------------------------------------------------------------------------------
# Set Python environment
#------------------------------------------------------------------------------
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="/cluster/home/$USERNAME/AISCO_code/subprojects:$PYTHONPATH"
cd "$PROJECT_ROOT"

#------------------------------------------------------------------------------
# Progress tracking
#------------------------------------------------------------------------------
TOTAL_SCENARIOS=8
CURRENT_SCENARIO=0
SCENARIO_START_TIME=0

# Function to print progress bar
print_progress_bar() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local filled=$((width * current / total))
    local empty=$((width - filled))

    printf "\r["
    printf "%${filled}s" | tr ' ' '='
    printf "%${empty}s" | tr ' ' '-'
    printf "] %d/%d (%d%%)" "$current" "$total" "$percentage"
}

# Function to format seconds to HH:MM:SS
format_time() {
    local seconds=$1
    printf "%02d:%02d:%02d" $((seconds/3600)) $((seconds%3600/60)) $((seconds%60))
}

#------------------------------------------------------------------------------
# Function to run a single test scenario
#------------------------------------------------------------------------------
run_scenario() {
    local scenario_name=$1
    local description=$2
    shift 2
    local modifications=("$@")

    CURRENT_SCENARIO=$((CURRENT_SCENARIO + 1))
    SCENARIO_START_TIME=$SECONDS

    echo ""
    echo "========================================================================"
    echo "SCENARIO $CURRENT_SCENARIO/$TOTAL_SCENARIOS: $scenario_name"
    echo "========================================================================"
    echo "Description: $description"
    print_progress_bar $CURRENT_SCENARIO $TOTAL_SCENARIOS
    echo ""
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------------------"

    # Create temporary config directory
    TMP_CONFIG_DIR="$SWEEP_DIR/configs_$scenario_name"
    mkdir -p "$TMP_CONFIG_DIR"

    # Copy original configs
    cp "$LOSSES_CONFIG" "$TMP_CONFIG_DIR/losses_v0_config.yaml"
    cp "$TRAINING_CONFIG" "$TMP_CONFIG_DIR/training_v0_config.yaml"

    # Apply modifications to losses config
    for mod in "${modifications[@]}"; do
        echo "  Applying modification: $mod"
        python3 -c "
import yaml
from pathlib import Path

config_path = Path('$TMP_CONFIG_DIR/losses_v0_config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

# Parse modification: 'category.loss_name.field=value'
mod = '$mod'
parts = mod.split('=')
path_parts = parts[0].split('.')
value = int(parts[1]) if parts[1].isdigit() else parts[1]

# Navigate to the target and update
target = config['losses']
for part in path_parts[:-1]:
    target = target[part]

# Update the compute flag (first element of list)
loss_name = path_parts[-2]
field_idx = 0 if path_parts[-1] == 'compute' else (1 if path_parts[-1] == 'weight' else 2)
target[loss_name][field_idx] = value

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
"
    done

    # Set epochs to 5 in training config
    python3 -c "
import yaml
from pathlib import Path

config_path = Path('$TMP_CONFIG_DIR/training_v0_config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

config['training_params']['num_epochs'] = 5

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
"

    # Run training
    echo ""
    echo "🚀 Starting training run..."
    echo "   Log: $SWEEP_DIR/${scenario_name}.log"
    echo ""
    SCENARIO_LOG="$SWEEP_DIR/${scenario_name}.log"

    # Run with real-time progress indicators
    python3 "$MAIN_SCRIPT" \
        --save_model \
        --problem_id="NullLoss_${scenario_name}_${TIMESTAMP}" \
        2>&1 | tee "$SCENARIO_LOG" | grep --line-buffered -E "Epoch|epoch|loss|Loss|===|Error|error" || true

    SCENARIO_EXIT_CODE=${PIPESTATUS[0]}
    SCENARIO_DURATION=$((SECONDS - SCENARIO_START_TIME))

    echo ""
    echo "------------------------------------------------------------------------"
    if [ $SCENARIO_EXIT_CODE -eq 0 ]; then
        echo "✅ Scenario completed successfully"
    else
        echo "❌ Scenario failed (exit code: $SCENARIO_EXIT_CODE)"
        echo "   See log: $SCENARIO_LOG"
    fi
    echo "   Duration: $(format_time $SCENARIO_DURATION)"
    echo "   Finished at: $(date '+%Y-%m-%d %H:%M:%S')"

    # Update overall progress
    echo ""
    print_progress_bar $CURRENT_SCENARIO $TOTAL_SCENARIOS

    TOTAL_ELAPSED=$SECONDS
    AVG_TIME_PER_SCENARIO=$((TOTAL_ELAPSED / CURRENT_SCENARIO))
    REMAINING_SCENARIOS=$((TOTAL_SCENARIOS - CURRENT_SCENARIO))
    EST_REMAINING=$((AVG_TIME_PER_SCENARIO * REMAINING_SCENARIOS))

    echo ""
    echo "   Total elapsed: $(format_time $TOTAL_ELAPSED)"
    if [ $REMAINING_SCENARIOS -gt 0 ]; then
        echo "   Estimated remaining: ~$(format_time $EST_REMAINING)"
        echo "   Estimated completion: $(date -d "@$(($(date +%s) + EST_REMAINING))" '+%Y-%m-%d %H:%M:%S')"
    fi
    echo "========================================================================"
}

#------------------------------------------------------------------------------
# Test Scenarios
#------------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "🎯 Starting Nullloss Validation Sweep"
echo "=========================================="
echo "Total scenarios: $TOTAL_SCENARIOS"
echo "Epochs per scenario: 5"
echo "Expected duration: ~2 hours"
echo ""
print_progress_bar 0 $TOTAL_SCENARIOS
echo ""
echo "=========================================="

# Scenario 1: Baseline (no nulllosses except solar to avoid memory issues)
run_scenario "baseline" "All regular losses active (solar null to avoid OOM)" \
    "solar.null_loss_solar_v1.compute=1" \
    "solar.solar_v3B_2.compute=0" \
    "solar.solar_v4.compute=0" \
    "solar.solar_v5.compute=0" \
    "solar.solar_v6.compute=0" \
    "solar.solar_v7.compute=0"

# Scenario 2: Aero nullloss (with solar null to avoid memory issues)
run_scenario "aero_null" "Aero nullloss active (solar null for OOM)" \
    "aero.null_loss_aero_v1.compute=1" \
    "aero.gaot_loss.compute=0" \
    "solar.null_loss_solar_v1.compute=1" \
    "solar.solar_v3B_2.compute=0" \
    "solar.solar_v4.compute=0" \
    "solar.solar_v5.compute=0" \
    "solar.solar_v6.compute=0" \
    "solar.solar_v7.compute=0"

# Scenario 3: Solar nullloss (already included, kept for consistency)
run_scenario "solar_null" "Solar nullloss active" \
    "solar.null_loss_solar_v1.compute=1" \
    "solar.solar_v3B_2.compute=0" \
    "solar.solar_v4.compute=0" \
    "solar.solar_v5.compute=0" \
    "solar.solar_v6.compute=0" \
    "solar.solar_v7.compute=0"

# Scenario 4: BB nullloss (with solar null to avoid memory issues)
run_scenario "bb_null" "BB nullloss active (solar null for OOM)" \
    "bb.null_loss_bb_v1.compute=1" \
    "bb.bb_v4_4.compute=0" \
    "solar.null_loss_solar_v1.compute=1" \
    "solar.solar_v3B_2.compute=0" \
    "solar.solar_v4.compute=0" \
    "solar.solar_v5.compute=0" \
    "solar.solar_v6.compute=0" \
    "solar.solar_v7.compute=0"

# Scenario 5: Eikonal nullloss (with solar null to avoid memory issues)
run_scenario "eikonal_null" "Eikonal nullloss active (solar null for OOM)" \
    "eikonal.null_loss_eikonal_v0.compute=1" \
    "eikonal.eikonal_v0.compute=0" \
    "solar.null_loss_solar_v1.compute=1" \
    "solar.solar_v3B_2.compute=0" \
    "solar.solar_v4.compute=0" \
    "solar.solar_v5.compute=0" \
    "solar.solar_v6.compute=0" \
    "solar.solar_v7.compute=0"

# Scenario 6: Smoothness nullloss (with solar null to avoid memory issues)
run_scenario "smoothness_null" "Smoothness nullloss active (solar null for OOM)" \
    "smoothness.null_loss_smoothness_v0.compute=1" \
    "smoothness.smoothness_v0.compute=0" \
    "solar.null_loss_solar_v1.compute=1" \
    "solar.solar_v3B_2.compute=0" \
    "solar.solar_v4.compute=0" \
    "solar.solar_v5.compute=0" \
    "solar.solar_v6.compute=0" \
    "solar.solar_v7.compute=0"

# Scenario 7: Developability nullloss (with solar null to avoid memory issues)
run_scenario "developability_null" "Developability nullloss active (solar null for OOM)" \
    "developability.null_loss_developability_v0.compute=1" \
    "developability.developability_v0.compute=0" \
    "solar.null_loss_solar_v1.compute=1" \
    "solar.solar_v3B_2.compute=0" \
    "solar.solar_v4.compute=0" \
    "solar.solar_v5.compute=0" \
    "solar.solar_v6.compute=0" \
    "solar.solar_v7.compute=0"

# Scenario 8: BB + Developability nulllosses (with solar null to avoid memory issues)
run_scenario "bb_and_dev_null" "BB and Developability nulllosses active (solar null for OOM)" \
    "bb.null_loss_bb_v1.compute=1" \
    "bb.bb_v4_4.compute=0" \
    "developability.null_loss_developability_v0.compute=1" \
    "developability.developability_v0.compute=0" \
    "solar.null_loss_solar_v1.compute=1" \
    "solar.solar_v3B_2.compute=0" \
    "solar.solar_v4.compute=0" \
    "solar.solar_v5.compute=0" \
    "solar.solar_v6.compute=0" \
    "solar.solar_v7.compute=0"

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "🎉 Nullloss Validation Sweep Complete!"
echo "=========================================="
print_progress_bar $TOTAL_SCENARIOS $TOTAL_SCENARIOS
echo ""
echo ""
echo "Job ended on: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total elapsed time: $(format_time $SECONDS)"
echo ""

# Count successes and failures
SUCCESS_COUNT=$(grep -c "✅ Scenario completed successfully" "$LOG_DIR/NullLoss_Validation_${SLURM_JOB_ID}.out" 2>/dev/null || echo "0")
FAIL_COUNT=$(grep -c "❌ Scenario failed" "$LOG_DIR/NullLoss_Validation_${SLURM_JOB_ID}.out" 2>/dev/null || echo "0")

echo "Results:"
echo "  ✅ Successful scenarios: $SUCCESS_COUNT/$TOTAL_SCENARIOS"
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "  ❌ Failed scenarios: $FAIL_COUNT/$TOTAL_SCENARIOS"
fi
echo ""
echo "Output locations:"
echo "  📁 Sweep directory: $SWEEP_DIR"
echo "  📄 SLURM log: $LOG_DIR/NullLoss_Validation_${SLURM_JOB_ID}.out"
echo "  📊 Scenario logs: $SWEEP_DIR/*.log"
echo "  ⚙️  Configs: $SWEEP_DIR/configs_*/"
echo ""
echo "Next steps:"
echo "  1. Check scenario logs: tail -20 $SWEEP_DIR/*.log"
echo "  2. View wandb runs: https://wandb.ai/aisco (search for 'NullLoss_')"
echo "  3. Verify metrics show zero for nulllosses"
echo "=========================================="
