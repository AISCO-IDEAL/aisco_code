#!/bin/bash

# Pfad zu deinen Skripten
BASE_PATH="/home/eschaerer/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/slurm_scripts/matthew_pc"

# Pfad, in dem die Logs gespeichert werden sollen
LOG_PATH="/home/eschaerer/AISCO_code/logs"

# Erstelle den Log-Ordner, falls er noch nicht existiert
mkdir -p "$LOG_PATH"


# ==============================================================================
# AUSFÜHRUNG
# ==============================================================================

echo "--- Starting batch processing at $(date) ---"

# Job 0
echo "[1/6] Starting wp1_main_local.sh..."
bash "$BASE_PATH/wp1_main_local.sh" > "$LOG_PATH/log_0.txt" 2>&1

# Job 1
echo "[2/6] Starting wp1_main_local_1.sh..."
bash "$BASE_PATH/wp1_main_local_1.sh" > "$LOG_PATH/log_1.txt" 2>&1

# Job 2
echo "[3/6] Starting wp1_main_local_2.sh..."
bash "$BASE_PATH/wp1_main_local_2.sh" > "$LOG_PATH/log_2.txt" 2>&1

# Job 3
echo "[4/6] Starting wp1_main_local_3.sh..."
bash "$BASE_PATH/wp1_main_local_3.sh" > "$LOG_PATH/log_3.txt" 2>&1

# Job 4
echo "[5/6] Starting wp1_main_local_4.sh..."
bash "$BASE_PATH/wp1_main_local_4.sh" > "$LOG_PATH/log_4.txt" 2>&1

# Job 5
echo "[6/6] Starting wp1_main_local_5.sh..."
bash "$BASE_PATH/wp1_main_local_5.sh" > "$LOG_PATH/log_5.txt" 2>&1

echo "--- All Jobs finished at $(date) ---"