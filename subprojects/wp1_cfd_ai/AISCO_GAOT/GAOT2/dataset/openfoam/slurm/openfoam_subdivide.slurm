#!/bin/bash
# =============================================================================
# Loop Subdivision of all OpenFOAM geometries
#
# Applies 2 iterations of Loop subdivision to all 234 PLY files.
# Output goes to of_geom_l2/ — completely isolated from current production.
#
# Expected runtime: ~30-60 min for 234 meshes (largest ~116K → ~1.86M verts)
# =============================================================================

#SBATCH --job-name=OF_Subdivide
#SBATCH --partition=normal.4h
#SBATCH --account=public
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/%u/scratch_logs/OF_Subdivide_%j.out
#SBATCH --error=/cluster/scratch/%u/scratch_logs/OF_Subdivide_%j.err

set -euo pipefail

echo "=== Loop Subdivision at $(date) ==="
echo "Job ID: ${SLURM_JOB_ID}"

# Activate Python environment
source /cluster/home/$USER/AISCO_code/aisco_env/bin/activate

SCRIPT_DIR="/cluster/home/$USER/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/dataset/openfoam"
INPUT_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/openfoam_geometry"
OUTPUT_DIR="/cluster/work/fuge/AISCO/wp1_cfd_ai/of_geom_l2"

python "${SCRIPT_DIR}/subdivide_all_geoms.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --target-verts 180000 \
    --workers 16

echo "=== Subdivision complete at $(date) ==="
echo "Output: ${OUTPUT_DIR}"
ls "${OUTPUT_DIR}" | wc -l
echo "files written."
