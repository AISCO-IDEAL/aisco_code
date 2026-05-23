#!/usr/bin/env bash
set -euo pipefail

python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
    --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/114_407963_logical-meadow/models/114_407963_logical-meadow_epoch3660.pth" \
    --resolution 60 \
    --fps 24 \
    --mode spiral \
    --show_assembly \
    --height 2160 \
    --width 2560 \
    --panel_width 1280 \
    --no-show-anchors \
    --out /home/aisco/AISCO_rollout/wp9/4k/latent_CSE_gif.gif \
    --show-cse-overlay \
    --show-cse-kmh

python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
    --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/114_407963_logical-meadow/models/114_407963_logical-meadow_epoch3660.pth" \
    --resolution 60 \
    --fps 24 \
    --mode spiral \
    --show_assembly \
    --height 2160 \
    --width 2560 \
    --panel_width 1280 \
    --no-show-anchors \
    --out /home/aisco/AISCO_rollout/wp9/4k/latent_CSE_gif.gif \
    --show-cse-overlay \
    --show-cse-kmh \
    --epoch_gif
