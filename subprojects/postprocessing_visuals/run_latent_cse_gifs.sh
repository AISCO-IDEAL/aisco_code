#!/usr/bin/env bash
set -euo pipefail

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/114_407963_logical-meadow/models/114_407963_logical-meadow_epoch3660.pth" \
#     --resolution 60 \
#     --fps 24 \
#     --mode spiral \
#     --show_assembly \
#     --height 2160 \
#     --width 2560 \
#     --panel_width 1280 \
#     --no-show-anchors \
#     --out /home/aisco/AISCO_rollout/wp9/4k/latent_CSE_gif.gif \
#     --show-cse-overlay \
#     --show-cse-kmh \
#     --spiral-equal-distance

python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
    --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/114_407963_logical-meadow/models/114_407963_logical-meadow_epoch6400.pth" \
    --resolution 96 \
    --fps 24 \
    --mode spiral \
    --show_assembly \
    --height 2160 \
    --width 3840 \
    --panel_width 1280 \
    --no-show-anchors \
    --out /home/aisco/AISCO_rollout/wp9/4k/latent_CSE_pressure_gif.gif \
    --show-cse-overlay \
    --show-cse-kmh \
    --show-cse-tacho \
    --show-pressure-field \
   
python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
    --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/114_407963_logical-meadow/models/114_407963_logical-meadow_epoch6400.pth" \
    --resolution 96 \
    --fps 24 \
    --mode spiral \
    --show_assembly \
    --height 2160 \
    --width 3840 \
    --panel_width 1280 \
    --no-show-anchors \
    --out /home/aisco/AISCO_rollout/wp9/4k/latent_CSE_gif.gif \
    --show-cse-overlay \
    --show-cse-kmh \
    --show-cse-tacho \

python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
    --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/114_407963_logical-meadow/models/114_407963_logical-meadow_epoch6400.pth" \
    --resolution 96 \
    --fps 24 \
    --mode spiral \
    --show_assembly \
    --height 2160 \
    --width 3840 \
    --panel_width 1280 \
    --no-show-anchors \
    --out /home/aisco/AISCO_rollout/wp9/4k/training_CSE_pressure_gif.gif \
    --show-cse-overlay \
    --show-cse-kmh \
    --show-cse-tacho \
    --show-pressure-field \
    --epoch_gif \
   
python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
    --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/114_407963_logical-meadow/models/114_407963_logical-meadow_epoch6400.pth" \
    --resolution 96 \
    --fps 24 \
    --mode spiral \
    --show_assembly \
    --height 2160 \
    --width 3840 \
    --panel_width 1280 \
    --no-show-anchors \
    --out /home/aisco/AISCO_rollout/wp9/4k/training_CSE_gif.gif \
    --show-cse-overlay \
    --show-cse-kmh \
    --show-cse-tacho \
    --epoch_gif \
   


# For epoch_gif mode, use e.g.:
# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/114_407963_logical-meadow/models/114_407963_logical-meadow_epoch3660.pth" \
#     --mode epoch_gif \
#     --show_assembly \
#     --show-cse-overlay \
#     --show-cse-kmh \
#     --show-cse-tacho \
#     --n_frames 4 \
#     --out /home/aisco/AISCO_rollout/wp9/4k/test_epoch_gif.gif
# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/114_407963_logical-meadow/models/114_407963_logical-meadow_epoch3660.pth" \
#     --resolution 60 \
#     --fps 24 \
#     --mode spiral \
#     --show_assembly \
#     --height 2160 \
#     --width 2560 \
#     --panel_width 1280 \
#     --no-show-anchors \
#     --out /home/aisco/AISCO_rollout/wp9/4k/latent_CSE_gif_fast_spiral.gif \
#     --show-cse-overlay \
#     --show-cse-kmh \
#     --no-spiral-equal-distance
