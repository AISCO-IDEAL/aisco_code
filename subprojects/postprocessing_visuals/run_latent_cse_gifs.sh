#!/usr/bin/env bash
set -euo pipefail




python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
    --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/114_407963_logical-meadow/models/114_407963_logical-meadow_epoch6400.pth" \
    --out /home/aisco/AISCO_rollout/wp9/4k/solar_epoch.gif \
    --show-solar-selection \
    --n_frames 20 \
    --epoch_gif

python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
    --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/114_407963_logical-meadow/models/114_407963_logical-meadow_epoch6400.pth" \
    --out /home/aisco/AISCO_rollout/wp9/4k/drag_epoch.gif \
    --show-pressure-field \
    --n_frames 20 \
    --epoch_gif

python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
    --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/114_407963_logical-meadow/models/114_407963_logical-meadow_epoch6400.pth" \
    --out /home/aisco/AISCO_rollout/wp9/4k/windshield_only_epoch.gif \
    --n_frames 20 \
    --epoch_gif

python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
    --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/114_407963_logical-meadow/models/114_407963_logical-meadow_epoch6400.pth" \
    --out /home/aisco/AISCO_rollout/wp9/4k/solar_transparent_epoch.gif \
    --show-solar-selection \
    --transparent-background \
    --n_frames 2 \
    --epoch_gif

###

python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
    --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/114_407963_logical-meadow/models/114_407963_logical-meadow_epoch6400.pth" \
    --out /home/aisco/AISCO_rollout/wp9/4k/latent/solar.gif \
    --show-solar-selection \
    --n_frames 20 \


python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
    --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/114_407963_logical-meadow/models/114_407963_logical-meadow_epoch6400.pth" \
    --out /home/aisco/AISCO_rollout/wp9/4k/latent/drag.gif \
    --show-pressure-field \
    --n_frames 20 \


python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
    --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/114_407963_logical-meadow/models/114_407963_logical-meadow_epoch6400.pth" \
    --out /home/aisco/AISCO_rollout/wp9/4k/latent/windshield_only.gif \
    --n_frames 20 \


python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista2.py \
    --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/114_407963_logical-meadow/models/114_407963_logical-meadow_epoch6400.pth" \
    --out /home/aisco/AISCO_rollout/wp9/4k/latent/solar_transparent.gif \
    --show-solar-selection \
    --transparent-background \
    --n_frames 2 \


