#!/usr/bin/env bash
set -euo pipefail

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#     --out /home/aisco/AISCO_rollout/wp9/4k/final/windshield_only_epoch.gif \
#     --epoch_gif

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#     --out /home/aisco/AISCO_rollout/wp9/4k/final/windshield_only_latent.gif \

python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
    --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
    --out /home/aisco/AISCO_rollout/wp9/4k/website/solar_selection.gif \
    --mesh_alpha 0.0 \
    --no-show-windshield \
    --n_frames 1
    