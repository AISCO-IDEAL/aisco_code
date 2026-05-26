#!/usr/bin/env bash
set -euo pipefail




# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#     --out /home/aisco/AISCO_rollout/wp9/4k/final/windshield_only_epoch.gif \
#     --epoch_gif

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#     --out /home/aisco/AISCO_rollout/wp9/4k/final/windshield_only_latent.gif \
#########################

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#      --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#      --out /home/aisco/AISCO_rollout/wp9/4k/präsi/-0.601_0.133/latent/cross_latent_recentered.gif \
#      --transparent-background \
#      --mode cross \
#      --cross_center -0.601 0.133 \
#      --n_frames 20

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#      --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#      --out /home/aisco/AISCO_rollout/wp9/4k/präsi/-0.601_0.133/latent/spiral_latent_recentered.gif \
#      --transparent-background \
#      --spiral_center -0.601 0.133 \
#      --spiral_radius_end 1.5 \
#      --n_frames 20

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#      --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#      --out /home/aisco/AISCO_rollout/wp9/4k/präsi/-0.601_0.133/latent/cross_latent_fast_forward.gif \
#      --transparent-background \
#      --mode cross \
#      --cross_fast_forward_to -0.601 0.133 \
#      --n_frames 20

python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
     --out /home/aisco/AISCO_rollout/wp9/4k/präsi/KEINE_WINDSCHUTZSCHEIBE/3_Slide_Lösungsraum.gif \
     --transparent-background \
     --spiral_fast_forward_to -0.601 0.133 \
     --no-show-epoch-label \
     --no-show-windshield \

python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
    --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
    --out /home/aisco/AISCO_rollout/wp9/4k/website/solar_selection.gif \
    --show-solar-selection \

python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
    --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
    --out /home/aisco/AISCO_rollout/wp9/4k/website/solar_quality.gif \
    --show-solar-quality-mask \

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#      --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#      --out /home/aisco/AISCO_rollout/wp9/4k/präsi/-0.601_0.133/epoch/shell_epoch.gif \
#      --no-show-latent-panel \
#      --transparent-background \
#      --epoch_gif \
#      --epoch_coords -0.601 0.133




#  python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#      --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#      --out /home/aisco/AISCO_rollout/wp9/4k/präsi/-0.601_0.133/solar_quality/solar_quality_no_windshield.gif \
#      --no-show-windshield \
#      --n_frames 1 \
#      --no-show-cse-overlay \
#      --no-show-latent-panel \
#      --transparent-background \
#      --no-show-epoch-label \
#      --mode line \
#      --line_start -0.601 0.133\
#      --line_end -0.601 0.133\
#      --show-solar-quality-mask

#  python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#      --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#      --out /home/aisco/AISCO_rollout/wp9/4k/präsi/-0.601_0.133/solar_quality/solar_quality.gif \
#      --no-show-windshield \
#      --n_frames 1 \
#      --no-show-cse-overlay \
#      --no-show-latent-panel \
#      --transparent-background \
#      --no-show-epoch-label \
#      --mode line \
#      --line_start -0.601 0.133\
#      --line_end -0.601 0.133\
#      --show-solar-quality-mask

#  python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#      --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#      --out /home/aisco/AISCO_rollout/wp9/4k/präsi/-0.601_0.133/cfd_no_windshield.gif \
#      --no-show-windshield \
#      --n_frames 1 \
#      --no-show-cse-overlay \
#      --no-show-latent-panel \
#      --transparent-background \
#      --no-show-epoch-label \
#      --show-pressure-field \
#      --mode line \
#      --line_start -0.601 0.133\
#      --line_end -0.601 0.133\

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#     --out /home/aisco/AISCO_rollout/wp9/4k/präsi/-0.601_0.133/cfd_only.gif \
#     --n_frames 1 \
#     --no-show-cse-overlay \
#     --no-show-latent-panel \
#     --transparent-background \
#     --no-show-epoch-label \
#     --show-pressure-field \
#     --mode line \
#     --line_start -0.601 0.133\
#     --line_end -0.601 0.133\

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#     --out /home/aisco/AISCO_rollout/wp9/4k/präsi/-0.601_0.133/solar_selection_only.gif \
#     --show-solar-selection \
#     --n_frames 1 \
#     --no-show-cse-overlay \
#     --no-show-latent-panel \
#     --transparent-background \
#     --no-show-epoch-label \
#     --mode line \
#     --line_start -0.601 0.133\
#     --line_end -0.601 0.133\

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#     --out /home/aisco/AISCO_rollout/wp9/4k/präsi/-0.601_0.133/solar_selection_no_windshield.gif \
#     --no-show-windshield \
#     --n_frames 1 \
#     --no-show-cse-overlay \
#     --no-show-latent-panel \
#     --transparent-background \
#     --no-show-epoch-label \
#     --show-solar-selection \
#     --mode line \
#     --line_start -0.601 0.133\
#     --line_end -0.601 0.133\

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#     --out /home/aisco/AISCO_rollout/wp9/4k/präsi/-0.601_0.133/car_with_CSE.gif \
#     --n_frames 1 \
#     --no-show-latent-panel \
#     --transparent-background \
#     --no-show-epoch-label \
#     --mode line \
#     --line_start -0.601 0.133\
#     --line_end -0.601 0.133\

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#     --out /home/aisco/AISCO_rollout/wp9/4k/präsi/-0.601_0.133/car_with_CSE_no_windshield.gif \
#     --no-show-windshield \
#     --n_frames 1 \
#     --no-show-latent-panel \
#     --transparent-background \
#     --no-show-epoch-label \
#     --mode line \
#     --line_start -0.601 0.133\
#     --line_end -0.601 0.133 \

# #########################
# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#      --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#      --out /home/aisco/AISCO_rollout/wp9/4k/präsi/0.5_-0.5/cfd_no_windshield.gif \
#      --no-show-windshield \
#      --n_frames 1 \
#      --no-show-cse-overlay \
#      --no-show-latent-panel \
#      --transparent-background \
#      --no-show-epoch-label \
#      --show-pressure-field \
#      --mode line \
#      --line_start 0.552 -0.562 \
#      --line_end 0.552 -0.562 \

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#     --out /home/aisco/AISCO_rollout/wp9/4k/präsi/0.5_-0.5/cfd_only.gif \
#     --n_frames 1 \
#     --no-show-cse-overlay \
#     --no-show-latent-panel \
#     --transparent-background \
#     --no-show-epoch-label \
#     --show-pressure-field \
#     --mode line \
#     --line_start 0.552 -0.562 \
#     --line_end 0.552 -0.562 \

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#     --out /home/aisco/AISCO_rollout/wp9/4k/präsi/0.5_-0.5/solar_selection_only.gif \
#     --show-solar-selection \
#     --n_frames 1 \
#     --no-show-cse-overlay \
#     --no-show-latent-panel \
#     --transparent-background \
#     --no-show-epoch-label \
#     --mode line \
#     --line_start 0.552 -0.562 \
#     --line_end 0.552 -0.562 \

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#     --out /home/aisco/AISCO_rollout/wp9/4k/präsi/0.5_-0.5/solar_selection_no_windshield.gif \
#     --no-show-windshield \
#     --n_frames 1 \
#     --no-show-cse-overlay \
#     --no-show-latent-panel \
#     --transparent-background \
#     --no-show-epoch-label \
#     --show-solar-selection \
#     --mode line \
#     --line_start 0.552 -0.562 \
#     --line_end 0.552 -0.562 \

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#     --out /home/aisco/AISCO_rollout/wp9/4k/präsi/0.5_-0.5/car_with_CSE.gif \
#     --n_frames 1 \
#     --no-show-latent-panel \
#     --transparent-background \
#     --no-show-epoch-label \
#     --mode line \
#     --line_start 0.552 -0.562 \
#     --line_end 0.552 -0.562 \

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#     --out /home/aisco/AISCO_rollout/wp9/4k/präsi/0.5_-0.5/car_with_CSE_no_windshield.gif \
#     --no-show-windshield \
#     --n_frames 1 \
#     --no-show-latent-panel \
#     --transparent-background \
#     --no-show-epoch-label \
#     --mode line \
#     --line_start 0.552 -0.562 \
#     --line_end 0.552 -0.562 \
############

# python3 /home/aisco/Documents/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif_pyvista.py \
#     --checkpoint "/home/aisco/AISCO_rollout/wp9/for_victor/6166b4e953/111_401543_valiant-jazz/models/111_401543_valiant-jazz_epoch7380.pth" \
#     --out /home/aisco/AISCO_rollout/wp9/4k/website/solar_selection.gif \
#     --mesh_alpha 0.0 \
#     --no-show-windshield \
#     --n_frames 1 \
#     --no-show-cse-overlay \
#     --no-show-latent-panel \
#     --transparent-background \
#     --no-show-epoch-label
