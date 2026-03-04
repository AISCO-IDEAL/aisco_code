To run this Solar estimaor, just run WP4 with the solar estimator v2 activated.
To configurate this loss got to WP4 losses config. There you can ajust the following options:
- solar_config:
  - wandb_logging: True # If set to True it loggs the following values: solar_final_loss, solar_selected_area_m2, total_area_of_mesh_m2
  - rendering: True # If wandb_logging is also set to True it loggs a visualisation of the mesh with the Bounding Boxes and a visualisation of the Solar area on the mesh
  - rendering_frequency: 1 # If the previous two are set to true, this value decides how often to render on the following condition: epoch_name % rendering_frequency == 0
  - area_budget: 6000000 # in mm^2
  - normalise_loss: True # If True: Loss between 0 (=No good area for Solar Cells) and 1 (=area in size of area_budget is perfectly oriented), but Gradients still can reach values in excess of 1.
  - solar_direction_solid_angle: [0.785398, 3.141592] # Here you can set the direction of the sun (seen from the car) units: rad
