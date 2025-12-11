# The purpose of this file is to assemble the visualiser class and visualiser utilities
# What is in this file is supposed to be hassle free! 
# If you want some custom stuff, dig yourself in visualiser.py or visualiser_utils.py, or even better implement your stuff in here!

from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser import Visualiser
class Wrapper(Visualiser):
    def __init__(self,
                image_size = 256,
                background_color = None,
                anti_aliasing = True, 
                camera_mode = "look_at",  
                device = "cuda"):
        # Initialise Visualiser class
        super().__init__(image_size, background_color, anti_aliasing, camera_mode, device)

        # handle storage solutions here and other stuff you might want to have in the wrapper