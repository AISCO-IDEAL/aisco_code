# The purpose of this file is to wrap the visualiser class and visualiser utilities into one nice thing that is compactly callable.
# What is in this file is supposed to be hassle free for the user! --> Config settable parameters only!
# If you want some custom stuff, dig yourself in visualiser.py or visualiser_utils.py, or even better implement your stuff in here!

from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser import Visualiser
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser_utils import *
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
class Wrapper(Visualiser):
    def __init__(self,

                verbose: bool = True):
        # Initialise Visualiser class
        self.config = load_config(load_config(cfg_name="general_config.yaml")["active_configs"]["neural_renderer"])
        super().__init__(**(self.config.get("neural_renderer", {})))

        if verbose:
            print("Initialized Visual Wrapper by loading config:")
            self.status()
            print("Starting initialisation of other stuff")


        # handle storage solutions here and other stuff you might want to have in the wrapper

        #create dict that stores for each enabled mode any frames you want to store during optimization
        #if you want to store stuff for gif making later, do it here and don't create separate lists anywhere unless you have a good reason
        self.stored_frames_dict = {}
        for mode, enabled in self.config.get("modes", {}).items():
            if enabled:
                self.stored_frames_dict[mode] = []



    def status(self):

        # get only the true values in config["modes"]
        print(f"The currently active modes are: {[k for k, v in self.config.get('modes', {}).items() if v]}")

    def update(self):
        # any update logic you want to have per optimization step
        pass

    def finalize(self):
        pass