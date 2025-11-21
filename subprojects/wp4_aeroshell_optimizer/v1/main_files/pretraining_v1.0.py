#Kolja code here

import torch
import importlib.import_module



from subprojects.wp4_aeroshell_optimizer.v1.src.models.WIRE import WIRE
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.ground_truth_masks.simple_geoms import sphere_gt
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config


def dynamic_loading(config_name: str="pretraining_config.yaml"):
    config = load_config(config_name) #not yet implemented there!
    architecture_name = config['model'] #dynamically import the model module
    module = importlib.import_module("subprojects.wp4_aeroshell_optimizer.v1.src.models." + architecture_name)
    ModelClass = getattr(module, architecture_name) #gets model class


    #adjust to include config parameters here
    model = ModelClass()
    return model    


def pretraining(config_name: str="pretraining_config.yaml"):
    
    config = load_config(config_name)
    model = dynamic_loading(config_name)
    


    return model



if __name__ == "__main__":
    config_name = "pretraining_config.yaml"
    pretraining(config_name)