#Kolja code here

import torch
import importlib



from subprojects.wp4_aeroshell_optimizer.v1.src.models.WIRE import WIRE
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.ground_truth_masks.simple_geoms import sphere_gt
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_general_config


def dynamic_loading(config_name: str="pretraining_v0_config.yaml"):
    """
    Dynamically loads the model architecture based on the configuration file.
    I.e. imports only from the relevant file, not all the models
    """
    config = load_general_config(cfg_name=config_name) #not yet implemented there!
    architecture_name = config['model'] #dynamically import the model module
    module = importlib.import_module("subprojects.wp4_aeroshell_optimizer.v1.src.models." + architecture_name)
    ModelClass = getattr(module, architecture_name) #gets model class


    #adjust to include config parameters here
    model = ModelClass()
    return model    


def pretraining(config_name: str="pretraining_v0_config.yaml"):
    
    # Does pretraining, for now only on a sphere

    config = load_general_config(cfg_name=config_name)
    model = dynamic_loading(config_name)

    # Try to use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    #Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training_loop']['learning_rate'])
    for i in range(config['training_loop']['epochs']):
        optimizer.zero_grad()
        
        
        # Randomly sample spatial points within a certain range (Normal dist with mu = 0, sigma=5000)
        input_data = torch.randn(3,config['training_loop']['random_batch_size'],
                                device=device)*5000
        
        output = model(input_data)
        
        # Dummy target data, replace with actual target generation
        target = sphere_gt(input_data)
        
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if i % config['training_loop']['print_interval'] == 0:
            print(f"Epoch {i}, Loss: {loss.item()}")
    

    return model



if __name__ == "__main__":
    config_name = "pretraining_v0_config.yaml"
    pretraining(config_name)