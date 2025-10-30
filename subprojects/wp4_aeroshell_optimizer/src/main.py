import yaml
from train import train_sdf_model
from sdf import SDFNetwork
from SDF_visualisation import visualize_loaded_model
import kaolin_test
import torch
def main():
    # 1. Load settings from config:
    with open('AISCO-code/subprojects/wp4_aeroshell_optimizer/src/wp4_config.yaml', 'r') as f:
        config = yaml.safe_load(f)


    load_pretrained = config["use_pretrained"]

    if load_pretrained:
        #This loads a model while we redo the training loop and main structure
        
        model=torch.load("AISCO-code/subprojects/wp4_aeroshell_optimizer/src/DummySphere/sdf_sphere_model_2025-10-24.pth")
    else:
        model=SDFNetwork()
    model.to('cuda')

    train_sdf_model(model,config)
    
    

    return




if __name__ == "__main__":
    main()