# just the models class which calls other models here
import torch
import torch.nn.functional as F
from torch import nn


class BaseModel(nn.Module):
    def __init__(self, config: dict, in_dim: int=3, hidden: int=64, layers: int=3):
        
        super(BaseModel, self).__init__()

        self.config = config
        self.in_dim = in_dim
        self.hidden = hidden
        self.layers = layers    


        
        #Define a simple MLP with Relu and Tanh activations
        self.model_layers = nn.ModuleList()
        self.model_layers.append(nn.Linear(in_dim, hidden))
        
        for i in range(layers - 1):
            self.model_layers.append(nn.Linear(hidden, hidden))
            #Alternate activations
            if i % 2:
                self.model_layers.append(nn.Tanh())
            else:
                self.model_layers.append(nn.ReLU())
        
        self.model_layers.append(nn.Linear(hidden, 1))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 1000.0  # Scale input to meters
        for layer in self.model_layers:
            x = layer(x)
        output = x.squeeze(-1)
        return output * 1000.0
    
    def optimise(self):
        super(BaseModel, self).train() #Set training attribute of inherited model to true, ensures things like e.g. dropout work properly

        # For textbook example of how our optimization works, look at basic_MLP_v1.py
        raise NotImplementedError("Train method not implemented yet in BaseModel. Please implement in the subclass inheriting.")
        
    
    def validation(self):
        super(BaseModel, self).eval() #Set training attribute of inherited model to false, ensures things like e.g. dropout are disabled
        #Validation Logic TODO
        return
    

    def get_model_info_dict(self):
    
        #Return model info as a dictionary
        model_info = {
            "config": self.config,
            "in_dim": self.in_dim,
            "hidden": self.hidden,
            "layers": self.layers,
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }
        return model_info

    def debug_info(self):
        #Print model architecture and number of parameters, or anthing you might want to debug
        print("===================================")
        print("Model Debug Info:")

        print(self)
        
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters())}")
        print("Model Layers:")
        for i, layer in enumerate(self.model_layers):
            print(f" Layer {i}: {layer}")
        
        print("Device/Gradient Info:")
        print(f" Model is on device: {self.device}")
        print(f"Do Model Parameters require Gradients? (Y/N): {any(p.requires_grad for p in self.parameters())}")

        print("End of Model Debug Info")
        print("===================================")
        return