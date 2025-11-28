#WIRE model

from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
import torch
from torch import nn

"""
WIRE architecture from https://arxiv.org/pdf/2301.05187

This code is from the GINN PAPER repo: https://github.com/ml-jku/GINNs-Geometry-informed-Neural-Networks/blob/main/models/wire.py
Modified by Alex to inherit from BaseModel class instead of nn.Module (BaseModel already inherits from nn.Module)

Note that this implementation uses complex numbers!
"""


class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=40.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())


class WIRE(BaseModel):
    def __init__(self, config = {}, in_features = 3, hidden_features=256, 
                 hidden_layers=16, 
                 out_features = 1, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=1.0, scale=1.0,
                 trainable_params = True,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__(config=config) #Initialize config to satisfy BaseModel inheritance (does nothing for now)
        
         # variables needed for model info stored here
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.trainable_params = trainable_params
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.scale = scale
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer
        
        # Since complex numbers are two real numbers, reduce the number of 
        # hidden parameters by 2
        hidden_features = int(hidden_features/torch.sqrt(torch.tensor(2.0)))
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = 'gabor'
        
        # Legacy parameter
        self.pos_encode = False
        
        self.net = []
        self.net.append(self.nonlin(in_features,
                                    hidden_features, 
                                    omega0=first_omega_0,
                                    sigma0=scale,
                                    is_first=True,
                                    trainable=self.trainable_params))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features, 
                                        omega0=hidden_omega_0,
                                        sigma0=scale,
                                        trainable=self.trainable_params))

        final_linear = nn.Linear(hidden_features,
                                 out_features,
                                 dtype=dtype)            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)

        # Small-weight init for stability: limit Linear weights, zero biases
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords=coords/1000 #Scale mm to meters, nicer for activations
        output = self.net(coords)
        
        if self.wavelet == 'gabor':
            return 1000*(output.real)  #Scale back to mm
        else:
            return output*1000 #Scale back to mm
    
    def get_model_info_dict(self):
        model_info = {
            "config" : self.config,
            "in_features": self.in_features,
            "hidden_features": self.hidden_features,
            "hidden_layers": self.hidden_layers,
            "out_features": self.out_features,
            "trainable_params": self.trainable_params,
            "first_omega_0":self.first_omega_0,
            "hidden_omega_0":self.hidden_omega_0,
            "scale":self.scale,
            "total_parameters": sum(p.numel() for p in self.parameters()),

        }
        return model_info