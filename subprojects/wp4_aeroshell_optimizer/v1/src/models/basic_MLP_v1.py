from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
import torch
from torch import nn


class BasicMLP(BaseModel):
    def __init__(self, config: dict = {}, in_dim: int=3, hidden: int=128, layers: int=6):
        super(BasicMLP, self).__init__(config, in_dim, hidden, layers)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super(BasicMLP, self).forward(x)
    
    def optimise(self, training_config: dict = None):
        if training_config is None:
            raise ValueError("Training config must be provided for optimisation.")
        return 