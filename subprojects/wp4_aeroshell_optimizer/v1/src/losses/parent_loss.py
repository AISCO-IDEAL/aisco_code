"""
Use this class as a Parent-class for all losses.
Do not overwrite the following methods:
- __init__
- get_loss
Overwrite the following methods:
- __setup()
- __loss()
"""

import torch


class ParentLoss:
    def __init__(self, device, weight: list[float]=[0.0, 0.0])->None:
        self.device = device
        self.__weight = weight
        if self.weight[0] == 0.0:
            return
        else:
            self.__setup()
            return
    
    def __setup(self)->None:
        return
    
    def get_loss(self, model)->torch.Tensor:
        if self.__weight[0] == 0.0:
            return torch.tensor(0.0, device=self.device)
        else:
            return self.__loss(model)*self.__weight[1]
    
    def __loss(self, model)->torch.Tensor:
        return torch.tensor(0.0, device=self.device)
