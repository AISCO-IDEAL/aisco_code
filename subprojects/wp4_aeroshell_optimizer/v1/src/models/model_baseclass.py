# just the models class which calls other models here
import torch
import torch.nn.functional as F
from torch import nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()



    def forward(self, x):
        raise NotImplementedError("Forward method not implemented!")
    
    def train(self):
        return 