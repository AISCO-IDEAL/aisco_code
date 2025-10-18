#Do the losses here



# @aalthaus
# You get: An SDF Network as an object OR ALTERNATIVELY it's predictions in certain areas/points, depending on what you need
# My Opinion: choice is yours, but I think it makes computational sense to have the Network as input instead of raw predictions
# We need losses that work 

import torch
from sdf import SDFNetwork
import typing as T

"""
Proposal: make a class that has losses as member functions and is able
to shift the weighting of different losses according to some type of input config (e.g. list of weights)
Goal: Should be modular if we implement different losses later on
"""


class SDFLosses:
    def __init__(self, weights: list[float] = [1.0,1.0]):
        self.weights = weights


    def eikonal_loss(self, model: SDFNetwork, points: torch.Tensor) -> torch.Tensor:
        #Compute Eikonal loss here
        placeholder = 0

        # Placeholder for Eikonal loss computation


        #Ensure compatibility
        eikonal_loss = torch.tensor(placeholder)  
        return eikonal_loss

    def smoothness_loss(self, model: SDFNetwork, points: torch.Tensor) -> torch.Tensor:
        #Compute Smoothness loss here
        placeholder = 0

        # Placeholder for Smoothness loss computation


        #Ensure compatibility
        smoothness_loss = torch.tensor(placeholder)  
        return smoothness_loss
    
    
    def get_total_loss(self, model: SDFNetwork) -> torch.Tensor:
        #Compute different losses here
        

        #Idea: sample here some points to compute losses on
        points = torch.tensor([0]) 



        # Placeholder for total loss computation
        placeholder = self.eikonal_loss(model, points ) + self.smoothness_loss(model, points)  # Example combination
        
        #Ensure compatibility
        total_loss = torch.tensor(placeholder)  
        return total_loss