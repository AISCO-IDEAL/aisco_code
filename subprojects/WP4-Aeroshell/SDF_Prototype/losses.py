#Do the losses here



# @aalthaus
# You get: An SDF Network as an object or it's predictions in certain areas/points, depending on what you need
# My Opinion: choice is yours, but I think it makes computational sense to have the Network as input instead of raw predictions
# We need losses that work 

"""

Proposal: make a class that has losses as member functions and is able
to shift the weighting of different losses according to some type of input config (e.g. list of weights)
Goal: Should be modular if we implement different losses later on

"""


class SDFLosses:
    def __init__(self, weights):
        self.weights = weights

    def get_losses(self):
        #Compute different losses here
        total_loss = 0
        
        return total_loss