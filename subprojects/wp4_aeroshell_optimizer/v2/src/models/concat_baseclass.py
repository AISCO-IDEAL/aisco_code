import torch
from torch import nn


class ConcatBaseModel(nn.Module):
    def __init__(
        self,
        *args,
        coords: int = 3, # Default to 3D coordinates (x, y, z) unless specified otherwise
        config: dict | None = None,
        **kwargs
    ):
        """
        Modifies the WIREV2 architecture by concatenating a latent vector to the input coordinates
        """
        super().__init__()  # gets all the stuff from nn.Module

        self.config = config if config is not None else {}


    def forward(self, *args) -> torch.Tensor:
        raise NotImplementedError("The forward method must be implemented in the subclass of ConcatBaseModel.")




    def get_model_info_dict(self) -> dict:
        """
        Returns a dictionary containing the model's configuration and hyperparameters.
        """
        raise NotImplementedError("The get_model_info_dict method must be implemented in the subclass of ConcatBaseModel.")
