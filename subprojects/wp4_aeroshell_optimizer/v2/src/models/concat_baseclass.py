import torch
from torch import nn


class ConcatBaseModel(nn.Module):
    def __init__(
        self,
        *_args,
        _coords: int = 3,  # Default to 3D coordinates (x, y, z) unless specified otherwise
        config: dict | None = None,
        **_kwargs,
    ):
        """
        Modifies the architecture by tracking a latent vector to concatenate to the input coordinates
        """
        super().__init__()
        self.config = config if config is not None else {}
        # Initialize active latent vector to an empty tensor, will be set properly during training using set_latent_vector method
        # Register buffer means that this tensor is affected by model.to(device) calls, but is not a trainable parameter. Can be accessed just like normal (self._active_z)
        self.register_buffer("_active_z", torch.zeros(0), persistent=False)

    def set_latent_vector(self, z: torch.Tensor) -> None:
        """
        Sets the active latent vector for subsequent forward passes.
        """
        self._active_z = z

    def get_latent_vector(self) -> torch.Tensor:
        """
        Returns the currently active latent vector.
        """
        return self._active_z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The forward method must be implemented in the subclass of ConcatBaseModel.")

    def get_model_info_dict(self) -> dict:
        """
        Returns a dictionary containing the model's configuration and hyperparameters.
        """
        raise NotImplementedError("The get_model_info_dict method must be implemented in the subclass of ConcatBaseModel.")
