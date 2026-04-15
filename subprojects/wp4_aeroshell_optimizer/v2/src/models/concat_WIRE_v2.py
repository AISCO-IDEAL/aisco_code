import torch
from torch import nn

from subprojects.wp4_aeroshell_optimizer.v2.src.models.concat_baseclass import ConcatBaseModel


class RealGaborLayer(nn.Module):
    """
    Implicit representation with Gabor nonlinearity

    Inputs;
        in_features: Input features
        out_features; Output features
        bias: if True, enable bias for the linear operation
        is_first: Legacy SIREN parameter
        omega_0: Legacy SIREN parameter
        omega: Frequency of Gabor sinusoid term
        scale: Scaling of Gabor Gaussian term
    """

    def __init__(self, in_features, out_features, bias=True, omega0=10.0, sigma0=10.0, train_omega=False, train_scale=False):
        # Inherit from torch.nn.Module
        super().__init__()

        self.omega_0 = omega0 if not train_omega else nn.Parameter(torch.tensor(omega0))
        self.scale_0 = sigma0 if not train_scale else nn.Parameter(torch.tensor(sigma0))

        self.in_features = in_features
        self.splitter = nn.Linear(in_features, 2 * out_features, bias=bias)

    def forward(self, input) -> torch.Tensor:
        # split the output of the linear layer into two parts: one for frequency and one for scale
        wave, decay = torch.chunk(self.splitter(input), 2, dim=-1)
        return torch.cos(self.omega_0 * wave) * torch.exp(-(self.scale_0 * decay**2))


class ConcatWIREV2(ConcatBaseModel):
    def __init__(
        self,
        z_dim: int,
        hidden_features: int,
        hidden_layers: int,
        coords: int = 3,  # default to 3D coordinates (x, y, z) unless specified otherwise
        config: dict | None = None,
        out_features: int = 1,
        first_omega_0: float = 5.0,
        first_scale: float = 10.0,
        hidden_omega_0: float = 5.0,
        hidden_scale: float = 10.0,
        train_omega: bool = False,
        train_scale: bool = False,
        dtype: str = "float32",
    ):
        """
        Modifies the WIREV2 architecture by concatenating a latent vector to the input coordinates

        Parameters:
        - z_dim: Dimension of the latent vector to concatenate to the input coordinates
        - hidden_features: Number of features in the hidden layers
        - hidden_layers: Number of hidden layers
        - coords: Number of coordinate dimensions (default is 3 for x, y, z)
        - config: Optional configuration dictionary (not used in this implementation but can be extended for future use)
        - out_features: Number of output features (default is 1 for scalar field)
        - first_omega_0: Initial frequency for the first Gabor layer
        - first_scale: Initial scale for the first Gabor layer
        - hidden_omega_0: Initial frequency for the hidden Gabor layers
        - hidden_scale: Initial scale for the hidden Gabor layers
        - train_omega: If True, make the omega parameters trainable
        - train_scale: If True, make the scale parameters trainable
        - dtype: Data type for the final output layer (default is "float32")

        """
        super().__init__(
            config=config
        )  # Skip base init since we're defining our own architecture here, creates the self.config variable
        assert hidden_layers >= 1, "There must be at least one hidden layer."

        try:
            dtype = getattr(torch, dtype)
        except AttributeError:
            raise ValueError(f"Invalid dtype given: {dtype}") from None

        # Build network layers
        layers: list[nn.Module] = []

        # First layer: x,y,z, + laten vector -> hidden dim
        layers.append(
            RealGaborLayer(
                coords + z_dim,
                hidden_features,
                bias=True,
                omega0=first_omega_0,
                sigma0=first_scale,
                train_omega=train_omega,
                train_scale=train_scale,
            )
        )

        # Hidden layers
        layers.extend(
            [
                RealGaborLayer(
                    hidden_features,
                    hidden_features,
                    bias=True,
                    omega0=hidden_omega_0,
                    sigma0=hidden_scale,
                    train_omega=train_omega,
                    train_scale=train_scale,
                )
                for _ in range(hidden_layers)
            ]
        )

        # Final layer: Hidden dim → output dim
        layers.append(nn.Linear(hidden_features, out_features, dtype=dtype))
        self.net = nn.Sequential(*layers)

        self.register_buffer(
            "_active_z", torch.zeros(z_dim)
        )  # Initialize active latent vector to zeros, will be set properly during training using set_latent_vector method

        # Store model info in a dictionary for easy access and logging
        self.model_info_dict = {
            "name": str(self.__class__.__name__),
            "num_coords": coords,
            "z_dim": z_dim,
            "hidden_features": hidden_features,
            "hidden_layers": hidden_layers,
            "out_features": out_features,
            "first_omega": first_omega_0,
            "first_scale": first_scale,
            "hidden_omega": hidden_omega_0,
            "hidden_scale": hidden_scale,
            "trainable_omega": train_omega,
            "trainable_scale": train_scale,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "dtype": str(dtype),
        }

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """
        If z is not provided in the forward pass, use the currently active latent vector set on the model via self.set_latent_vector,
        which is what is done during training. Keeps the option to pass z here for flexibility, but the main training loop should use the set_latent_vector method to set z for each forward pass.
        This behaviour keeps backward compatibility with all the losses/meshing utils which only pass x (the coords) to the model
        """
        if z is None:
            z = self._active_z
        if z is None:
            raise ValueError("No latent vector provided or set on ConcatWIREV2 model.")

        # Ensure z is broadcastable to x shape: [batch, z_dim]
        if z.dim() == 1:
            z = z.unsqueeze(0).expand(x.shape[0], -1)

        input = torch.cat([x / 1000, z], dim=-1)
        return self.net(input) * 1000

    def get_model_info_dict(self) -> dict:
        """
        Returns a dictionary containing the model's configuration and hyperparameters.
        """
        return self.model_info_dict
