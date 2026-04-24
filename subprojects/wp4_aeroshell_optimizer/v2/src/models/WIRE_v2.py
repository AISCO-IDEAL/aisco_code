import torch
from torch import nn

from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_baseclass import BaseModel


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

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        omega0=10.0,
        sigma0=10.0,
        train_omega=False,
        train_scale=False,
        dtype=torch.float32,
    ):
        # Inherit from torch.nn.Module
        super().__init__()

        self.omega_0 = omega0 if not train_omega else nn.Parameter(torch.tensor(omega0, dtype=dtype))
        self.scale_0 = sigma0 if not train_scale else nn.Parameter(torch.tensor(sigma0, dtype=dtype))

        self.in_features = in_features
        self.splitter = nn.Linear(in_features, 2 * out_features, bias=bias)

    def forward(self, input) -> torch.Tensor:
        # split the output of the linear layer into two parts: one for frequency and one for scale
        wave, decay = torch.chunk(self.splitter(input), 2, dim=-1)
        return torch.cos(self.omega_0 * wave) * torch.exp(-(self.scale_0 * decay**2))


class WIREV2(BaseModel):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
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
        Our second attempt at the WIRE Model, this time with only the real part of the activations and a more compact architecture.
        We also add the option to make the frequency and scale parameters trainable, which was not present in the original WIRE implementation.
        For intuition of the WIRE activation function (Gabor), see the geogebra file on the drive: https://drive.google.com/file/d/160NFMgfVo15KoIf-N_tYzQaR5ZHf1gXg/view?usp=drive_link
        """
        super().__init__(config=config, overwrite_base_init=True)  # Skip base init since we're defining our own architecture here
        assert hidden_layers >= 1, "There must be at least one hidden layer."
        config = config if config is not None else {}

        try:
            dtype = getattr(torch, dtype)
        except AttributeError:
            raise ValueError(f"Invalid dtype given: {dtype}") from None

        # Build network layers
        layers: list[nn.Module] = []

        # First layer: input → hidden dim
        layers.append(
            RealGaborLayer(
                in_features,
                hidden_features,
                bias=True,
                omega0=first_omega_0,
                sigma0=first_scale,
                train_omega=train_omega,
                train_scale=train_scale,
                dtype=dtype,
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
                    dtype=dtype,
                )
                for _ in range(hidden_layers)
            ]
        )

        # Final layer: Hidden dim → output dim
        layers.append(nn.Linear(hidden_features, out_features, dtype=dtype))

        self.net = nn.Sequential(*layers)

        # Store model info in a dictionary for easy access and logging
        self.model_info_dict = {
            "name": str(self.__class__.__name__),
            "in_features": in_features,
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

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords / 1000) * 1000

    def get_model_info_dict(self) -> dict:
        """
        Returns a dictionary containing the model's configuration and hyperparameters.
        """
        return self.model_info_dict
