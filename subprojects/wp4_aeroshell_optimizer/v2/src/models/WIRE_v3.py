import math

import torch
from torch import nn

from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_baseclass import BaseModel


class RealGaborLayer(nn.Module):
    """
    Gabor activation layer with log-spaced per-neuron frequency and scale initialization.
    Neurons are spread log-uniformly from omega_min to omega_max (and scale_min to scale_max),
    giving the layer a built-in frequency filterbank. When min == max, all neurons share the same value.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        omega_min: float = 10.0,
        omega_max: float = 10.0,
        scale_min: float = 10.0,
        scale_max: float = 10.0,
        train_omega: bool = False,
        train_scale: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        omega_vals = torch.logspace(math.log10(omega_min), math.log10(omega_max), out_features, dtype=dtype)
        scale_vals = torch.logspace(math.log10(scale_min), math.log10(scale_max), out_features, dtype=dtype)

        if train_omega:
            self.omega_0 = nn.Parameter(omega_vals)
        else:
            self.register_buffer("omega_0", omega_vals)

        if train_scale:
            self.scale_0 = nn.Parameter(scale_vals)
        else:
            self.register_buffer("scale_0", scale_vals)

        self.in_features = in_features
        self.splitter = nn.Linear(in_features, 2 * out_features, bias=bias, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        wave, decay = torch.chunk(self.splitter(input), 2, dim=-1)
        return torch.cos(self.omega_0 * wave) * torch.exp(-(self.scale_0 * decay**2))


class WIREV3(BaseModel):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        config: dict | None = None,
        out_features: int = 1,
        first_omega_min: float = 5.0,
        first_omega_max: float = 5.0,
        first_scale_min: float = 10.0,
        first_scale_max: float = 10.0,
        hidden_omega_min: float = 5.0,
        hidden_omega_max: float = 5.0,
        hidden_scale_min: float = 10.0,
        hidden_scale_max: float = 10.0,
        train_omega: bool = False,
        train_scale: bool = False,
        dtype: str = "float32",
    ):
        """
        WIRE model with log-spaced per-neuron frequency (omega) and scale initialization.
        Setting omega_min == omega_max gives uniform initialization (same as original WIRE).
        For intuition of the Gabor activation, see: https://drive.google.com/file/d/160NFMgfVo15KoIf-N_tYzQaR5ZHf1gXg/view?usp=drive_link
        """
        super().__init__(config=config, overwrite_base_init=True)
        assert hidden_layers >= 1, "There must be at least one hidden layer."
        config = config if config is not None else {}

        try:
            torch_dtype = getattr(torch, dtype)
        except AttributeError:
            raise ValueError(f"Invalid dtype given: {dtype}") from None
        if not isinstance(torch_dtype, torch.dtype):
            raise TypeError(f"Invalid dtype given: {dtype}")

        layers: list[nn.Module] = []

        layers.append(
            RealGaborLayer(
                in_features,
                hidden_features,
                bias=True,
                omega_min=first_omega_min,
                omega_max=first_omega_max,
                scale_min=first_scale_min,
                scale_max=first_scale_max,
                train_omega=train_omega,
                train_scale=train_scale,
                dtype=torch_dtype,
            )
        )

        layers.extend(
            [
                RealGaborLayer(
                    hidden_features,
                    hidden_features,
                    bias=True,
                    omega_min=hidden_omega_min,
                    omega_max=hidden_omega_max,
                    scale_min=hidden_scale_min,
                    scale_max=hidden_scale_max,
                    train_omega=train_omega,
                    train_scale=train_scale,
                    dtype=torch_dtype,
                )
                for _ in range(hidden_layers)
            ]
        )

        layers.append(nn.Linear(hidden_features, out_features, dtype=torch_dtype))

        self.net = nn.Sequential(*layers)

        self.model_info_dict = {
            "name": str(self.__class__.__name__),
            "in_features": in_features,
            "hidden_features": hidden_features,
            "hidden_layers": hidden_layers,
            "out_features": out_features,
            "first_omega_min": first_omega_min,
            "first_omega_max": first_omega_max,
            "first_scale_min": first_scale_min,
            "first_scale_max": first_scale_max,
            "hidden_omega_min": hidden_omega_min,
            "hidden_omega_max": hidden_omega_max,
            "hidden_scale_min": hidden_scale_min,
            "hidden_scale_max": hidden_scale_max,
            "trainable_omega": train_omega,
            "trainable_scale": train_scale,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "dtype": str(torch_dtype),
        }

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords / 1000) * 1000

    def get_model_info_dict(self) -> dict:
        return self.model_info_dict
