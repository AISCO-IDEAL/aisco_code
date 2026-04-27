import torch
from torch import nn

from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v2.src.models.WIRE_v3 import RealGaborLayer


class WIREV4(BaseModel):
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
        skip_layer_idx: int = 2,
        train_omega: bool = False,
        train_scale: bool = False,
        dtype: str = "float32",
    ):
        """
        V3 (log-spaced Gabor filterbank) with an xyz skip connection.
        After the Gabor layer at skip_layer_idx, the normalised input coordinates are
        concatenated to the activations before being passed to the next layer. This
        re-anchors the network to the spatial position at mid-depth, preventing the
        network from 'forgetting' where it is — particularly useful for deep stacks.

        Args:
            skip_layer_idx: Which Gabor layer to inject after (0-indexed). The layer
                immediately following receives hidden_features + in_features as input.
                Valid range: [0, hidden_layers - 1]. Default is hidden_layers // 2.
        """
        super().__init__(config=config, overwrite_base_init=True)
        assert hidden_layers >= 1, "There must be at least one hidden layer."
        num_gabor_layers = hidden_layers + 1
        assert 0 <= skip_layer_idx < num_gabor_layers - 1, (
            f"skip_layer_idx must be in [0, {num_gabor_layers - 2}] for hidden_layers={hidden_layers}"
        )
        config = config if config is not None else {}

        try:
            torch_dtype = getattr(torch, dtype)
        except AttributeError:
            raise ValueError(f"Invalid dtype given: {dtype}") from None
        if not isinstance(torch_dtype, torch.dtype):
            raise TypeError(f"Invalid dtype given: {dtype}")

        self.skip_layer_idx = skip_layer_idx
        self.in_features = in_features

        self.gabor_layers = nn.ModuleList()
        for i in range(num_gabor_layers):
            if i == 0:
                layer_in = in_features
                omega_min, omega_max = first_omega_min, first_omega_max
                scale_min, scale_max = first_scale_min, first_scale_max
            elif i == skip_layer_idx + 1:
                layer_in = hidden_features + in_features  # skip reinjects normalised xyz
                omega_min, omega_max = hidden_omega_min, hidden_omega_max
                scale_min, scale_max = hidden_scale_min, hidden_scale_max
            else:
                layer_in = hidden_features
                omega_min, omega_max = hidden_omega_min, hidden_omega_max
                scale_min, scale_max = hidden_scale_min, hidden_scale_max

            self.gabor_layers.append(
                RealGaborLayer(
                    layer_in,
                    hidden_features,
                    bias=True,
                    omega_min=omega_min,
                    omega_max=omega_max,
                    scale_min=scale_min,
                    scale_max=scale_max,
                    train_omega=train_omega,
                    train_scale=train_scale,
                    dtype=torch_dtype,
                )
            )

        self.output_layer = nn.Linear(hidden_features, out_features, dtype=torch_dtype)

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
            "skip_layer_idx": skip_layer_idx,
            "trainable_omega": train_omega,
            "trainable_scale": train_scale,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "dtype": str(torch_dtype),
        }

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords / 1000
        h = x
        for i, layer in enumerate(self.gabor_layers):
            if i == self.skip_layer_idx + 1:
                h = torch.cat([h, x], dim=-1)
            h = layer(h)
        return self.output_layer(h) * 1000

    def get_model_info_dict(self) -> dict:
        return self.model_info_dict
