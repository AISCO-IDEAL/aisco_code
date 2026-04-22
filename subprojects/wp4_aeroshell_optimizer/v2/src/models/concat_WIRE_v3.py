import torch
from torch import nn

from subprojects.wp4_aeroshell_optimizer.v2.src.models.concat_baseclass import ConcatBaseModel
from subprojects.wp4_aeroshell_optimizer.v2.src.models.WIRE_v3 import RealGaborLayer


class ConcatWIREV3(ConcatBaseModel):
    def __init__(
        self,
        z_dim: int,
        hidden_features: int,
        hidden_layers: int,
        coords: int = 3,
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
        Diversity model: V3 (log-spaced Gabor filterbank) + latent vector input + xyz skip connection.
        A latent vector z is concatenated to the input coordinates, letting the network represent a
        family of shapes — optimisation searches over z rather than all weights.
        The xyz skip connection re-injects only the spatial coordinates (not z) at mid-depth,
        re-anchoring the network to position without re-introducing the shape code.

        Args:
            z_dim: Dimension of the per-instance latent vector concatenated to the input.
            coords: Number of spatial coordinate dimensions (default 3 for x, y, z).
            skip_layer_idx: Which Gabor layer to inject after (0-indexed). Valid range [0, hidden_layers-1].
        """
        super().__init__(config=config)
        assert hidden_layers >= 1, "There must be at least one hidden layer."
        num_gabor_layers = hidden_layers + 1
        assert 0 <= skip_layer_idx < num_gabor_layers - 1, (
            f"skip_layer_idx must be in [0, {num_gabor_layers - 2}] for hidden_layers={hidden_layers}"
        )

        try:
            torch_dtype = getattr(torch, dtype)
        except AttributeError:
            raise ValueError(f"Invalid dtype given: {dtype}") from None
        if not isinstance(torch_dtype, torch.dtype):
            raise TypeError(f"Invalid dtype given: {dtype}")

        self.skip_layer_idx = skip_layer_idx
        self.coords = coords
        self._active_z = torch.zeros(z_dim)

        self.gabor_layers = nn.ModuleList()
        for i in range(num_gabor_layers):
            if i == 0:
                layer_in = coords + z_dim  # xyz + latent vector
                omega_min, omega_max = first_omega_min, first_omega_max
                scale_min, scale_max = first_scale_min, first_scale_max
            elif i == skip_layer_idx + 1:
                layer_in = hidden_features + coords  # skip reinjects xyz only (not z)
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
            "coords": coords,
            "z_dim": z_dim,
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

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """
        If z is not provided, uses the latent vector set via set_latent_vector().
        The xyz skip reinjects only the spatial coordinates, not z — z is already
        baked in at the first layer and the skip is purely for spatial re-anchoring.
        """
        if z is None:
            z = self._active_z
        if z is None:
            raise ValueError("No latent vector provided or set on ConcatWIREV3 model.")

        if z.dim() == 1:
            z = z.unsqueeze(0).expand(x.shape[0], -1)

        x_norm = x / 1000
        h = torch.cat([x_norm, z], dim=-1)

        for i, layer in enumerate(self.gabor_layers):
            if i == self.skip_layer_idx + 1:
                h = torch.cat([h, x_norm], dim=-1)
            h = layer(h)

        return self.output_layer(h) * 1000

    def get_model_info_dict(self) -> dict:
        return self.model_info_dict
