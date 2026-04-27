import math
from typing import cast

import torch
from torch import nn

from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v2.src.models.WIRE_v5 import GaborLayerV5, _log_tile


class FourierFeatures(nn.Module):
    """
    Fixed multi-scale sinusoidal encoding of input coordinates.

    For each coordinate axis x_k (assumed to be in [-1, 1] after bbox normalisation),
    computes [sin(2^0 π x_k), cos(2^0 π x_k), ..., sin(2^(K-1) π x_k), cos(2^(K-1) π x_k)]
    and concatenates across all axes. Output dimension: 2 * num_bands * in_features.

    At band k, one full oscillation spans 2 / 2^k normalised units, so band 0 captures
    features at the scale of the whole bounding box, and band K-1 captures features
    2^(K-1) times finer.

    Why this helps: gradient descent is biased toward low-frequency solutions (spectral bias).
    By pre-computing a fixed multi-scale basis from the coordinates, we hand the Gabor layers
    a rich frequency palette to select from rather than requiring them to generate high
    frequencies from scratch against their own bias.
    """

    def __init__(self, in_features: int, num_bands: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        freqs = torch.pow(2.0, torch.arange(num_bands, dtype=dtype)) * math.pi
        self.register_buffer("freqs", freqs)
        self.encoded_features = 2 * num_bands * in_features

    @property
    def freqs_tensor(self) -> torch.Tensor:
        return cast(torch.Tensor, self.freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dimensions of x: [..., in_features]
        angles = x.unsqueeze(-1) * self.freqs_tensor  # [..., in_features, num_bands]
        return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)  # [..., 2*num_bands*in_features]


class WIREV6(BaseModel):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        config: dict | None = None,
        out_features: int = 1,
        num_fourier_bands: int = 6,
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
        bbox_center: list[float] | None = None,
        bbox_half_extent: float = 1000.0,
        dtype: str = "float32",
    ):
        """
        V5 + Fourier feature encoding of input coordinates before the first Gabor layer.

        The normalised coordinates are passed through FourierFeatures to produce a
        2*num_fourier_bands*in_features-dimensional input to the Gabor stack. The skip
        connection reinjection also uses the Fourier-encoded coordinates (not raw xyz),
        so positional information is re-anchored at the same resolution as the input to
        the first layer.

        The Fourier encoding has no learnable parameters — it is a fixed transformation.
        The Gabor layers learn which frequency bands to attend to and how to combine them.

        Args:
            num_fourier_bands: K in [sin(2^k π x), cos(2^k π x)] for k = 0..K-1.
                Band k resolves features at spatial scale 2/2^k (in normalised units).
                For K=6 and bbox_half_extent=1000mm, the finest band (k=5) resolves
                features at 2000/32 ≈ 62mm. Increase K to capture finer detail.
            skip_layer_idx: Hidden layer index (0-indexed) that receives the re-injected
                Fourier-encoded coordinates. Valid range [0, hidden_layers - 1].
            bbox_center: Geometry centroid in the same units as coords. Defaults to origin.
            bbox_half_extent: Scalar half-size of the bounding box in the same units as coords.
        """
        super().__init__(config=config, overwrite_base_init=True)
        assert hidden_layers >= 1, "There must be at least one hidden layer."
        assert 0 <= skip_layer_idx < hidden_layers, (
            f"skip_layer_idx must be in [0, {hidden_layers - 1}] for hidden_layers={hidden_layers}"
        )
        config = config if config is not None else {}

        try:
            torch_dtype = getattr(torch, dtype)
        except AttributeError:
            raise ValueError(f"Invalid dtype given: {dtype}") from None
        if not isinstance(torch_dtype, torch.dtype):
            raise TypeError(f"Invalid dtype given: {dtype}")

        center: torch.Tensor = (
            torch.zeros(in_features, dtype=torch_dtype) if bbox_center is None else torch.tensor(bbox_center, dtype=torch_dtype)
        )
        self.register_buffer("bbox_center", center)
        self.register_buffer("bbox_half_extent", torch.tensor(bbox_half_extent, dtype=torch_dtype))

        self.skip_layer_idx = skip_layer_idx
        self.in_features = in_features

        self.fourier_enc = FourierFeatures(in_features, num_fourier_bands, dtype=torch_dtype)
        enc_features = self.fourier_enc.encoded_features  # 2 * num_fourier_bands * in_features

        omega_bands = _log_tile(hidden_omega_min, hidden_omega_max, hidden_layers)
        scale_bands = _log_tile(hidden_scale_min, hidden_scale_max, hidden_layers)

        self.gabor_layers = nn.ModuleList()

        self.gabor_layers.append(
            GaborLayerV5(
                enc_features,
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

        for j in range(hidden_layers):
            layer_in = hidden_features + enc_features if j == skip_layer_idx else hidden_features
            self.gabor_layers.append(
                GaborLayerV5(
                    layer_in,
                    hidden_features,
                    bias=True,
                    omega_min=omega_bands[j][0],
                    omega_max=omega_bands[j][1],
                    scale_min=scale_bands[j][0],
                    scale_max=scale_bands[j][1],
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
            "num_fourier_bands": num_fourier_bands,
            "enc_features": enc_features,
            "first_omega_min": first_omega_min,
            "first_omega_max": first_omega_max,
            "first_scale_min": first_scale_min,
            "first_scale_max": first_scale_max,
            "hidden_omega_min": hidden_omega_min,
            "hidden_omega_max": hidden_omega_max,
            "hidden_scale_min": hidden_scale_min,
            "hidden_scale_max": hidden_scale_max,
            "hidden_omega_bands": [list(b) for b in omega_bands],
            "hidden_scale_bands": [list(b) for b in scale_bands],
            "skip_layer_idx": skip_layer_idx,
            "trainable_omega": train_omega,
            "trainable_scale": train_scale,
            "bbox_center": bbox_center if bbox_center is not None else [0.0] * in_features,
            "bbox_half_extent": bbox_half_extent,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "dtype": str(torch_dtype),
        }

        raise NotImplementedError(
            "This model version is not ready yet. It is not stable due to yet unknown reasons and WIP. Stay tuned for WIREV6!"
        )

    @property
    def bbox_center_tensor(self) -> torch.Tensor:
        return cast(torch.Tensor, self.bbox_center)

    @property
    def bbox_half_extent_tensor(self) -> torch.Tensor:
        return cast(torch.Tensor, self.bbox_half_extent)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = (coords - self.bbox_center_tensor) / self.bbox_half_extent_tensor
        x_enc = self.fourier_enc(x)
        h = x_enc
        for i, layer in enumerate(self.gabor_layers):
            if i == self.skip_layer_idx + 1:
                h = torch.cat([h, x_enc], dim=-1)
            h = layer(h)
        return self.output_layer(h) * self.bbox_half_extent_tensor

    def get_model_info_dict(self) -> dict:
        return self.model_info_dict
