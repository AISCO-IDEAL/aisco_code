from typing import cast

import torch
from torch import nn
from torch.profiler import record_function

from subprojects.wp2_assembler.version4.utils.wp4_utils import get_outer_bb_info
from subprojects.wp4_aeroshell_optimizer.v2.src.models.concat_baseclass import ConcatBaseModel
from subprojects.wp4_aeroshell_optimizer.v2.src.models.WIRE_v5 import GaborLayerV5, _log_tile


class FiLMWIREV5(ConcatBaseModel):
    def __init__(
        self,
        z_dim: int,
        hidden_features: int,
        hidden_layers: int,
        coords: int = 3,
        config: dict | None = None,
        out_features: int = 1,
        film_hidden_features: int = 128,
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
        use_spectral_norm: bool = False,
        bbox_center: list[float] | None = None,
        bbox_half_extent: float = 1000.0,
        use_assembly_normalisation: bool = False,
        use_smooth_output: bool = False,
        dtype: str = "float32",
    ):
        """
        Deep-conditioning variant of ConcatWIREV5: input latent concat + xyz/z skip + FiLM.

        ConcatWIREV5 conditions on z only at the input layer. This model adds two deeper
        conditioning paths so z reaches every layer:
        - The mid-depth skip re-injects the latent z alongside the (normalised) coordinates
          (ConcatWIREV5 re-injects coordinates only).
        - A FiLM generator maps z to a per-layer (gamma, beta) pair; after every Gabor
          layer the activations are modulated as `h = gamma * h + beta`.

        The FiLM generator's final layer is zero-initialised and gamma is parameterised as
        `1 + gamma_raw`, so at initialisation FiLM is the identity and the model is
        numerically equivalent to ConcatWIREV5 — training then perturbs away from it.

        Args:
            z_dim: Dimension of the per-instance latent vector.
            film_hidden_features: Width of the FiLM generator's hidden layer.
            skip_layer_idx: Hidden layer index (0-indexed) receiving the re-injected
                normalised coordinates and latent. Valid range [0, hidden_layers - 1].
            bbox_center / bbox_half_extent: geometry-aware coordinate normalisation; see
                ConcatWIREV5. The remaining args mirror ConcatWIREV5 / WIREV5.
        """
        super().__init__(config=config)
        assert hidden_layers >= 1, "There must be at least one hidden layer."
        assert 0 <= skip_layer_idx < hidden_layers, (
            f"skip_layer_idx must be in [0, {hidden_layers - 1}] for hidden_layers={hidden_layers}"
        )

        try:
            torch_dtype = getattr(torch, dtype)
        except AttributeError:
            raise ValueError(f"Invalid dtype given: {dtype}") from None
        if not isinstance(torch_dtype, torch.dtype):
            raise TypeError(f"Invalid dtype given: {dtype}")

        if use_assembly_normalisation:
            outer_bb_info = get_outer_bb_info()
            bbox_center = outer_bb_info["centroid"]
            bbox_half_extent = max(outer_bb_info["width"], outer_bb_info["length"], outer_bb_info["height"]) / 2.0
            print(
                f"[{self.__class__.__name__}] Using automatic assembly normalisation with bbox_center={bbox_center} and bbox_half_extent={bbox_half_extent}"
            )

        center: torch.Tensor = (
            torch.zeros(coords, dtype=torch_dtype) if bbox_center is None else torch.tensor(bbox_center, dtype=torch_dtype)
        )
        self.register_buffer("bbox_center", center)
        self.register_buffer("bbox_half_extent", torch.tensor(bbox_half_extent, dtype=torch_dtype))

        self.skip_layer_idx = skip_layer_idx
        self.coords = coords
        self.z_dim = z_dim
        self.hidden_features = hidden_features
        self._active_z = torch.zeros(z_dim, dtype=torch_dtype)

        omega_bands = _log_tile(hidden_omega_min, hidden_omega_max, hidden_layers)
        scale_bands = _log_tile(hidden_scale_min, hidden_scale_max, hidden_layers)

        self.gabor_layers = nn.ModuleList()

        # First layer takes xyz + latent vector
        self.gabor_layers.append(
            GaborLayerV5(
                coords + z_dim,
                hidden_features,
                bias=True,
                omega_min=first_omega_min,
                omega_max=first_omega_max,
                scale_min=first_scale_min,
                scale_max=first_scale_max,
                train_omega=train_omega,
                train_scale=train_scale,
                use_spectral_norm=use_spectral_norm,
                dtype=torch_dtype,
            )
        )

        # Hidden layers; the one after skip_layer_idx receives the xyz + z reinjection
        for j in range(hidden_layers):
            layer_in = hidden_features + coords + z_dim if j == skip_layer_idx else hidden_features
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
                    use_spectral_norm=use_spectral_norm,
                    dtype=torch_dtype,
                )
            )

        # FiLM generator: z -> per-Gabor-layer (gamma, beta). Every Gabor layer is modulated.
        self.n_modulated_layers = len(self.gabor_layers)
        self.film_generator = nn.Sequential(
            nn.Linear(z_dim, film_hidden_features, dtype=torch_dtype),
            nn.SiLU(),
            nn.Linear(film_hidden_features, 2 * hidden_features * self.n_modulated_layers, dtype=torch_dtype),
        )
        # Identity init: zero the final layer so gamma = 1 + 0, beta = 0 at start.
        final_linear = cast(nn.Linear, self.film_generator[-1])
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

        self.smooth_proj = nn.Linear(hidden_features, hidden_features, dtype=torch_dtype) if use_smooth_output else None
        self.output_layer = nn.Linear(hidden_features, out_features, dtype=torch_dtype)

        self.model_info_dict = {
            "name": str(self.__class__.__name__),
            "coords": coords,
            "z_dim": z_dim,
            "hidden_features": hidden_features,
            "hidden_layers": hidden_layers,
            "film_hidden_features": film_hidden_features,
            "n_modulated_layers": self.n_modulated_layers,
            "out_features": out_features,
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
            "use_smooth_output": use_smooth_output,
            "use_spectral_norm": use_spectral_norm,
            "bbox_center": bbox_center if bbox_center is not None else [0.0] * coords,
            "bbox_half_extent": bbox_half_extent,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "dtype": str(torch_dtype),
        }

    @property
    def bbox_center_tensor(self) -> torch.Tensor:
        return cast(torch.Tensor, self.bbox_center)

    @property
    def bbox_half_extent_tensor(self) -> torch.Tensor:
        return cast(torch.Tensor, self.bbox_half_extent)

    @record_function("FiLMWIREV5.forward")
    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """
        If z is not provided, uses the latent vector set via set_latent_vector().
        FiLM (gamma, beta) are derived once per distinct latent and broadcast over points;
        the skip re-injects both the normalised coordinates and the latent z.
        """
        if z is None:
            z = self._active_z
        if z is None:
            raise ValueError("No latent vector provided or set on FiLMWIREV5 model.")

        # Per-latent FiLM parameters (one row per distinct latent; broadcasts over points).
        z_film = z if z.dim() == 2 else z.unsqueeze(0)
        film = self.film_generator(z_film)
        film = film.view(z_film.shape[0], self.n_modulated_layers, 2, self.hidden_features)
        gamma = 1.0 + film[:, :, 0, :]  # [Bz, n_layers, hidden_features]
        beta = film[:, :, 1, :]

        # Per-point latent for the input concat and the skip re-injection.
        z_cat = z.unsqueeze(0).expand(x.shape[0], -1) if z.dim() == 1 else z

        x_norm = (x - self.bbox_center_tensor) / self.bbox_half_extent_tensor
        h = torch.cat([x_norm, z_cat], dim=-1)

        for i, layer in enumerate(self.gabor_layers):
            if i == self.skip_layer_idx + 1:
                h = torch.cat([h, x_norm, z_cat], dim=-1)
            h = layer(h)
            h = gamma[:, i, :] * h + beta[:, i, :]

        if self.smooth_proj is not None:
            h = torch.nn.functional.silu(self.smooth_proj(h))

        return self.output_layer(h) * self.bbox_half_extent_tensor

    def get_model_info_dict(self) -> dict:
        return self.model_info_dict
