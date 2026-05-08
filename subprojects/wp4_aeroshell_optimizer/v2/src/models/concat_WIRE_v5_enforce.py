from typing import Callable, cast  # noqa: UP035

import torch
from torch import nn
from torch.profiler import record_function

from subprojects.wp2_assembler.version4.utils.wp4_utils import get_outer_bb_info
from subprojects.wp4_aeroshell_optimizer.v2.src.models.concat_baseclass import ConcatBaseModel
from subprojects.wp4_aeroshell_optimizer.v2.src.models.enforce_projector import AdaNPProjector
from subprojects.wp4_aeroshell_optimizer.v2.src.models.WIRE_v5 import GaborLayerV5, _log_tile


class ConcatWIREV5_ENFORCE(ConcatBaseModel):  # noqa: N801 — paper-aligned naming, mirrors ConcatWIREV5
    """
    Standalone copy of ConcatWIREV5 with an `AdaNPProjector` (ENFORCE) attached at the output.

    Diversity model: per-instance latent vector z is concatenated to the input coordinates,
    base network is identical to ConcatWIREV5 (GaborLayerV5 stack, log omega/scale, coarse-
    to-fine tiling, geometry-aware normalisation, xyz skip). After the base SDF is computed,
    AdaNPProjector applies a Newton-step projection toward the active constraint context.
    When `enforce_layer.is_active=False` or no context is set, the projector is a no-op.

    Activation is driven externally — `main_v2.py` reads
    `general_info.enforce_projection.activation_epoch_ratio` from the losses config and
    flips `enforce_layer.is_active` at the corresponding epoch.
    """

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
        use_spectral_norm: bool = False,
        bbox_center: list[float] | None = None,
        bbox_half_extent: float = 1000.0,
        use_assembly_normalisation: bool = False,
        use_smooth_output: bool = False,
        dtype: str = "float32",
    ):
        super().__init__(config=config)
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
        self._active_z = torch.zeros(z_dim, dtype=torch_dtype)

        omega_bands = _log_tile(hidden_omega_min, hidden_omega_max, hidden_layers)
        scale_bands = _log_tile(hidden_scale_min, hidden_scale_max, hidden_layers)

        self.gabor_layers = nn.ModuleList()
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
        for j in range(hidden_layers):
            layer_in = hidden_features + coords if j == skip_layer_idx else hidden_features
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

        self.smooth_proj = nn.Linear(hidden_features, hidden_features, dtype=torch_dtype) if use_smooth_output else None
        self.output_layer = nn.Linear(hidden_features, out_features, dtype=torch_dtype)

        enforce_cfg = config.get("general_info", {}).get("enforce_projection", {})
        self.enforce_layer = AdaNPProjector(
            max_iters=enforce_cfg.get("max_iters", 5),
            tolerance=enforce_cfg.get("tolerance", 1e-3),
        )
        self.current_enforce_context: str | None = None

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
            "enforce_max_iters": enforce_cfg.get("max_iters", 5),
            "enforce_tolerance": enforce_cfg.get("tolerance", 1e-3),
        }

    @property
    def bbox_center_tensor(self) -> torch.Tensor:
        return cast(torch.Tensor, self.bbox_center)

    @property
    def bbox_half_extent_tensor(self) -> torch.Tensor:
        return cast(torch.Tensor, self.bbox_half_extent)

    def get_model_info_dict(self) -> dict:
        return self.model_info_dict

    def register_enforce_constraint(self, context: str, fn: Callable) -> None:
        self.enforce_layer.register_constraint(context, fn)

    @record_function("ConcatWIREV5_ENFORCE.forward")
    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        if z is None:
            z = self._active_z
        if z is None:
            raise ValueError("No latent vector provided or set on ConcatWIREV5_ENFORCE model.")

        if z.dim() == 1:
            z = z.unsqueeze(0).expand(x.shape[0], -1)

        x_norm = (x - self.bbox_center_tensor) / self.bbox_half_extent_tensor
        h = torch.cat([x_norm, z], dim=-1)

        for i, layer in enumerate(self.gabor_layers):
            if i == self.skip_layer_idx + 1:
                h = torch.cat([h, x_norm], dim=-1)
            h = layer(h)

        if self.smooth_proj is not None:
            h = torch.nn.functional.silu(self.smooth_proj(h))

        sdf_raw = self.output_layer(h) * self.bbox_half_extent_tensor
        return self.enforce_layer(x, sdf_raw, self.current_enforce_context)
