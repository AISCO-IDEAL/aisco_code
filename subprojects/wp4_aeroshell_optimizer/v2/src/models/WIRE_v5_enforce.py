from typing import Callable, cast  # noqa: UP035

import torch
from torch import nn
from torch.profiler import record_function

from subprojects.wp2_assembler.version4.utils.wp4_utils import get_outer_bb_info
from subprojects.wp4_aeroshell_optimizer.v2.src.models.enforce_projector import AdaNPProjector
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v2.src.models.WIRE_v5 import GaborLayerV5, _log_tile


class WIREV5_ENFORCE(BaseModel):  # noqa: N801 — paper-aligned naming, mirrors WIREV5
    """
    Standalone copy of WIREV5 with an `AdaNPProjector` (ENFORCE) attached at the output.

    The base network is identical to WIREV5: GaborLayerV5 stack with decoupled wave/decay,
    log-parameterised omega/scale, coarse-to-fine frequency tiling, geometry-aware
    normalisation, and an xyz skip connection. After the base SDF is computed, the
    AdaNPProjector applies a Newton-step projection toward whichever constraint context is
    currently active (set via `enforce_layer_context`). When `enforce_layer.is_active=False`
    or no context is set, the projector is a no-op and the model returns sdf_raw.

    Activation of the projector is driven externally — `model_baseclass.optimise()` and
    `main_v2.py` both read `general_info.enforce_projection.activation_epoch_ratio` from
    the losses config and flip `enforce_layer.is_active` at the corresponding epoch.
    """

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
        use_spectral_norm: bool = False,
        bbox_center: list[float] | None = None,
        bbox_half_extent: float = 1000.0,
        use_assembly_normalisation: bool = False,
        use_smooth_output: bool = False,
        dtype: str = "float32",
    ):
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

        if use_assembly_normalisation:
            outer_bb_info = get_outer_bb_info()
            bbox_center = outer_bb_info["centroid"]
            bbox_half_extent = max(outer_bb_info["width"], outer_bb_info["length"], outer_bb_info["height"]) / 2.0
            print(
                f"[{self.__class__.__name__}] Using automatic assembly normalisation with bbox_center={bbox_center} and bbox_half_extent={bbox_half_extent}"
            )

        center: torch.Tensor = (
            torch.zeros(in_features, dtype=torch_dtype) if bbox_center is None else torch.tensor(bbox_center, dtype=torch_dtype)
        )
        self.register_buffer("bbox_center", center)
        self.register_buffer("bbox_half_extent", torch.tensor(bbox_half_extent, dtype=torch_dtype))

        self.skip_layer_idx = skip_layer_idx
        self.in_features = in_features

        omega_bands = _log_tile(hidden_omega_min, hidden_omega_max, hidden_layers)
        scale_bands = _log_tile(hidden_scale_min, hidden_scale_max, hidden_layers)

        self.gabor_layers = nn.ModuleList()
        self.gabor_layers.append(
            GaborLayerV5(
                in_features,
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
            layer_in = hidden_features + in_features if j == skip_layer_idx else hidden_features
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
            "hidden_omega_bands": [list(b) for b in omega_bands],
            "hidden_scale_bands": [list(b) for b in scale_bands],
            "skip_layer_idx": skip_layer_idx,
            "trainable_omega": train_omega,
            "trainable_scale": train_scale,
            "use_smooth_output": use_smooth_output,
            "use_spectral_norm": use_spectral_norm,
            "bbox_center": bbox_center if bbox_center is not None else [0.0] * in_features,
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

    @record_function("WIREV5_ENFORCE.forward")
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = (coords - self.bbox_center_tensor) / self.bbox_half_extent_tensor
        h = x
        for i, layer in enumerate(self.gabor_layers):
            if i == self.skip_layer_idx + 1:
                h = torch.cat([h, x], dim=-1)
            h = layer(h)
        if self.smooth_proj is not None:
            h = torch.nn.functional.silu(self.smooth_proj(h))
        sdf_raw = self.output_layer(h) * self.bbox_half_extent_tensor
        return self.enforce_layer(coords, sdf_raw, self.current_enforce_context)
