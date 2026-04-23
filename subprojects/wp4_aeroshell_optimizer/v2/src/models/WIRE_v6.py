import math

import torch
from torch import nn

import wandb
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.total_loss import TotalLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v2.src.models.WIRE_v5 import GaborLayerV5, _log_tile
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.debug_manager import (
    DebugManager,
    collect_alm_state,
    collect_cuda_memory,
    collect_loss_components,
    collect_loss_diagnostics,
    collect_loss_weights,
    collect_optimizer_state,
    collect_param_stats,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.storage_utils import allround_save_data
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.visual_wrapper import VisualisationManager, Wrapper


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]
        angles = x.unsqueeze(-1) * self.freqs  # [..., in_features, num_bands]
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

        center = (
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

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = (coords - self.bbox_center) / self.bbox_half_extent
        x_enc = self.fourier_enc(x)
        h = x_enc
        for i, layer in enumerate(self.gabor_layers):
            if i == self.skip_layer_idx + 1:
                h = torch.cat([h, x_enc], dim=-1)
            h = layer(h)
        return self.output_layer(h) * self.bbox_half_extent

    def optimise(
        self,
        general_config,
        training_config,
        renderer_manager: VisualisationManager | None = None,
        renderer_wrapper: Wrapper | None = None,  # noqa: ARG002
        wandb_name: str | None = None,
        debug_manager: DebugManager | None = None,
    ) -> None:
        if training_config is None:
            raise ValueError("Training config must be provided for optimisation.")

        self.train()

        dbg = debug_manager if debug_manager is not None else DebugManager(config={"GLOBAL_DEBUG": False})

        if wandb.run and "learning_rate " in wandb.config:
            learning_rate = wandb.config.learning_rate
            print(f"Sweep: Updated learning_rate to {learning_rate}")
        else:
            learning_rate = training_config["training_params"].get("learning_rate", 1e-3)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        num_epochs = training_config["training_params"].get("num_epochs", 1000)

        device = str(next(self.parameters()).device)
        loss_function = TotalLoss(device, renderer_manager=renderer_manager)

        dbg.log_run_config(
            model=self,
            training_config=training_config,
            general_config=general_config,
            device=device,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            optimizer_type="Adam",
            loss_config={
                "use_alm": getattr(loss_function, "_use_alm", False),
                "use_cse": getattr(loss_function, "_use_cse", False),
                "manual_weights": collect_loss_weights(loss_function),
            },
        )

        for epoch in range(num_epochs):
            dbg.begin_iteration(epoch)

            optimizer.zero_grad()

            loss_function.get_total_loss(model=self, epoch_name=epoch)

            trainable_params = [p for p in self.parameters() if p.requires_grad]
            prev_grad = [torch.zeros_like(p) for p in trainable_params]
            log_grad_mem = loss_function._gradient_memory_diagnostics
            device_str = str(next(self.parameters()).device)
            total_weighted_loss = 0.0
            per_component_grad_norms: dict[str, float] = {}
            for component_name, weighted_loss in loss_function._last_weighted_losses.items():
                if not weighted_loss.requires_grad:
                    continue
                total_weighted_loss += weighted_loss.item()
                if log_grad_mem and "cuda" in device_str:
                    mem_before = torch.cuda.memory_allocated() / 1024**2
                weighted_loss.backward(retain_graph=False)
                if log_grad_mem and "cuda" in device_str:
                    mem_after = torch.cuda.memory_allocated() / 1024**2
                    print(
                        f"  [Epoch {epoch}] backward({component_name}): "
                        f"{mem_before:.1f} MB → {mem_after:.1f} MB "
                        f"(freed {mem_before - mem_after:.1f} MB)"
                    )
                    if wandb.run:
                        wandb.log({f"grad_memory/{component_name}_after_mb": mem_after}, step=epoch)
                        wandb.log({f"grad_memory/{component_name}_freed_mb": mem_before - mem_after}, step=epoch)
                curr_grad = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in trainable_params]
                delta_norm = sum((c - pv).pow(2).sum().item() for c, pv in zip(curr_grad, prev_grad, strict=False)) ** 0.5
                per_component_grad_norms[component_name] = delta_norm
                if wandb.run:
                    wandb.log({f"{component_name}_gradient": delta_norm}, step=epoch)
                prev_grad = curr_grad

            grad_norm = sum(p.grad.detach().pow(2).sum().item() for p in self.parameters() if p.grad is not None) ** 0.5
            lr = optimizer.param_groups[0]["lr"]

            _last_cs = getattr(loss_function, "_last_cruising_speed", None)
            dbg.log_values(
                epoch,
                loss=total_weighted_loss,
                loss_components=collect_loss_components(loss_function),
                per_component_grad_norms=per_component_grad_norms,
                grad_norm=grad_norm,
                lr=lr,
                param_stats=collect_param_stats(self) if dbg.print_raw_values and dbg.log_param_stats else None,
                optimizer_state=collect_optimizer_state(optimizer) if dbg.print_raw_values and dbg.log_optimizer_state else None,
                cuda_memory=collect_cuda_memory() if dbg.print_raw_values and dbg.log_cuda_memory else None,
                alm_state=collect_alm_state(loss_function) if dbg.print_raw_values and dbg.log_alm_state else None,
                loss_weights=collect_loss_weights(loss_function),
                loss_diagnostics=collect_loss_diagnostics(loss_function)
                if dbg.print_raw_values and dbg.log_loss_diagnostics
                else None,
                extra={
                    "cruising_speed": _last_cs.item() if _last_cs is not None else "N/A",
                },
            )

            dbg.check_warnings(
                epoch,
                loss=total_weighted_loss,
                grad_norm=grad_norm,
                loss_components=collect_loss_components(loss_function),
                loss_diagnostics=collect_loss_diagnostics(loss_function) if dbg.emit_warnings else None,
            )

            dbg.check_insights(
                epoch,
                loss=total_weighted_loss,
                lr=lr,
                grad_norm=grad_norm,
            )

            mesh_save_interval = training_config["training_params"].get("mesh_save_interval", 50)
            if (epoch + 1) % mesh_save_interval == 0:
                allround_save_data(
                    model=self,
                    storage_path=general_config["storage_paths"].get(
                        "storage_dir", "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/bulk_storage"
                    ),
                    save_model=True,
                    save_mesh=True,
                    epoch=epoch + 1,
                    verbose=general_config["general_info"].get("verbose", False),
                    pretraining=False,
                    wandb_run=wandb_name,
                    marching_cubes_usage=training_config["training_params"].get("marching_cubes_usage", "False"),
                )

            optimizer.step()

            dbg.end_iteration(
                epoch,
                loss=total_weighted_loss,
                grad_norm=grad_norm,
                lr=lr,
            )

        dbg.stop()

    def get_model_info_dict(self) -> dict:
        return self.model_info_dict
