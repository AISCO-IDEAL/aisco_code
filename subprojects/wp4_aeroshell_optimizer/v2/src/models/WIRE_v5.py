import math

import torch
from torch import nn

import wandb
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.total_loss import TotalLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_baseclass import BaseModel
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


def _log_tile(lo: float, hi: float, n: int) -> list[tuple[float, float]]:
    """Divides [lo, hi] into n equal sub-bands in log-space."""
    if lo == hi or n == 1:
        return [(lo, hi)] * n
    edges = [10 ** (math.log10(lo) + i * (math.log10(hi) - math.log10(lo)) / n) for i in range(n + 1)]
    return [(edges[i], edges[i + 1]) for i in range(n)]


class GaborLayerV5(nn.Module):
    """
    Gabor activation layer with two improvements over RealGaborLayer:

    1. Decoupled wave and decay projections: independent linear layers instead of a shared
       splitter. The old design forced both the oscillatory and envelope components through
       the same weight matrix, coupling directions that should be independent. Decoupling
       lets the network learn orthogonal subspaces for frequency vs. spatial localisation.
       Parameter count is identical to the old splitter.

    2. Log-parameterised omega and scale: stored and optimised in log-space so they stay
       strictly positive. The old approach (raw Parameter) allowed omega/scale to cross zero
       or go negative under gradient descent, causing silent sign flips in the activation.
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

        log_omega_vals = torch.logspace(math.log10(omega_min), math.log10(omega_max), out_features, dtype=dtype).log()
        log_scale_vals = torch.logspace(math.log10(scale_min), math.log10(scale_max), out_features, dtype=dtype).log()

        if train_omega:
            self.log_omega = nn.Parameter(log_omega_vals)
        else:
            self.register_buffer("log_omega", log_omega_vals)

        if train_scale:
            self.log_scale = nn.Parameter(log_scale_vals)
        else:
            self.register_buffer("log_scale", log_scale_vals)

        self.wave_proj = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        self.decay_proj = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        omega = torch.exp(torch.clamp(self.log_omega, min=0.0, max=5.0))
        scale = torch.exp(torch.clamp(self.log_scale, min=0.0, max=5.0))
        return torch.cos(omega * self.wave_proj(x)) * torch.exp(-(scale * self.decay_proj(x) ** 2))


class WIREV5(BaseModel):
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
        bbox_center: list[float] | None = None,
        bbox_half_extent: float = 1000.0,
        dtype: str = "float32",
    ):
        """
        V4 + three architectural changes:

        1. GaborLayerV5: decoupled wave/decay projections and log-parameterised omega/scale.

        2. Coarse-to-fine frequency schedule: hidden layers are tiled log-uniformly across
           [hidden_omega_min, hidden_omega_max] so each layer specialises in a distinct
           frequency band. Layer 0 covers the lowest band, the last hidden layer the highest.
           Scale is tiled in the same direction (larger scale = narrower Gaussian envelope,
           appropriate for higher-frequency neurons). Setting hidden_omega_min == hidden_omega_max
           recovers the original uniform behaviour.

        3. Geometry-aware normalisation: coordinates are shifted by bbox_center and divided
           by bbox_half_extent instead of the hard-coded /1000. With defaults (center=0,
           extent=1000) this is backward-compatible. Use a scalar bbox_half_extent — per-axis
           scaling is not supported because it breaks the SDF eikonal constraint (|∇f|=1 only
           holds when the coordinate scaling is isotropic).

        Args:
            skip_layer_idx: Hidden layer index (0-indexed) that receives the re-injected
                normalised coordinates. Valid range [0, hidden_layers - 1].
            bbox_center: Geometry centroid in the same units as coords. Defaults to origin.
            bbox_half_extent: Scalar half-size of the bounding box in the same units as coords.
                Coordinates map to approximately [-1, 1] when geometry fits inside the box.
                Also used to rescale the SDF output back to physical units.
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
        h = x
        for i, layer in enumerate(self.gabor_layers):
            if i == self.skip_layer_idx + 1:
                h = torch.cat([h, x], dim=-1)
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
