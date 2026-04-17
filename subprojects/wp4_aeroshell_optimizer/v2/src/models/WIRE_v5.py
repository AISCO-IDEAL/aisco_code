import math

import torch
from torch import nn

import wandb
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.total_loss import TotalLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v2.src.models.WIRE_v3 import RealGaborLayer
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


class WIREV5(BaseModel):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        config: dict | None = None,
        out_features: int = 1,
        num_fourier_features: int = 128,
        fourier_scale_min: float = 0.1,
        fourier_scale_max: float = 100.0,
        fourier_seed: int = 42,
        hidden_omega_0: float = 1.0,
        hidden_scale: float = 10.0,
        train_omega: bool = False,
        train_scale: bool = False,
        dtype: str = "float32",
    ):
        """
        WIRE model with a Fourier Feature Embedding (FFE) front-end, followed by a stack of RealGaborLayers.
        FFE maps raw coordinates to [sin(2π·B·x), cos(2π·B·x)] where B is a fixed matrix of random projections
        scaled log-uniformly across [fourier_scale_min, fourier_scale_max]. This explicitly covers a wide
        frequency range at the input, decoupling multi-scale coverage from the Gabor backbone's design.

        Args:
            num_fourier_features: Number of random frequency projections. Encoded input dim = 2 * num_fourier_features.
            fourier_scale_min: Minimum frequency scale for the random projection matrix B.
            fourier_scale_max: Maximum frequency scale for the random projection matrix B.
            fourier_seed: Random seed for reproducible B initialization.
            hidden_omega_0: Shared omega for all Gabor backbone layers.
            hidden_scale: Shared scale for all Gabor backbone layers.
        """
        super().__init__(config=config, overwrite_base_init=True)
        assert hidden_layers >= 1, "There must be at least one hidden layer."
        config = config if config is not None else {}

        try:
            dtype = getattr(torch, dtype)
        except AttributeError:
            raise ValueError(f"Invalid dtype given: {dtype}") from None

        # Build fixed Fourier projection matrix B: [in_features, num_fourier_features]
        # Each column is a random unit direction scaled by a log-uniformly sampled frequency.
        rng = torch.Generator()
        rng.manual_seed(fourier_seed)
        random_dirs = torch.randn(in_features, num_fourier_features, generator=rng, dtype=dtype)
        random_dirs = random_dirs / random_dirs.norm(dim=0, keepdim=True).clamp(min=1e-8)
        log_freqs = torch.linspace(math.log(fourier_scale_min), math.log(fourier_scale_max), num_fourier_features, dtype=dtype)
        freqs = torch.exp(log_freqs)
        self.register_buffer("B", random_dirs * freqs.unsqueeze(0))

        # Gabor backbone: input dim is 2 * num_fourier_features (sin + cos channels)
        backbone_in = 2 * num_fourier_features
        layers: list[nn.Module] = [
            RealGaborLayer(
                backbone_in,
                hidden_features,
                bias=True,
                omega_min=hidden_omega_0,
                omega_max=hidden_omega_0,
                scale_min=hidden_scale,
                scale_max=hidden_scale,
                train_omega=train_omega,
                train_scale=train_scale,
                dtype=dtype,
            )
        ]
        layers.extend(
            [
                RealGaborLayer(
                    hidden_features,
                    hidden_features,
                    bias=True,
                    omega_min=hidden_omega_0,
                    omega_max=hidden_omega_0,
                    scale_min=hidden_scale,
                    scale_max=hidden_scale,
                    train_omega=train_omega,
                    train_scale=train_scale,
                    dtype=dtype,
                )
                for _ in range(hidden_layers)
            ]
        )
        layers.append(nn.Linear(hidden_features, out_features, dtype=dtype))

        self.net = nn.Sequential(*layers)

        self.model_info_dict = {
            "name": str(self.__class__.__name__),
            "in_features": in_features,
            "num_fourier_features": num_fourier_features,
            "fourier_scale_min": fourier_scale_min,
            "fourier_scale_max": fourier_scale_max,
            "fourier_seed": fourier_seed,
            "hidden_features": hidden_features,
            "hidden_layers": hidden_layers,
            "out_features": out_features,
            "hidden_omega_0": hidden_omega_0,
            "hidden_scale": hidden_scale,
            "trainable_omega": train_omega,
            "trainable_scale": train_scale,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "dtype": str(dtype),
        }

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords / 1000
        x_proj = 2 * math.pi * (x @ self.B)
        x_enc = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.net(x_enc) * 1000

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
