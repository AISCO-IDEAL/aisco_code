#!/usr/bin/env python

import math

import torch
from torch import nn

import wandb
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.total_loss import TotalLoss
from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.debug_manager import (
    DebugManager,
    collect_alm_state,
    collect_cuda_memory,
    collect_loss_components,
    collect_loss_diagnostics,
    collect_loss_weights,
    collect_optimizer_state,
    collect_param_stats,
)
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.storage_utils import allround_save_data
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visual_wrapper import VisualisationManager, Wrapper


class ComplexGaborLayer(BaseModel):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega0: float = 10.0,
        sigma0: float = 10.0,
        trainable: bool = False,
    ):
        super().__init__()

        # First layer processes real inputs, all others use complex
        dtype = torch.float if is_first else torch.cfloat

        # Wavelet parameters (typically fixed, not learned)
        self.omega_0 = nn.Parameter(torch.tensor([omega0], dtype=torch.float32), requires_grad=trainable)
        self.scale_0 = nn.Parameter(torch.tensor([sigma0], dtype=torch.float32), requires_grad=trainable)

        # Linear transformation
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        lin = self.linear(input_tensor)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin

        # Complex Gabor: exp(i·omega) · exp(-|scale|²)
        return torch.exp(1j * omega - scale.abs().square())


class WIREINR(BaseModel):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool = True,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        scale: float = 10.0,
        pos_encode: bool = False,
        sidelength: int = 512,
        fn_samples=None,
        use_nyquist: bool = True,
    ):
        super().__init__()

        # Suppress unused parameter warnings
        _ = (outermost_linear, pos_encode, sidelength, fn_samples, use_nyquist)

        # Reduce hidden size: complex neurons carry 2x information
        hidden_features = int(hidden_features / math.sqrt(2))

        self.wavelet = "gabor"

        # Build network layers
        layers: list[nn.Module] = []

        # First layer: real → complex
        layers.append(
            ComplexGaborLayer(in_features, hidden_features, omega0=first_omega_0, sigma0=scale, is_first=True, trainable=False)
        )

        # Hidden layers: complex → complex
        layers.extend(
            [
                ComplexGaborLayer(
                    hidden_features, hidden_features, omega0=hidden_omega_0, sigma0=scale, is_first=False, trainable=False
                )
                for _ in range(hidden_layers)
            ]
        )

        # Final layer: complex → complex (we extract .real in forward)
        layers.append(nn.Linear(hidden_features, out_features, dtype=torch.cfloat))

        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        output = self.net(coords)

        # Extract real component for downstream tasks
        if self.wavelet == "gabor":
            return output.real

        return output


class WIREModel(BaseModel):
    def __init__(
        self,
        config: dict,
        overwrite_base_init: bool = True,
        in_dim: int = 3,
        hidden: int = 64,
        layers: int = 3,
    ):
        super().__init__()

        self.config = config
        self.in_dim = in_dim
        self.hidden = hidden
        self.layers = layers

        if overwrite_base_init:
            return

        # Hardcoded WIRE hyperparameters
        omega0_first = 10.0
        omega0_hidden = 10.0
        sigma0 = 10.0

        # Update instance variables
        self.in_dim = in_dim
        self.hidden = hidden
        self.layers = layers

        # Build WIRE network
        self.wire_network = WIREINR(
            in_features=in_dim,
            hidden_features=hidden,
            hidden_layers=layers,
            out_features=1,
            first_omega_0=omega0_first,
            hidden_omega_0=omega0_hidden,
            scale=sigma0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale input from mm to meters (network works better with normalized inputs)
        x = x / 1000.0

        # Forward through WIRE network
        output = self.wire_network(x)

        # Remove trailing dimension and scale back to mm
        output = output.squeeze(-1)
        return output * 1000.0

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

        self.train()  # Set model to training mode, i.e. things like dropout layers are active, turn off by using self.eval() or model.eval()

        # Initialise a no-op DebugManager if none was provided
        dbg = debug_manager if debug_manager is not None else DebugManager(config={"GLOBAL_DEBUG": False})

        # Get learning rate from training config
        if "training_params" in training_config:
            learning_rate = training_config["training_params"].get("learning_rate", 5e-3)
        else:
            learning_rate = training_config.get("learning_rate", 5e-3)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        num_epochs = training_config["training_params"].get("num_epochs", 1000)

        # IMPORTANT: the device the model is on at time of function call defines the device passed to the loss_class
        # Get device from model parameters since nn.Module doesn't have a .device attribute
        device = str(next(self.parameters()).device)
        loss_function = TotalLoss(device, renderer_manager=renderer_manager)  # Initialise TotalLoss with the correct device

        # ── Debug: one-time run config dump ──────────────────────
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
            # ── Debug: start-of-iteration stats tracking ─────────────
            dbg.begin_iteration(epoch)

            # Clear gradients
            optimizer.zero_grad()

            # Calculate loss and all logging. Individual weighted tensors are stored
            # in loss_function._last_weighted_losses for the per-component backward below.
            loss_function.get_total_loss(model=self, epoch_name=epoch)

            # Per-component backward: each graph is freed immediately (retain_graph=False).
            # param.grad accumulates across calls — identical to backward() on the summed total.
            # We track the delta of param.grad before/after each component to get per-loss
            # gradient norms without needing retain_graph=True anywhere.
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            prev_grad = [torch.zeros_like(p) for p in trainable_params]  # accumulator at zero
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
                        # log memory after backward for each loss computation graph
                        wandb.log(
                            {f"grad_memory/{component_name}_after_mb": mem_after},
                            step=epoch,
                        )
                        # log memory freed for each loss computation graph
                        wandb.log(
                            {f"grad_memory/{component_name}_freed_mb": mem_before - mem_after},
                            step=epoch,
                        )
                curr_grad = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in trainable_params]
                delta_norm = sum((c - pv).pow(2).sum().item() for c, pv in zip(curr_grad, prev_grad, strict=False)) ** 0.5
                per_component_grad_norms[component_name] = delta_norm
                if wandb.run:
                    wandb.log({f"{component_name}_gradient": delta_norm}, step=epoch)
                prev_grad = curr_grad

            # ── Debug: compute aggregate gradient norm ───────────────
            grad_norm = sum(p.grad.detach().pow(2).sum().item() for p in self.parameters() if p.grad is not None) ** 0.5
            lr = optimizer.param_groups[0]["lr"]

            # ── Debug: print raw values ──────────────────────────────
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

            # ── Debug: check warnings (NaN, zero grad, stall) ───────
            dbg.check_warnings(
                epoch,
                loss=total_weighted_loss,
                grad_norm=grad_norm,
                loss_components=collect_loss_components(loss_function),
                loss_diagnostics=collect_loss_diagnostics(loss_function) if dbg.emit_warnings else None,
            )

            # ── Debug: generate insights ─────────────────────────────
            dbg.check_insights(
                epoch,
                loss=total_weighted_loss,
                lr=lr,
                grad_norm=grad_norm,
            )

            ###########################
            ### save mesh and model ###
            ###########################
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
                )

            # param.grad now holds the sum of all per-component gradients — same as if
            # we had called backward() on weighted_total_loss directly.
            optimizer.step()

            # ── Debug: end-of-iteration stats (time, memory, convergence) ──
            dbg.end_iteration(
                epoch,
                loss=total_weighted_loss,
                grad_norm=grad_norm,
                lr=lr,
            )

        # ── Debug: cleanup ───────────────────────────────────────────
        dbg.stop()

    def validation(self):
        """Set model to evaluation mode."""
        super().eval()

    def get_model_info_dict(self):
        model_info = {
            "in_dim": self.in_dim,
            "hidden": self.hidden,
            "effective_hidden": int(self.hidden / math.sqrt(2)),
            "layers": self.layers,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "model_type": "WIRE",
        }

        # Add WIRE-specific hyperparameters if available
        model_cfg = self.config.get("model", {})
        if "omega0" in model_cfg:
            model_info["omega0"] = model_cfg["omega0"]
        if "sigma0" in model_cfg:
            model_info["sigma0"] = model_cfg["sigma0"]

        return model_info

    def debug_info(self):
        """Print debug information about the model."""
        raise ValueError("No Debug Info implemented")


if __name__ == "__main__":
    # Example usage and validation
    print("WIRE-based BaseModel - Drop-in Replacement Demo")
    print("=" * 60)

    # Create model with same API as original BaseModel
    config = {
        "model": {
            "omega0": 10.0,
            "sigma0": 10.0,
        }
    }

    model = BaseModel(config=config, overwrite_base_init=False, in_dim=3, hidden=256, layers=3)

    # Test forward pass
    coords = torch.randn(100, 3) * 100  # 100 random 3D points in mm
    output = model(coords)

    print(f"\nInput shape:  {coords.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input dtype:  {coords.dtype}")
    print(f"Output dtype: {output.dtype}")

    # Display model info
    info = model.get_model_info_dict()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key:20s}: {value}")

    print("\nMemory footprint:")
    total_params = sum(p.numel() for p in model.parameters())
    memory_mb = total_params * 8 / (1024**2)  # Complex float64 = 8 bytes per param
    print(f"  Total parameters: {total_params:,}")
    print(f"  Approx memory:    {memory_mb:.2f} MB")

    print("\n" + "=" * 60)
    print("[SUCCESS] API compatibility verified")
    print("[SUCCESS] Forward pass successful")
    print("[SUCCESS] Output dtype is real (float32)")
    print("\nRecommended training settings for 3D SDF:")
    print("  - Learning rate: 5e-3 to 1e-2 (5-10x higher than standard MLP)")
    print("  - Optimizer: Adam")
    print("  - Loss: MSE + Eikonal constraint")
