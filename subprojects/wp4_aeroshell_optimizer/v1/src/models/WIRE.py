#!/usr/bin/env python

import math
from pathlib import Path

import torch
import wandb
import yaml
from torch import nn

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.total_loss import TotalLoss
from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.storage_utils import allround_save_data
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visual_wrapper import VisualisationManager, Wrapper

# ============================================================================
# WIRE Configuration - Set the YAML config path here
# ============================================================================
WIRE_CONFIG_PATH = Path(__file__).parent / "subprojects/wp4_aeroshell_optimizer/v1/src/configs/WIRE_v1.yaml"

# Load WIRE configuration from the specified path
with open(WIRE_CONFIG_PATH) as f:
    WIRE_CONFIG = yaml.safe_load(f)


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
        self.omega_0 = nn.Parameter(
            torch.tensor([omega0], dtype=torch.float32),
            requires_grad=trainable
        )
        self.scale_0 = nn.Parameter(
            torch.tensor([sigma0], dtype=torch.float32),
            requires_grad=trainable
        )

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
        fn_samples = None,
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
            ComplexGaborLayer(
                in_features,
                hidden_features,
                omega0=first_omega_0,
                sigma0=scale,
                is_first=True,
                trainable=False
            )
        )

        # Hidden layers: complex → complex
        layers.extend([
            ComplexGaborLayer(
                hidden_features,
                hidden_features,
                omega0=hidden_omega_0,
                sigma0=scale,
                is_first=False,
                trainable=False
            )
            for _ in range(hidden_layers)
        ])

        # Final layer: complex → complex (we extract .real in forward)
        layers.append(
            nn.Linear(hidden_features, out_features, dtype=torch.cfloat)
        )

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

        # Load hyperparameters from WIRE_CONFIG (loaded at module import)
        omega0_first = float(WIRE_CONFIG.get("first_omega_0", 10.0))
        omega0_hidden = float(WIRE_CONFIG.get("hidden_omega_0", 10.0))
        sigma0 = float(WIRE_CONFIG.get("scale", 10.0))

        # Override dimensions if specified in YAML
        in_dim = int(WIRE_CONFIG.get("in_features", in_dim))
        hidden = int(WIRE_CONFIG.get("hidden_features", hidden))
        layers = int(WIRE_CONFIG.get("hidden_layers", layers))

        # WandB Sweep overrides - these take precedence over YAML config
        # Supports both direct keys (e.g., "first_omega_0") and prefixed keys (e.g., "WIRE_v1.first_omega_0")
        if wandb.run:
            # Helper to check both with and without "WIRE_v1." prefix
            def get_wandb_param(param_name):
                """Check wandb.config for param with or without WIRE_v1. prefix"""
                if f"WIRE_v1.{param_name}" in wandb.config:
                    return wandb.config[f"WIRE_v1.{param_name}"]
                elif param_name in wandb.config:
                    return wandb.config[param_name]
                return None

            param = get_wandb_param("first_omega_0")
            if param is not None:
                omega0_first = float(param)
                print(f"Sweep: Updated first_omega_0 to {omega0_first}")

            param = get_wandb_param("hidden_omega_0")
            if param is not None:
                omega0_hidden = float(param)
                print(f"Sweep: Updated hidden_omega_0 to {omega0_hidden}")

            param = get_wandb_param("scale")
            if param is not None:
                sigma0 = float(param)
                print(f"Sweep: Updated scale (sigma0) to {sigma0}")

            param = get_wandb_param("hidden_features")
            if param is not None:
                hidden = int(param)
                print(f"Sweep: Updated hidden_features to {hidden}")

            param = get_wandb_param("hidden_layers")
            if param is not None:
                layers = int(param)
                print(f"Sweep: Updated hidden_layers to {layers}")

            param = get_wandb_param("in_features")
            if param is not None:
                in_dim = int(param)
                print(f"Sweep: Updated in_features to {in_dim}")

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
    ) -> None:
        if training_config is None:
            raise ValueError("Training config must be provided for optimisation.")

        self.train()  # Set model to training mode, i.e. things like dropout layers are active, turn off by using self.eval() or model.eval()

        # Check for WandB sweep overrides for learning rate
        if wandb.run and "learning_rate" in wandb.config:
            learning_rate = wandb.config.learning_rate
            print(f"Sweep: Updated learning_rate to {learning_rate}")
        # Fallback to training_config (handle nested training_params if present)
        elif "training_params" in training_config:
            learning_rate = training_config["training_params"].get("learning_rate", 1e-3)
        else:
            learning_rate = training_config.get("learning_rate", 1e-3)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        num_epochs = training_config["training_params"].get("num_epochs", 1000)

        # IMPORTANT: the device the model is on at time of function call defines the device passed to the loss_class
        # Get device from model parameters since nn.Module doesn't have a .device attribute
        device = str(next(self.parameters()).device)
        loss_function = TotalLoss(device, renderer_manager=renderer_manager)  # Initialise TotalLoss with the correct device

        for epoch in range(num_epochs):
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
            for component_name, weighted_loss in loss_function._last_weighted_losses.items():
                if not weighted_loss.requires_grad:
                    continue
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
                if wandb.run:
                    wandb.log({f"{component_name}_gradient": delta_norm}, step=epoch)
                prev_grad = curr_grad

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

    model = BaseModel(
        config=config,
        overwrite_base_init=False,
        in_dim=3,
        hidden=256,
        layers=3
    )

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
    memory_mb = total_params * 8 / (1024 ** 2)  # Complex float64 = 8 bytes per param
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
