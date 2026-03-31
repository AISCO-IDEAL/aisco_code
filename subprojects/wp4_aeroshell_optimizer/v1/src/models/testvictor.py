from __future__ import annotations

from typing import TYPE_CHECKING  # this prevents from circular imports

import torch
from torch import nn

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
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visual_wrapper import Wrapper

if TYPE_CHECKING:
    from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visual_wrapper import VisualisationManager


class VictorModel(BaseModel):
    # For now just a copy of BasicMLP, to be modified later
    def __init__(self, config: dict | None = None, in_dim: int = 3, hidden: int = 64, layers: int = 3, out_dim: int = 1):
        if config is None:
            config = {}
        super().__init__()

        self.config = config
        self.in_dim = in_dim
        self.hidden = hidden
        self.layers = layers
        self.out_dim = out_dim

        self.model_layers = nn.ModuleList()
        self.model_layers.append(nn.Linear(in_dim, hidden))
        for _ in range(layers - 1):
            self.model_layers.append(nn.Linear(hidden, hidden))
            self.model_layers.append(nn.SiLU())

        self.model_layers.append(nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 1000.0  # Scale input to meters
        for layer in self.model_layers:
            x = layer(x)
        output = x.squeeze(-1)
        return output * 1000.0

    def optimise(
        self,
        general_config,  # noqa: ARG002
        training_config,
        renderer_manager: VisualisationManager | None = None,
        renderer_wrapper: Wrapper | None = None,  # noqa: ARG002
        wandb_name: str | None = None,  # noqa: ARG002
        debug_manager: DebugManager | None = None,
    ) -> None:
        if training_config is None:
            raise ValueError("Training config must be provided for optimisation.")

        self.train()  # Set model to training mode, i.e. things like dropout layers are active, turn off by using self.eval() or model.eval()

        # Initialise a no-op DebugManager if none was provided
        dbg = debug_manager if debug_manager is not None else DebugManager(config={"GLOBAL_DEBUG": False})

        learning_rate = training_config.get("learning_rate", 1e-3)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        num_epochs = training_config["training_params"].get("num_epochs", 1000)

        # IMPORTANT: the device the model is on at time of function call defines the device passed to the loss_class
        # Get device from model parameters since nn.Module doesn't have a .device attribute
        device_str = next(self.parameters()).device.type
        loss_function = TotalLoss(device_str, renderer_manager=renderer_manager)  # Initialise TotalLoss with the correct device
        learning_rate = training_config.get("learning_rate", 1e-3)

        # ── Debug: one-time run config dump ──────────────────────
        dbg.log_run_config(
            model=self,
            training_config=training_config,
            device=device_str,
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

            # Calculate loss, Tracking should also be done in TotalLoss
            loss = loss_function.get_total_loss(model=self, epoch_name=epoch)

            # Optimize w.r.t. total loss
            # Do this by calculating the gradients of the weights w.r.t. the loss
            loss.backward()

            # ── Debug: compute aggregate gradient norm ───────────────
            grad_norm = sum(p.grad.detach().pow(2).sum().item() for p in self.parameters() if p.grad is not None) ** 0.5

            loss_val = loss.item()
            lr = optimizer.param_groups[0]["lr"]

            # ── Debug: print raw values ──────────────────────────────
            _last_cs = getattr(loss_function, "_last_cruising_speed", None)
            dbg.log_values(
                epoch,
                loss=loss_val,
                loss_components=collect_loss_components(loss_function),
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

            # ── Debug: check warnings ────────────────────────────────
            dbg.check_warnings(
                epoch,
                loss=loss_val,
                grad_norm=grad_norm,
                loss_components=collect_loss_components(loss_function),
                loss_diagnostics=collect_loss_diagnostics(loss_function) if dbg.emit_warnings else None,
            )

            # ── Debug: generate insights ─────────────────────────────
            dbg.check_insights(
                epoch,
                loss=loss_val,
                lr=lr,
                grad_norm=grad_norm,
            )

            # The optimizer takes the weights of self.parameters(), i.e. the weights and updates them based on the gradients calculated in loss.backward()
            optimizer.step()

            # ── Debug: end-of-iteration stats ───────────────────────
            dbg.end_iteration(
                epoch,
                loss=loss_val,
                grad_norm=grad_norm,
                lr=lr,
            )

        # ── Debug: cleanup ───────────────────────────────────────────
        dbg.stop()

    def get_model_info_dict(self):
        # Return model info as a dictionary
        model_info = {
            "name": "VictorModel",
            "in_dim": self.in_dim,
            "hidden": self.hidden,
            "layers": self.layers,
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }
        return model_info

    def debug_info(self):
        # Print model architecture and number of parameters, or anthing you might want to debug
        print("===================================")
        print("Model Debug Info:")

        print(self)

        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters())}")
        print("Model Layers:")
        for i, layer in enumerate(self.model_layers):
            print(f" Layer {i}: {layer}")

        print("Device/Gradient Info:")
        device = next(self.parameters()).device
        print(f" Model is on device: {device}")
        print(f"Do Model Parameters require Gradients? (Y/N): {any(p.requires_grad for p in self.parameters())}")

        print("End of Model Debug Info")
        print("===================================")
