# Basic MLP Model

import torch

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


class BasicMLP(BaseModel):
    def __init__(
        self, config: dict | None = None, overwrite_base_init: bool = False, in_dim: int = 3, hidden: int = 128, layers: int = 6
    ):
        if config is None:
            config = {}
        super().__init__(config, overwrite_base_init, in_dim, hidden, layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

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

        # Check for WandB sweep overrides for learning rate
        if wandb.run and "learning_rate " in wandb.config:
            learning_rate = wandb.config.learning_rate
            print(f"Sweep: Updated learning_rate to {learning_rate}")
        else:
            learning_rate = training_config["training_params"].get("learning_rate", 1e-3)

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
                    marching_cubes_usage=training_config["training_params"].get("marching_cubes_usage", "False"),
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

    def get_model_info_dict(self):
        # Return model info as a dictionary
        model_info = {
            "Model_name": "BasicMLP",
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
