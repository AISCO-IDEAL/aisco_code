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


class RealGaborLayer(nn.Module):
    """
    Implicit representation with Gabor nonlinearity

    Inputs;
        in_features: Input features
        out_features; Output features
        bias: if True, enable bias for the linear operation
        is_first: Legacy SIREN parameter
        omega_0: Legacy SIREN parameter
        omega: Frequency of Gabor sinusoid term
        scale: Scaling of Gabor Gaussian term
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        omega0=10.0,
        sigma0=10.0,
        train_omega=False,
        train_scale=False,
        dtype=torch.float32,
    ):
        # Inherit from torch.nn.Module
        super().__init__()

        self.omega_0 = omega0 if not train_omega else nn.Parameter(torch.tensor(omega0, dtype=dtype))
        self.scale_0 = sigma0 if not train_scale else nn.Parameter(torch.tensor(sigma0, dtype=dtype))

        self.in_features = in_features
        self.splitter = nn.Linear(in_features, 2 * out_features, bias=bias)

    def forward(self, input) -> torch.Tensor:
        # split the output of the linear layer into two parts: one for frequency and one for scale
        wave, decay = torch.chunk(self.splitter(input), 2, dim=-1)
        return torch.cos(self.omega_0 * wave) * torch.exp(-(self.scale_0 * decay**2))


class WIREV2(BaseModel):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        config: dict | None = None,
        out_features: int = 1,
        first_omega_0: float = 5.0,
        first_scale: float = 10.0,
        hidden_omega_0: float = 5.0,
        hidden_scale: float = 10.0,
        train_omega: bool = False,
        train_scale: bool = False,
        dtype: str = "float32",
    ):
        """
        Our second attempt at the WIRE Model, this time with only the real part of the activations and a more compact architecture.
        We also add the option to make the frequency and scale parameters trainable, which was not present in the original WIRE implementation.
        For intuition of the WIRE activation function (Gabor), see the geogebra file on the drive: https://drive.google.com/file/d/160NFMgfVo15KoIf-N_tYzQaR5ZHf1gXg/view?usp=drive_link
        """
        super().__init__(config=config, overwrite_base_init=True)  # Skip base init since we're defining our own architecture here
        assert hidden_layers >= 1, "There must be at least one hidden layer."
        config = config if config is not None else {}

        try:
            dtype = getattr(torch, dtype)
        except AttributeError:
            raise ValueError(f"Invalid dtype given: {dtype}") from None

        # Build network layers
        layers: list[nn.Module] = []

        # First layer: input → hidden dim
        layers.append(
            RealGaborLayer(
                in_features,
                hidden_features,
                bias=True,
                omega0=first_omega_0,
                sigma0=first_scale,
                train_omega=train_omega,
                train_scale=train_scale,
                dtype=dtype,
            )
        )

        # Hidden layers
        layers.extend(
            [
                RealGaborLayer(
                    hidden_features,
                    hidden_features,
                    bias=True,
                    omega0=hidden_omega_0,
                    sigma0=hidden_scale,
                    train_omega=train_omega,
                    train_scale=train_scale,
                    dtype=dtype,
                )
                for _ in range(hidden_layers)
            ]
        )

        # Final layer: Hidden dim → output dim
        layers.append(nn.Linear(hidden_features, out_features, dtype=dtype))

        self.net = nn.Sequential(*layers)

        # Store model info in a dictionary for easy access and logging
        self.model_info_dict = {
            "name": str(self.__class__.__name__),
            "in_features": in_features,
            "hidden_features": hidden_features,
            "hidden_layers": hidden_layers,
            "out_features": out_features,
            "first_omega": first_omega_0,
            "first_scale": first_scale,
            "hidden_omega": hidden_omega_0,
            "hidden_scale": hidden_scale,
            "trainable_omega": train_omega,
            "trainable_scale": train_scale,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "dtype": str(dtype),
        }

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords / 1000) * 1000

    def optimise(  # noqa: PLR0912, PLR0915
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

        # ── Post-optimization (developability) ──────────────────────────
        dev_entry = loss_function._constraint_losses.get("developability")
        if dev_entry is not None:
            _dev_name, dev_instance = dev_entry
            if dev_instance.post_opt_enabled:
                verts, faces = dev_instance.mesh_extractor.extract_mesh(sdf_model=self)
                run_name = wandb_name or (wandb.run.name if wandb.run else "run")
                storage_dir = general_config["storage_paths"].get(
                    "storage_dir", "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/bulk_storage"
                )
                dev_instance.post_optimize(
                    vertices=verts.detach().cpu().numpy().astype("float64"),
                    faces=faces.detach().cpu().numpy().astype("int32"),
                    run_name=run_name,
                    storage_dir=storage_dir,
                )

    def get_model_info_dict(self) -> dict:
        """
        Returns a dictionary containing the model's configuration and hyperparameters.
        """
        return self.model_info_dict
