# just the models class which calls other models here

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.profiler import record_function

import wandb
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.total_loss import TotalLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsModelContract
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import NullVisManager
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import build_optimizer
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
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.gradient_conflict_tracker import GradientConflictTracker
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.storage_utils import allround_save_data
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.visual_wrapper import Wrapper
from subprojects.wp5.unified_mesh_extraction.sdf2mesh import MeshSDFExtractor

if TYPE_CHECKING:
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import CentralVisManagerV2
    from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.visual_wrapper import VisualisationManager


class BaseModel(nn.Module):
    def __init__(
        self, config: dict | None = None, overwrite_base_init: bool = True, in_dim: int = 3, hidden: int = 64, layers: int = 3
    ):
        super().__init__()  # gets all the stuff from nn.Module

        self.config = config if config is not None else {}

        if overwrite_base_init:
            return  # Skip base init if overwritten in subclass, avoids creating redundant variables

        self.in_dim = in_dim
        self.hidden = hidden
        self.layers = layers

        self.visual_config: dict = {}

        # Define a simple MLP with Relu and Tanh activations
        self.model_layers = nn.ModuleList()
        self.model_layers.append(nn.Linear(in_dim, hidden))

        for i in range(layers - 1):
            self.model_layers.append(nn.Linear(hidden, hidden))
            # Alternate activations
            if i % 2:
                self.model_layers.append(nn.Tanh())
            else:
                self.model_layers.append(nn.ReLU())

        self.model_layers.append(nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 1000.0  # Scale input to meters
        for layer in self.model_layers:
            x = layer(x)
            output = x
        return output * 1000.0

    @record_function("ModelBaseclass.optimise")
    def optimise(  # noqa: PLR0912, PLR0915
        self,
        general_config: dict[str, Any],
        training_config: dict[str, Any],
        renderer_manager: VisualisationManager | None = None,
        renderer_wrapper: Wrapper | None = None,  # noqa: ARG002
        wandb_name: str | None = None,
        debug_manager: DebugManager | None = None,
        conflict_tracker: GradientConflictTracker | None = None,
        conflict_cadence: int = 0,
        vis_manager: CentralVisManagerV2 | NullVisManager = NullVisManager(),
        prof: torch.profiler.profile | None = None,
    ) -> None:
        if training_config is None:
            raise ValueError("Training config must be provided for optimisation.")

        self.train()

        dbg = debug_manager if debug_manager is not None else DebugManager(config={"GLOBAL_DEBUG": False})
        # Prepare a MeshSDFExtractor instance. Prefer the vis_manager-provided
        # extractor when available (CentralVisManagerV2 returns one); otherwise
        # fall back to a local MeshSDFExtractor. Declare the variable once to
        # avoid mypy redefinition errors.
        visual_mesh_extractor: MeshSDFExtractor
        if vis_manager is not None and hasattr(vis_manager, "make_vis_extractor"):
            extractor_candidate = vis_manager.make_vis_extractor(str(next(self.parameters()).device))
            visual_mesh_extractor = extractor_candidate if extractor_candidate is not None else MeshSDFExtractor()
        else:
            visual_mesh_extractor = MeshSDFExtractor()

        if wandb.run and "learning_rate " in wandb.config:
            learning_rate = wandb.config.learning_rate
            print(f"Sweep: Updated learning_rate to {learning_rate}")
        else:
            learning_rate = training_config["training_params"].get("learning_rate", 1e-3)

        optimizer = build_optimizer(self.parameters(), training_config["training_params"])
        num_epochs = training_config["training_params"].get("num_epochs", 1000)
        gradient_clip_norm: float | None = training_config["training_params"].get("gradient_clip_norm", None)

        # Get device from model parameters since nn.Module doesn't have a .device attribute
        device = str(next(self.parameters()).device)
        loss_function = TotalLoss(device, renderer_manager=renderer_manager, vis_manager=vis_manager)

        if loss_function._norm_manager.is_enabled():
            assert isinstance(self, SupportsModelContract), (
                "Model must implement SupportsModelContract to use the LossNormalisationManager."
            )
            loss_function._calibrate(self, optimizer)

        dbg.log_run_config(
            model=self,
            training_config=training_config,
            general_config=general_config,
            device=device,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            optimizer_type=type(optimizer).__name__,
            loss_config={
                "use_alm": getattr(loss_function, "_use_alm", False),
                "use_cse": getattr(loss_function, "_use_cse", False),
                "manual_weights": collect_loss_weights(loss_function),
            },
        )
        assert isinstance(self, SupportsModelContract)
        mode_cfg: dict = training_config.get("mode", {})
        diagnostic_mode = mode_cfg.get("training_mode", "diagnostic") == "diagnostic"
        print_timing: bool = mode_cfg.get("print_timing", False)
        enforce_cfg = loss_function.losses_config.get("general_info", {}).get("enforce_projection", {})
        enforce_activation_epoch = int(num_epochs * enforce_cfg.get("activation_epoch_ratio", 0.9))
        for epoch in range(num_epochs):
            if epoch == enforce_activation_epoch and hasattr(self, "enforce_layer") and not self.enforce_layer.is_active:
                print(f"Activating ENFORCE projection at epoch {epoch} / {num_epochs}")
                self.enforce_layer.is_active = True
            dbg.begin_iteration(epoch)
            # get basic visuals (such as mesh assembly)
            vis_verts, vis_faces = None, None
            if vis_manager.is_enabled("mesh_assembly_v4"):
                vis_verts, vis_faces = visual_mesh_extractor.extract_mesh(sdf_model=self)
                vis_manager.vis_mesh_assembly_v4(epoch=epoch, sdf_verts=vis_verts, sdf_faces=vis_faces)
            if vis_manager.is_enabled("mesh_shape"):
                vis_verts, vis_faces = (
                    visual_mesh_extractor.extract_mesh(sdf_model=self)
                    if vis_verts is None or vis_faces is None
                    else (vis_verts, vis_faces)
                )
                vis_manager.vis_mesh_shape(epoch=epoch, verts=vis_verts, faces=vis_faces)

            t0 = time.perf_counter()
            per_component_grad_norms: dict[str, float] = {}
            grad_log_dict: dict[str, float] = {}

            if isinstance(optimizer, torch.optim.LBFGS):
                # L-BFGS requires the forward + backward pass inside a closure.
                # With max_iter=1 and no line search it is called exactly once per step.
                _closure_captures: dict = {}

                def _lbfgs_closure() -> torch.Tensor:
                    # for the lbfgs optimizer, requires closure
                    optimizer.zero_grad()
                    loss_function.get_total_loss(model=self, epoch_name=epoch)  # noqa: B023
                    backpropable = {k: v for k, v in loss_function._last_weighted_losses.items() if v.requires_grad}
                    scalar = sum(backpropable.values()) if backpropable else torch.tensor(0.0, device=device)
                    scalar.backward()
                    _closure_captures["total_weighted_loss"] = (  # noqa: B023
                        sum(v.item() for v in backpropable.values()) if backpropable else 0.0
                    )
                    if wandb.run:
                        pending = getattr(loss_function, "_pending_wandb_log", {})
                        _closure_captures["pending"] = pending.copy()  # noqa: B023
                        loss_function._pending_wandb_log = {}
                    return scalar

                optimizer.step(_lbfgs_closure)
                total_weighted_loss = _closure_captures.get("total_weighted_loss", 0.0)
                if wandb.run:
                    wandb.log(_closure_captures.get("pending", {}), step=epoch)
            elif not diagnostic_mode:
                # Performance mode: single combined backward, no per-component grad tracking.
                # Gradient conflict tracking and per-loss wandb gradient logs are disabled.
                optimizer.zero_grad()
                loss_function.get_total_loss(model=self, epoch_name=epoch)
                backpropable = {k: v for k, v in loss_function._last_weighted_losses.items() if v.requires_grad}
                total_weighted_loss = sum(v.item() for v in backpropable.values())
                if backpropable:
                    sum(backpropable.values()).backward()
                if wandb.run:
                    pending = getattr(loss_function, "_pending_wandb_log", {})
                    wandb.log(pending, step=epoch)
                    loss_function._pending_wandb_log = {}
            else:
                # Diagnostic mode: per-component backward so we can track per-loss gradient norms
                # and cosine conflict angles. Each graph is freed immediately (retain_graph=False);
                # param.grad accumulates across calls — identical to backward() on the summed total.
                optimizer.zero_grad()
                loss_function.get_total_loss(model=self, epoch_name=epoch)
                trainable_params = [p for p in self.parameters() if p.requires_grad]
                prev_grad = [torch.zeros_like(p) for p in trainable_params]
                log_grad_mem = loss_function._gradient_memory_diagnostics
                device_str = str(next(self.parameters()).device)
                total_weighted_loss = 0.0
                capture_conflict = conflict_tracker is not None and conflict_cadence > 0 and epoch % conflict_cadence == 0
                per_component_grad_vectors: dict[str, np.ndarray] = {}
                for component_name, weighted_loss in loss_function._last_weighted_losses.items():
                    if not weighted_loss.requires_grad:
                        per_component_grad_norms[component_name] = 0.0
                        grad_log_dict[f"optimizer_gradients/{component_name}"] = 0.0
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
                        grad_log_dict[f"grad_memory/{component_name}_after_mb"] = mem_after
                        grad_log_dict[f"grad_memory/{component_name}_freed_mb"] = mem_before - mem_after
                    curr_grad = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in trainable_params]
                    delta_norm = sum((c - pv).pow(2).sum().item() for c, pv in zip(curr_grad, prev_grad, strict=False)) ** 0.5
                    per_component_grad_norms[component_name] = delta_norm
                    grad_log_dict[f"optimizer_gradients/{component_name}"] = delta_norm
                    if capture_conflict:
                        delta_grad = torch.cat(
                            [(c - pv).detach().cpu().flatten() for c, pv in zip(curr_grad, prev_grad, strict=False)]
                        )
                        per_component_grad_vectors[component_name] = delta_grad.numpy()
                    prev_grad = curr_grad

                # Scheduled-off losses aren't in _last_weighted_losses; log explicit 0 so
                # WandB doesn't interpolate from a stale value.
                zero_weighted_cats: set = getattr(loss_function, "_zero_weighted_categories", set())
                if zero_weighted_cats:
                    for loss_name, category in getattr(loss_function, "_loss_name_to_category", {}).items():
                        if category not in zero_weighted_cats:
                            continue
                        if getattr(loss_function, "_use_cse", False) and category in {
                            getattr(loss_function, "_cse_drag_category", None),
                            getattr(loss_function, "_cse_solar_category", None),
                        }:
                            continue
                        if loss_name in per_component_grad_norms:
                            continue
                        per_component_grad_norms[loss_name] = 0.0
                        grad_log_dict[f"optimizer_gradients/{loss_name}"] = 0.0

                if wandb.run:
                    pending = getattr(loss_function, "_pending_wandb_log", {})
                    wandb.log({**pending, **grad_log_dict}, step=epoch)
                    loss_function._pending_wandb_log = {}

                if capture_conflict and per_component_grad_vectors and conflict_tracker is not None:
                    conflict_tracker.save_from_grad_vectors(per_component_grad_vectors, epoch)

            if print_timing:
                mode_label = "Diagnostic" if diagnostic_mode else "Performance"
                n_losses = len(loss_function._last_weighted_losses)
                print(f"{mode_label} forward/backward of {n_losses} Losses: {time.perf_counter() - t0:.4f} seconds")

            grad_norm = sum(p.grad.detach().pow(2).sum().item() for p in self.parameters() if p.grad is not None) ** 0.5
            lr = optimizer.param_groups[0]["lr"]

            raw_metrics = {
                name: float(tensor.detach().item())
                for name, tensor in getattr(loss_function, "latest_raw_metrics", {}).items()
                if torch.is_tensor(tensor)
            }
            due_recalibration_names = loss_function._norm_manager.get_recalibration_requests(epoch, raw_metrics)
            if due_recalibration_names:
                print(f"[LossNormalisation] Recalibrating after epoch {epoch}: {sorted(due_recalibration_names)}")
                loss_function._calibrate(self, optimizer)
                loss_function._norm_manager.mark_recalibrated(due_recalibration_names, epoch)

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
                _run_locally = general_config["general_info"].get("run_locally", "False") == "True"
                allround_save_data(
                    model=self,
                    storage_path=(
                        general_config["storage_paths"].get("beast_storage_dir")
                        or general_config["storage_paths"].get("local_storage_dir", "./local_storage_dir")
                    )
                    if _run_locally
                    else general_config["storage_paths"].get(
                        "storage_dir", "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/bulk_storage"
                    ),
                    save_model=True,
                    save_mesh=True,
                    epoch=epoch + 1,
                    verbose=general_config["general_info"].get("verbose", False),
                    pretraining=False,
                    wandb_run=wandb_name,
                )

            if not isinstance(optimizer, torch.optim.LBFGS):
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_norm)
                optimizer.step()

            dbg.end_iteration(
                epoch,
                loss=total_weighted_loss,
                grad_norm=grad_norm,
                lr=lr,
            )
            if prof is not None:
                prof.step()

        dbg.stop()

        # Post-optimization step: runs if developability loss is active and post_opt_enabled=True.
        dev_entry = loss_function._constraint_losses.get("developability")
        if dev_entry is not None:
            _dev_name, dev_instance = dev_entry
            if dev_instance.post_opt_enabled:
                verts, faces = dev_instance.mesh_extractor.extract_mesh(sdf_model=self)
                run_name = wandb_name or (wandb.run.name if wandb.run else "run")
                _storage_paths = general_config.get("storage_paths", {})
                _run_locally = general_config.get("general_info", {}).get("run_locally", "False") == "True"
                if _run_locally:
                    storage_dir = _storage_paths.get("beast_storage_dir") or _storage_paths.get(
                        "local_storage_dir", "./local_storage_dir"
                    )
                else:
                    storage_dir = _storage_paths.get(
                        "storage_dir", "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/bulk_storage"
                    )
                dev_instance.post_optimize(
                    vertices=verts.detach().cpu().numpy().astype("float64"),
                    faces=faces.detach().cpu().numpy().astype("int32"),
                    run_name=run_name,
                    storage_dir=storage_dir,
                )

    def validation(self):
        super().eval()  # Set training attribute of inherited model to false, ensures things like e.g. dropout are disabled
        # Validation Logic TODO

    def get_model_info_dict(self) -> dict:
        """
        Should return a dictionary uniquely identifying the model architecture and number of parameters, for logging and comparison purposes. Example:


            model_info = {
                "name": self.__class__.__name__,
                "in_dim": self.in_dim,
                "hidden": self.hidden,
                "layers": self.layers,
                "total_parameters": sum(p.numel() for p in self.parameters()),
            }
            return model_info
        """
        raise NotImplementedError(
            "get_model_info_dict method not implemented yet. Please implement in the subclass inheriting, see "
        )

        return {}

    def debug_info(self):
        raise ValueError("No Debug Info implemented")

    def init_neural_renderer(self, *_, **kwargs):
        """Deprecated: The renderer is now managed by the shared Manager.

        This method is kept for backwards compatibility but no longer creates
        a per-model Visualiser. Use a shared `Manager` and call its `update()`
        during optimization instead.
        """
        if kwargs.get("verbose", False):
            print("init_neural_renderer is deprecated. Use Manager + update() calls instead.")
