"""Diversity-aware population-based trainer for latent vector optimization.

Owns:
* The per-vector training step (compute -> weight -> backward in VRAM-friendly order)
* External ALM multiplier updates across the population
* All logging (wandb scalars + competition charts)
* Optional mesh rendering

Does NOT own loss definitions (TotalLossV2) or ALM state (ALMManager is passed in).
With ``diversity_manager=None`` and ``population_size=1`` it degrades into a plain
single-z trainer with no diversity phase.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.profiler import record_function

import wandb
from subprojects.wp2_assembler.version4.utils.reg_imp import get_outer_prism_bounds_mm, scale_outer_prism
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.alm import ALMManager
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.diversity_loss_manager import DiversityLossManager
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.loss_normalisation import LossNormalisationManager
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.total_loss_v2 import TotalLossV2
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import (
    NullVisManager,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import (
    CentralVisManagerV2,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import build_optimizer, get_optimizer_telemetry
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.gradient_conflict_tracker import GradientConflictTracker
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.visualizer import CADVisualizer
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.visual_wrapper import VisualisationManager
from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor, load_mesh_extraction_config


class DiversityTrainer:
    """Two-phase memory-efficient trainer for population-of-latents SDF optimization.

    Phase 1 — sequential per-vector expensive backward (CFD, raytracing, constraints).
              One expensive autograd graph in VRAM at a time; gradients accumulate.
    Phase 2 — cheap WIRE-only diversity backward (all N small graphs held together).
    """

    def __init__(  # noqa: PLR0912, PLR0915
        self,
        model: SupportsLatentConcat,
        total_loss: TotalLossV2,
        alm_manager: ALMManager | None,
        population_size: int,
        z_dim: int,
        device: torch.device | str = "cuda",
        model_lr: float = 1e-4,
        optimizer_config: dict | None = None,
        diversity_manager: DiversityLossManager | None = None,
        diversity_num_points: int = 2048,
        diversity_bbox_scale_factor: float = 1.0,
        diversity_norm_order: int = 2,
        diversity_neighbor_agg_fn: str = "min",
        force_z0_to_origin: bool = True,
        z_resample_cadence: int = 0,  # so the original pretrained spots remain
        initial_Z: Tensor | None = None,
        z_resample_start_epoch: int = 0,
        z_resample_stop_epoch: int = -1,
        gradient_clip_norm: float | None = None,
        renderer_manager: VisualisationManager | None = None,
        rendering_freq: int | None = None,
        use_neural_renderer: bool = False,
        use_wandb: bool = False,
        log_dir: Path | None = None,
        conflict_tracker: GradientConflictTracker | None = None,
        conflict_cadence: int = 0,
        diagnostic_mode: bool = True,
        print_timing: bool = False,
        vis_manager: CentralVisManagerV2 | NullVisManager = NullVisManager(),
        Z_pretrained: torch.Tensor | None = None,
        vis_cadence_optimisation: int = 0,
        vis_strategy: str = "dynamic",
        vis_all_losses: bool = False,
        z_resample_mode: str = "on_circle",
        aggregation_method: str = "mean",
        l2_backward_strategy: str = "two_pass",
    ) -> None:
        """
        Args:
            model: SDF network supporting set_latent_vector / get_latent_vector.
            total_loss: Pure-computation loss manager (TotalLossV2).
            alm_manager: External ALM state manager. ``None`` -> manual constraint weights.
            population_size: Number of latents (N). Set to 1 for single-model training.
            z_dim: Dimensionality of each latent vector.
            device: Compute device.
            model_lr: Learning rate — used only when optimizer_config is None (backward compat).
            optimizer_config: The training_params sub-dict from training_config.yaml. When provided,
                the optimizer type and lr are loaded from it via optimizers.json. Falls back to Adam
                with model_lr when None.
            gradient_clip_norm: If set, clips the total parameter gradient L2-norm to this
                value before each optimizer step.  Prevents gradient explosion when eikonal
                or other losses produce large step sizes on the first few epochs.  ``None``
                (default) disables clipping (legacy behaviour).            diversity_manager: Optional population-level diversity manager. Set to
                ``None`` to skip Phase 2 entirely.
            diversity_num_points / diversity_bbox_scale_factor / norm_order / neighbor_agg_fn: trainer-level
                sampling params passed to the diversity manager compute call.
            diversity_bbox_scale_factor: Multiplier applied to the outer_prism bounds before
                sampling.  ``1.0`` (default) uses the exact outer_prism volume.  Values
                ``> 1.0`` expand the sample region; ``< 1.0`` shrink it.
            force_z0_to_origin: pin Z[0] at the origin (reference shape in GINN).
            z_resample_cadence: resample Z every N epochs (0 disables resampling).
            initial_Z: Optional pretrained latent population of shape (N, z_dim).  When
                provided the random initialisation is replaced by this tensor.  A copy
                is made so the caller's tensor is not modified in-place.
            z_resample_start_epoch: First epoch at which resampling is allowed (Mode C
                warm-up guard).  0 (default) means resampling can start immediately
                (backward-compatible with existing behaviour).  Set to a positive value
                to hold the pretrained Z population for that many epochs before the
                normal z_resample_cadence logic kicks in.
        mesh_extraction_config: dict removed — extractor is constructed internally using
                the WP5 unified mesh extractor and scaled outer_prism bounds.
            renderer_manager + rendering_freq + use_neural_renderer: rendering wiring.
            use_wandb: if False, logging is silently skipped.
            log_dir: directory for transient STL/PNG renders when falling back to CAD vis.
        """
        self._device: torch.device = torch.device(device) if isinstance(device, str) else device
        self._model: SupportsLatentConcat = model.to(self._device)
        self._total_loss: TotalLossV2 = total_loss
        self._alm_manager: ALMManager | None = alm_manager

        self._population_size: int = population_size
        self._z_dim: int = z_dim

        self._conflict_tracker: GradientConflictTracker | None = conflict_tracker
        self._conflict_cadence: int = max(0, int(conflict_cadence))
        self._diagnostic_mode: bool = diagnostic_mode
        self._print_timing: bool = print_timing

        self._diversity_manager: DiversityLossManager | None = diversity_manager
        self._diversity_num_points: int = diversity_num_points
        _prism = scale_outer_prism(get_outer_prism_bounds_mm(), diversity_bbox_scale_factor)
        self._diversity_bbox_lo: list[float] = [_prism["x"][0], _prism["y"][0], _prism["z"][0]]
        self._diversity_bbox_hi: list[float] = [_prism["x"][1], _prism["y"][1], _prism["z"][1]]
        self._diversity_norm_order: int = diversity_norm_order
        self._diversity_neighbor_agg_fn: str = diversity_neighbor_agg_fn

        self._force_z0_to_origin: bool = force_z0_to_origin
        self._z_resample_cadence: int = max(0, int(z_resample_cadence))
        self._z_resample_start_epoch: int = max(0, int(z_resample_start_epoch))
        self._z_resample_stop_epoch: int = int(z_resample_stop_epoch)
        self._mode_c_transition_logged: bool = False

        self._mesh_extractor: MeshSDFExtractor
        _mesh_config = load_mesh_extraction_config()
        _mesh_config = _dc_replace(_mesh_config, device=str(self._device))
        self._mesh_extractor = MeshSDFExtractor(config=_mesh_config)
        self._mesh_extractor.set_grid(
            bbox=[
                [self._diversity_bbox_lo[0], self._diversity_bbox_hi[0]],
                [self._diversity_bbox_lo[1], self._diversity_bbox_hi[1]],
                [self._diversity_bbox_lo[2], self._diversity_bbox_hi[2]],
            ],
            resolution=_mesh_config.base_resolution,
        )
        # Separate vis extractor with a larger bbox (vis_extraction.bbox_scale_factor from
        # vis_config.yaml, default 1.6x) so the mesh is never clipped at the outer-prism
        # boundary when the shape reaches maximum length at high epochs.
        # _mesh_extractor keeps the tight diversity bbox for loss domain sampling.
        self._vis_mesh_extractor: MeshSDFExtractor
        self._renderer_manager: VisualisationManager | None = renderer_manager
        self._rendering_freq: int | None = rendering_freq
        self._use_neural_renderer: bool = use_neural_renderer

        self._use_wandb: bool = use_wandb
        self._log_dir: Path = log_dir or Path("subprojects/wp4_aeroshell_optimizer/v2/logs")
        self._vis_manager: CentralVisManagerV2 | NullVisManager = vis_manager
        if isinstance(self._vis_manager, CentralVisManagerV2):
            self._vis_mesh_extractor = self._vis_manager.make_vis_extractor(device=str(self._device))
        else:
            self._vis_mesh_extractor = self._mesh_extractor

        _valid_vis_strategies = ("static", "dynamic")
        if vis_strategy not in _valid_vis_strategies:
            raise ValueError(
                f"vis_strategy must be one of {_valid_vis_strategies}, got '{vis_strategy}'. "
                "Set vis_strategy in vis_config.yaml latent_optimisation section."
            )
        if vis_strategy == "static" and Z_pretrained is None:
            raise ValueError(
                "vis_strategy='static' requires Z_pretrained to be provided so mesh extraction "
                "uses frozen pretrained positions. Either pass Z_pretrained or set "
                "vis_strategy='dynamic' in vis_config.yaml latent_optimisation section."
            )
        self._vis_strategy: str = vis_strategy
        self._vis_all_losses: bool = vis_all_losses
        self._vis_cadence_optimisation: int = max(0, int(vis_cadence_optimisation))
        # Store a detached copy — not updated by the optimizer
        self._Z_pretrained: torch.Tensor | None = (
            Z_pretrained.detach().clone().to(self._device) if Z_pretrained is not None else None
        )
        _valid_resample_modes = ("on_circle", "in_circle", "random")
        if z_resample_mode not in _valid_resample_modes:
            raise ValueError(
                f"z_resample_mode must be one of {_valid_resample_modes}, got '{z_resample_mode}'. "
                "Set z_resample_mode in training_config.yaml."
            )
        self._z_resample_mode: str = z_resample_mode

        _valid_aggregation = ("mean", "l2_norm")
        if aggregation_method not in _valid_aggregation:
            raise ValueError(
                f"aggregation_method must be one of {_valid_aggregation}, got '{aggregation_method}'. "
                "Set training_params.multi_shape_aggregation.method in training_config.yaml."
            )
        _valid_l2_strategy = ("single_pass", "two_pass")
        if l2_backward_strategy not in _valid_l2_strategy:
            raise ValueError(
                f"l2_backward_strategy must be one of {_valid_l2_strategy}, got '{l2_backward_strategy}'. "
                "Set training_params.multi_shape_aggregation.l2_backward_strategy in training_config.yaml."
            )
        self._aggregation_method: str = aggregation_method
        self._l2_backward_strategy: str = l2_backward_strategy
        self._l2_grad_log_warned: bool = False

        generator_device = self._device.type if self._device.type in {"cpu", "cuda"} else "cpu"
        self._resample_generator = torch.Generator(device=generator_device)

        # Fixed population of latent vectors (not trained). Z[0] optionally pinned to origin.
        if initial_Z is not None:
            expected_shape = (population_size, z_dim)
            if tuple(initial_Z.shape) != expected_shape:
                raise ValueError(
                    f"initial_Z shape {tuple(initial_Z.shape)} does not match expected "
                    f"{expected_shape}. "
                    f"Set population_size={initial_Z.shape[0]} in training_config.yaml to match the "
                    f"pretrained population, or re-run pretraining with "
                    f"population_size={population_size}."
                )
            self._Z: Tensor = initial_Z.clone().to(self._device)
        else:
            self._Z = torch.randn(population_size, z_dim, device=self._device, dtype=torch.float32)

        if self._force_z0_to_origin and self._population_size > 0:
            self._Z[0] = torch.zeros(z_dim, device=self._device, dtype=torch.float32)

        if optimizer_config is not None:
            self._model_optimizer: Optimizer = build_optimizer(self._model.parameters(), optimizer_config)
        else:
            self._model_optimizer = torch.optim.Adam(self._model.parameters(), lr=model_lr)
        self._gradient_clip_norm: float | None = gradient_clip_norm if gradient_clip_norm and gradient_clip_norm > 0 else None

        # Rolling history for the stacked-area competition chart
        self._raw_loss_history: list[dict[str, float]] = []
        self._normalized_loss_history: list[dict[str, float]] = []
        self._weighted_loss_history: list[dict[str, float]] = []
        self._optimizer_loss_history: list[dict[str, float]] = []

        # Per-slot raw-loss line charts (one interactive line per slot, one chart per loss).
        # History is reset on each resample; epoch indices always store the real epoch number so
        # the x-axis is continuous across windows (e.g. window-1: [0..4], window-2: [5..9], …).
        self._per_slot_loss_history: dict[str, list[list[float]]] = {}  # loss_name -> [epoch_0_vals, epoch_1_vals, ...]
        self._per_slot_epoch_indices: list[int] = []
        self._per_slot_charts_active: bool = True

        norm_cfg = self._total_loss.losses_config.get("general_info", {}).get("loss_normalisation", {})
        self._norm_manager = LossNormalisationManager(norm_cfg)

    # ------------------------------------------------------------------
    # Latent resampling
    # ------------------------------------------------------------------
    def _resample_population(self, epoch: int) -> None:
        """Resample latents with a local generator to avoid global RNG side effects."""
        if self._population_size <= 0 or self._z_resample_cadence <= 0:
            return

        start_idx = 1 if self._force_z0_to_origin else 0
        if start_idx >= self._population_size:
            if self._force_z0_to_origin:
                self._Z[0] = torch.zeros(self._z_dim, device=self._device, dtype=torch.float32)
            return

        self._resample_generator.manual_seed(epoch // self._z_resample_cadence)
        n_resample = self._population_size - start_idx
        if self._z_resample_mode == "on_circle" and self._Z_pretrained is not None:
            # on_circle: restore each slot to its exact pretrained position on the circle
            self._Z[start_idx:] = self._Z_pretrained[start_idx:].detach().clone()
        elif self._z_resample_mode == "in_circle" and self._Z_pretrained is not None and self._z_dim >= 2:
            # in_circle: sample uniformly inside the pretrained circle disk (2D plane, higher dims = 0)
            # Radius from pretrained: use max 2D-norm across all slots as the disk boundary
            r_max = self._Z_pretrained[:, :2].norm(dim=1).max().item()
            U = torch.rand(n_resample, generator=self._resample_generator, device=self._device)
            theta = torch.rand(n_resample, generator=self._resample_generator, device=self._device) * (2.0 * math.pi)
            r = r_max * U.sqrt()  # sqrt for uniform area density
            new_z = torch.zeros(n_resample, self._z_dim, device=self._device, dtype=torch.float32)
            new_z[:, 0] = r * theta.cos()
            new_z[:, 1] = r * theta.sin()
            self._Z[start_idx:] = new_z
        else:
            # random (fallback when pretrained positions are unavailable)
            self._Z[start_idx:] = torch.randn(
                n_resample,
                self._z_dim,
                device=self._device,
                dtype=torch.float32,
                generator=self._resample_generator,
            )
        if self._force_z0_to_origin:
            self._Z[0] = torch.zeros(self._z_dim, device=self._device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Loss & gradient normalisation calibration
    # ------------------------------------------------------------------
    def _calibrate(self) -> None:  # noqa: PLR0912, PLR0915
        """Calibrate loss and gradient scale factors before training begins.

        Runs one full forward pass over all population members with calibration_mode=True
        (bypasses epoch scheduling so every loss is computed regardless of _disabled_epochs).
        Loss scales are derived from the mean absolute raw value across the population.
        Gradient scales are derived from the per-component gradient L2 norm on the first
        population member, measured after loss scales have already been applied.

        Null losses (requires_grad=False on the component tensor) are skipped — both scales
        remain 1.0 for them. Optimizer gradients are zeroed before and after to leave no trace.
        """
        print("[LossNormalisation] Starting calibration pass...")
        self._model_optimizer.zero_grad()

        loss_scales: dict[str, float] = {}
        grad_scales: dict[str, float] = {}
        calibrated_names: set[str] = set()

        active_names: set[str] = set(self._total_loss._all_loss_instances().keys())
        if getattr(self._total_loss, "use_cse", False):
            drag_name = self._total_loss._objective_by_category.get(self._total_loss._cse_drag_category, (None,))[0]
            solar_name = self._total_loss._objective_by_category.get(self._total_loss._cse_solar_category, (None,))[0]
            for merged_name in (drag_name, solar_name):
                if merged_name:
                    active_names.discard(merged_name)
            active_names.add("cse_objective")

        if active_names and all(self._norm_manager.has_both_fixed_scales(name) for name in active_names):
            for name in sorted(active_names):
                fixed_loss = self._norm_manager.get_fixed_loss_scale(name)
                fixed_grad = self._norm_manager.get_fixed_grad_scale(name)
                if fixed_loss is None or fixed_grad is None:
                    continue
                loss_scales[name] = fixed_loss
                grad_scales[name] = fixed_grad
                calibrated_names.add(name)
            self._norm_manager.record_calibration(loss_scales, grad_scales, calibrated_names)
            self._model_optimizer.zero_grad()
            print("[LossNormalisation] Calibration skipped (all active components use fixed loss+grad scales).")
            return

        # --- Loss scale calibration: forward-only pass over full population ---
        loss_value_sums: dict[str, float] = {}

        for i in range(self._population_size):
            self._model.set_latent_vector(self._Z[i])
            bundle = self._total_loss.compute(self._model, epoch_name=0, calibration_mode=True)

            for role_dict, role_bundle in (
                (self._total_loss._objective_by_category, bundle.objectives),
                (self._total_loss._constraint_by_category, bundle.constraints),
                (self._total_loss._aux_by_category, bundle.aux),
            ):
                for category, tensor in role_bundle.items():
                    loss_name = role_dict[category][0]
                    calibrated_names.add(loss_name)
                    loss_value_sums[loss_name] = loss_value_sums.get(loss_name, 0.0) + abs(tensor.detach().item())

        for name in calibrated_names:
            fixed_loss = self._norm_manager.get_fixed_loss_scale(name)
            if fixed_loss is not None:
                loss_scales[name] = fixed_loss
                print(f"  Loss scale for {name}: fixed={fixed_loss:.6g}")
                continue
            mean_val = loss_value_sums[name] / self._population_size
            loss_scales[name] = self._norm_manager._target_loss / mean_val if mean_val > 0 else 1.0
            print(f"  Loss scale for {name}: mean_raw={mean_val:.6g}, scale={loss_scales[name]:.6g}")

        # CSE: loss_scale for cse_objective is derived from the raw v_cruise value.
        # Individual drag/solar categories are excluded — they are consumed by CSE internally.
        if getattr(self._total_loss, "use_cse", False):
            print("  CSE mode detected: computing cse_objective scale...")
            fixed_cse_loss = self._norm_manager.get_fixed_loss_scale("cse_objective")
            if fixed_cse_loss is not None:
                loss_scales["cse_objective"] = fixed_cse_loss
                calibrated_names.add("cse_objective")
                print(f"  Loss scale for cse_objective: fixed={fixed_cse_loss:.6g}")
            else:
                self._model.set_latent_vector(self._Z[0])
                bundle = self._total_loss.compute(self._model, epoch_name=0, calibration_mode=True)
                # weight() without a norm_manager gives us the raw v_cruise-based cse_objective
                raw_result = self._total_loss.weight(
                    bundle, self._alm_manager, return_components=True, norm_manager=None, calibration_pass=True
                )
                if "cse_objective" in raw_result.components:
                    cse_val = abs(raw_result.components["cse_objective"].detach().item())
                    loss_scales["cse_objective"] = self._norm_manager._target_loss / cse_val if cse_val > 0 else 1.0
                    calibrated_names.add("cse_objective")
                    print(f"  Loss scale for cse_objective: raw={cse_val:.6g}, scale={loss_scales['cse_objective']:.6g}")
            # Remove drag/solar from calibrated set since they're merged into cse_objective
            drag_name = self._total_loss._objective_by_category.get(self._total_loss._cse_drag_category, (None,))[0]
            solar_name = self._total_loss._objective_by_category.get(self._total_loss._cse_solar_category, (None,))[0]
            for merged_name in (drag_name, solar_name):
                if merged_name:
                    print(f"  Excluding {merged_name} (merged into cse_objective)")
                    calibrated_names.discard(merged_name)
                    loss_scales.pop(merged_name, None)

        # Make the refreshed loss scales visible during gradient calibration.
        self._norm_manager.record_calibration(loss_scales, {}, calibrated_names)

        # Store loss scales so weight() can apply them during gradient calibration below
        # (kept for backwards compatibility with the original calibration flow)

        # --- Gradient scale calibration: backward pass on first population member ---
        print("[LossNormalisation] Computing gradient scales...")
        self._model_optimizer.zero_grad()
        self._model.set_latent_vector(self._Z[0])
        bundle = self._total_loss.compute(self._model, epoch_name=0, calibration_mode=True)
        # weight() with norm_manager applies loss_scales; we measure grad norms on top of them
        result = self._total_loss.weight(
            bundle, self._alm_manager, return_components=True, norm_manager=self._norm_manager, calibration_pass=True
        )

        trainable_params = [p for p in self._model.parameters() if p.requires_grad]
        prev_grad = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in trainable_params]

        components_with_grad = [(name, t) for name, t in result.components.items() if t.requires_grad]
        components_to_backward = [
            (name, t) for name, t in components_with_grad if not self._norm_manager.has_fixed_grad_scale(name)
        ]

        for idx, (name, component) in enumerate(components_to_backward):
            is_last = idx == len(components_to_backward) - 1
            component.backward(retain_graph=not is_last)
            curr_grad = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in trainable_params]
            delta_norm = sum((c - pv).pow(2).sum().item() for c, pv in zip(curr_grad, prev_grad, strict=False)) ** 0.5
            grad_scales[name] = self._norm_manager._target_grad / delta_norm if delta_norm > 0 else 1.0
            print(f"  Grad scale for {name}: delta_norm={delta_norm:.6g}, scale={grad_scales[name]:.6g}")
            prev_grad = curr_grad

        for name in calibrated_names:
            fixed_grad = self._norm_manager.get_fixed_grad_scale(name)
            if fixed_grad is not None:
                grad_scales[name] = fixed_grad
                print(f"  Grad scale for {name}: fixed={fixed_grad:.6g}")

        # Record final calibration with both scale dicts
        self._norm_manager.record_calibration(loss_scales, grad_scales, calibrated_names)
        self._model_optimizer.zero_grad()

        summary_items = []
        for n in sorted(calibrated_names):
            loss_scale = loss_scales.get(n, 1.0)
            grad_scale = grad_scales.get(n, 1.0)
            loss_status = f"loss={loss_scale:.3g}" if n in loss_scales else "loss=default"
            grad_status = f"grad={grad_scale:.3g}" if n in grad_scales else "grad=fixed/default"
            summary_items.append(f"{n}({loss_status}, {grad_status})")
        print(f"[LossNormalisation] Calibration complete. {len(calibrated_names)} losses calibrated:")
        for item in summary_items:
            print(f"  ✓ {item}")

    # ------------------------------------------------------------------
    # Phase 1 — per-vector backward (dispatches on aggregation method)
    # ------------------------------------------------------------------
    @record_function("DiversityManager._per_vector_backward")
    def _per_vector_backward(
        self,
        epoch: int,
    ) -> tuple[
        float,
        list[Tensor] | None,
        dict[str, float],
        list[int],
        dict[str, float],
        dict[str, np.ndarray] | None,
        dict[str, float],
        list[tuple[int, dict[str, Any] | None]],
    ]:
        """Dispatch the Phase-1 backward according to ``self._aggregation_method``.

        Returns the same tuple shape from all paths:
        (avg_total_loss, aggregated_constraints, avg_log_data, active_constraint_indices,
         avg_grad_log_data, phase1_conflict_vectors, per_slot_log_data, per_slot_grad_payloads).
        ``aggregated_constraints`` is the list fed to ALM's ``update_multipliers`` —
        ``mean(c_per_z)`` under method="mean", ``‖c_per_z‖₂`` under method="l2_norm".
        """
        if self._aggregation_method == "mean":
            return self._backward_mean(epoch)
        if self._aggregation_method == "l2_norm":
            if self._l2_backward_strategy == "single_pass":
                return self._backward_l2_single_pass(epoch)
            if self._l2_backward_strategy == "two_pass":
                return self._backward_l2_two_pass(epoch)
            raise ValueError(f"Unknown l2_backward_strategy: {self._l2_backward_strategy}")
        raise ValueError(f"Unknown aggregation method: {self._aggregation_method}")

    def _backward_mean(  # noqa: PLR0912, PLR0915
        self,
        epoch: int,
    ) -> tuple[
        float,
        list[Tensor] | None,
        dict[str, float],
        list[int],
        dict[str, float],
        dict[str, np.ndarray] | None,
        dict[str, float],
        list[tuple[int, dict[str, Any] | None]],
    ]:
        """Original mean-aggregation backward — per-z loss divided by N and accumulated.

        Each z's full weighted loss (incl. per-z ALM penalty) gets its own backward
        scaled by 1/N. ALM update receives ``mean(c_per_z)`` per constraint. This is
        the legacy behaviour; new runs should consider ``method=l2_norm`` if
        population_size>1 and constraints are stuck.
        """
        use_alm: bool = self._alm_manager is not None
        all_alm_constraints: list[list[Tensor]] = []
        summed_log_data: dict[str, float] = {}
        summed_grad_log_data: dict[str, float] = {}
        per_slot_log_data: dict[str, float] = {}
        per_slot_grad_payloads: list[tuple[int, dict[str, Any] | None]] = []
        total_loss_val: float = 0.0
        phase1_conflict_vectors: dict[str, np.ndarray] | None = None

        # Full constraint category order from YAML (never changes during a run)
        all_constraint_cats: list[str] = self._total_loss.constraint_category_order
        # Active constraint indices are epoch-deterministic (same across all population members)
        active_constraint_cats: list[str] = []
        active_constraint_indices: list[int] = []

        t0 = time.perf_counter()
        for i in range(self._population_size):
            self._model.set_latent_vector(self._Z[i])
            if self._vis_manager is not None:
                self._vis_manager.set_active_slot(i)

            bundle = self._total_loss.compute(self._model, epoch_name=epoch)

            # Collect per-slot gradient payload immediately after compute() while extractor
            # buffers still hold this slot's dL/dv (next compute() call will reset them).
            if self._vis_all_losses and self._vis_manager is not None:
                _slot_payload = self._total_loss.collect_vertex_grads_for_visualisation(
                    weights_by_name=self._total_loss._manual_weights,
                )
                per_slot_grad_payloads.append((i, _slot_payload))

            # Compute active constraint info for ALM once per epoch (same for all z_i since it depends only on epoch)
            if i == 0:
                active_constraint_cats = [cat for cat in all_constraint_cats if cat in bundle.constraints]
                active_constraint_indices = [j for j, cat in enumerate(all_constraint_cats) if cat in bundle.constraints]

            if self._diagnostic_mode:
                capture_components = (
                    self._conflict_tracker is not None
                    and self._conflict_cadence > 0
                    and epoch % self._conflict_cadence == 0
                    and i == 0
                )
                result = self._total_loss.weight(
                    bundle, alm_manager=self._alm_manager, return_components=True, norm_manager=self._norm_manager
                )
            else:
                capture_components = False
                result = self._total_loss.weight(
                    bundle, alm_manager=self._alm_manager, return_components=False, norm_manager=self._norm_manager
                )

            # Detach ALM-space constraints before backward — graph will be freed.
            # If loss normalisation is enabled, ALM updates operate on normalized
            # constraint values (same space as ALM penalties in TotalLossV2.weight).
            if use_alm:
                alm_constraints_for_member: list[Tensor] = []
                for cat in active_constraint_cats:
                    c_tensor = bundle.constraints[cat]
                    c_name = self._total_loss._constraint_by_category[cat][0]
                    c_for_alm = c_tensor * self._norm_manager.get_loss_scale(c_name)
                    alm_constraints_for_member.append(c_for_alm.detach())
                all_alm_constraints.append(alm_constraints_for_member)

            total_loss_val += result.log_data.get("raw_losses/total_loss", result.scalar.item())
            for k, v in result.log_data.items():
                summed_log_data[k] = summed_log_data.get(k, 0.0) + v
                if k.startswith("raw_losses/"):
                    per_slot_log_data[f"slots/slot_{i}/{k}"] = v

            if not self._diagnostic_mode:
                # Performance mode: single combined backward, no per-component grad tracking or conflict capture.
                (result.scalar / self._population_size).backward()
            else:
                # Diagnostic mode: per-component backward so we can track per-loss gradient norms and cosine conflict angles.
                trainable_params = [p for p in self._model.parameters() if p.requires_grad]
                prev_grad = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in trainable_params]
                per_population_conflict_vectors: dict[str, np.ndarray] | None = (
                    {} if capture_components and self._conflict_tracker is not None else None
                )

                for component_name, weighted_loss in result.components.items():
                    grad_key = f"optimizer_gradients/{component_name}"
                    if not weighted_loss.requires_grad:
                        summed_grad_log_data[grad_key] = summed_grad_log_data.get(grad_key, 0.0)
                        continue
                    (weighted_loss / self._population_size).backward(retain_graph=False)
                    curr_grad = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in trainable_params]
                    delta_norm = sum((c - pv).pow(2).sum().item() for c, pv in zip(curr_grad, prev_grad, strict=False)) ** 0.5
                    summed_grad_log_data[grad_key] = summed_grad_log_data.get(grad_key, 0.0) + delta_norm

                    if per_population_conflict_vectors is not None:
                        delta_grad = torch.cat(
                            [(c - pv).detach().cpu().flatten() for c, pv in zip(curr_grad, prev_grad, strict=False)]
                        )
                        per_population_conflict_vectors[component_name] = delta_grad.numpy()

                    prev_grad = curr_grad

                # Explicit zero-gradient logs for scheduled zero-weighted losses.
                for category in bundle.zero_weighted:
                    lookup = (
                        self._total_loss._objective_by_category.get(category)
                        or self._total_loss._constraint_by_category.get(category)
                        or self._total_loss._aux_by_category.get(category)
                    )
                    if lookup is None:
                        continue
                    if getattr(self._total_loss, "use_cse", False) and category in {
                        getattr(self._total_loss, "_cse_drag_category", None),
                        getattr(self._total_loss, "_cse_solar_category", None),
                    }:
                        continue
                    loss_name = lookup[0]
                    grad_key = f"optimizer_gradients/{loss_name}"
                    summed_grad_log_data[grad_key] = summed_grad_log_data.get(grad_key, 0.0)

                if per_population_conflict_vectors is not None:
                    phase1_conflict_vectors = per_population_conflict_vectors

        if self._vis_manager is not None:
            self._vis_manager.set_active_slot(None)

        if self._print_timing:
            mode_label = "Diagnostic" if self._diagnostic_mode else "Performance"
            n_losses = (
                len(self._total_loss._objective_by_category)
                + len(self._total_loss._constraint_by_category)
                + len(self._total_loss._aux_by_category)
            )
            print(f"{mode_label} mode forward/backward of: {n_losses} Losses: {time.perf_counter() - t0:.4f} seconds")

        mean_constraints: list[Tensor] | None = None
        if use_alm and all_alm_constraints:
            num_c = len(all_alm_constraints[0])
            mean_constraints = [torch.stack([pop[c_idx] for pop in all_alm_constraints]).mean() for c_idx in range(num_c)]

        avg_log_data = {k: v / self._population_size for k, v in summed_log_data.items()}
        avg_grad_log_data = {k: v / self._population_size for k, v in summed_grad_log_data.items()}
        avg_total_loss = total_loss_val / self._population_size
        return (
            avg_total_loss,
            mean_constraints,
            avg_log_data,
            active_constraint_indices,
            avg_grad_log_data,
            phase1_conflict_vectors,
            per_slot_log_data,
            per_slot_grad_payloads,
        )

    def _backward_l2_single_pass(  # noqa: PLR0915
        self,
        epoch: int,
    ) -> tuple[
        float,
        list[Tensor] | None,
        dict[str, float],
        list[int],
        dict[str, float],
        dict[str, np.ndarray] | None,
        dict[str, float],
        list[tuple[int, dict[str, Any] | None]],
    ]:
        """L2-norm population aggregation, single combined backward.

        Forwards every z without backward (keeps all N compute graphs alive in VRAM),
        builds the aggregated ALM penalty ``λ·‖c‖₂ + 0.5·μ·‖c‖₂²`` per constraint
        across the population plus the per-z non-ALM scalars (objectives + aux,
        each divided by N as in the mean path), then backwards once through the
        full combined loss. Memory cost scales ~N× over the mean path.
        """
        if not self._l2_grad_log_warned:
            print(
                "[DiversityTrainer] l2_norm aggregation: per-component grad-norm and "
                "conflict-vector logging not available; logging zeros."
            )
            self._l2_grad_log_warned = True

        N = self._population_size
        use_alm = self._alm_manager is not None
        summed_log_data: dict[str, float] = {}
        per_slot_log_data: dict[str, float] = {}
        per_slot_grad_payloads: list[tuple[int, dict[str, Any] | None]] = []
        total_loss_val: float = 0.0

        all_constraint_cats = self._total_loss.constraint_category_order
        active_constraint_cats: list[str] = []
        active_constraint_indices: list[int] = []
        per_z_c_scaled: list[list[Tensor]] = []
        per_z_non_alm: list[Tensor] = []

        t0 = time.perf_counter()
        for i in range(N):
            self._model.set_latent_vector(self._Z[i])
            if self._vis_manager is not None:
                self._vis_manager.set_active_slot(i)

            bundle = self._total_loss.compute(self._model, epoch_name=epoch)

            if i == 0:
                active_constraint_cats = [cat for cat in all_constraint_cats if cat in bundle.constraints]
                active_constraint_indices = [j for j, cat in enumerate(all_constraint_cats) if cat in bundle.constraints]

            member_c: list[Tensor] = []
            for cat in active_constraint_cats:
                c_name = self._total_loss._constraint_by_category[cat][0]
                member_c.append(bundle.constraints[cat] * self._norm_manager.get_loss_scale(c_name))
            per_z_c_scaled.append(member_c)

            result = self._total_loss.weight(
                bundle, alm_manager=self._alm_manager, return_components=True, norm_manager=self._norm_manager
            )
            non_alm = sum(
                (v for k, v in result.components.items() if not k.startswith("alm_")),
                torch.tensor(0.0, device=self._device),
            )
            per_z_non_alm.append(non_alm)

            total_loss_val += result.log_data.get("raw_losses/total_loss", result.scalar.item())
            for log_k, log_v in result.log_data.items():
                summed_log_data[log_k] = summed_log_data.get(log_k, 0.0) + log_v
                if log_k.startswith("raw_losses/"):
                    per_slot_log_data[f"slots/slot_{i}/{log_k}"] = log_v

        if self._vis_manager is not None:
            self._vis_manager.set_active_slot(None)

        aggregated_constraints: list[Tensor] | None = None
        alm_penalty: Tensor = torch.tensor(0.0, device=self._device)
        c_l2_live: list[Tensor] = []
        if use_alm and active_constraint_cats:
            assert self._alm_manager is not None
            for k in range(len(active_constraint_cats)):
                c_stack = torch.stack([per_z_c_scaled[i][k] for i in range(N)])
                c_l2_live.append(torch.sqrt(c_stack.pow(2).sum() + 1e-12))
            for k, cat in enumerate(active_constraint_cats):
                c_name = self._total_loss._constraint_by_category[cat][0]
                idx = active_constraint_indices[k]
                lam = self._alm_manager.lamdas[idx].detach().squeeze()
                mu = self._alm_manager.mus[idx].detach().squeeze()
                grad_scale = self._norm_manager.get_grad_scale(f"alm_{c_name}")
                alm_penalty = alm_penalty + grad_scale * (lam * c_l2_live[k] + 0.5 * mu * c_l2_live[k].pow(2))
            aggregated_constraints = [c.detach() for c in c_l2_live]

        non_alm_total = sum(per_z_non_alm, torch.tensor(0.0, device=self._device)) / N
        (alm_penalty + non_alm_total).backward()

        if self._print_timing:
            print(f"L2 single-pass backward: {time.perf_counter() - t0:.4f} seconds")

        avg_log_data = {k: v / N for k, v in summed_log_data.items()}
        return (
            total_loss_val / N,
            aggregated_constraints,
            avg_log_data,
            active_constraint_indices,
            {},
            None,
            per_slot_log_data,
            per_slot_grad_payloads,
        )

    def _backward_l2_two_pass(  # noqa: PLR0912, PLR0915
        self,
        epoch: int,
    ) -> tuple[
        float,
        list[Tensor] | None,
        dict[str, float],
        list[int],
        dict[str, float],
        dict[str, np.ndarray] | None,
        dict[str, float],
        list[tuple[int, dict[str, Any] | None]],
    ]:
        """L2-norm population aggregation, two-pass backward (1× memory).

        Pass 1 (no_grad): forward each z, collect detached c values + per-z log data.
        Pass 2 (grad): forward each z, build per-z weighted ALM term using precomputed
        detached weights ``c_i·(λ/‖c‖₂ + μ)·grad_scale``, backward immediately so only
        one compute graph is alive at a time. Final θ.grad equals what single_pass
        would produce (1e-6 numerical agreement expected).

        Note on determinism: Pass 1 and Pass 2 must produce identical constraint
        values. If any active loss uses stochastic point sampling, you'll see drift
        between the two passes. Recommended: keep `bb`-style mesh sampling fixed
        across an epoch (done at _setup), and treat any unexpected drift as a bug.
        """
        if not self._l2_grad_log_warned:
            print(
                "[DiversityTrainer] l2_norm aggregation: per-component grad-norm and "
                "conflict-vector logging not available; logging zeros."
            )
            self._l2_grad_log_warned = True

        N = self._population_size
        use_alm = self._alm_manager is not None
        summed_log_data: dict[str, float] = {}
        per_slot_log_data: dict[str, float] = {}
        per_slot_grad_payloads: list[tuple[int, dict[str, Any] | None]] = []
        total_loss_val: float = 0.0

        all_constraint_cats = self._total_loss.constraint_category_order
        active_constraint_cats: list[str] = []
        active_constraint_indices: list[int] = []
        c_per_z_detached: list[list[Tensor]] = []

        t0 = time.perf_counter()
        # Pass 1 — no-grad forward to collect c values (with loss_scale applied).
        with torch.no_grad():
            for i in range(N):
                self._model.set_latent_vector(self._Z[i])
                bundle = self._total_loss.compute(self._model, epoch_name=epoch)
                if i == 0:
                    active_constraint_cats = [cat for cat in all_constraint_cats if cat in bundle.constraints]
                    active_constraint_indices = [j for j, cat in enumerate(all_constraint_cats) if cat in bundle.constraints]
                member_c: list[Tensor] = []
                for cat in active_constraint_cats:
                    c_name = self._total_loss._constraint_by_category[cat][0]
                    member_c.append((bundle.constraints[cat] * self._norm_manager.get_loss_scale(c_name)).clone())
                c_per_z_detached.append(member_c)

        # Build detached aggregation weights once.
        num_c = len(active_constraint_cats)
        c_l2_detached: list[Tensor] = []
        weights: list[list[Tensor]] = []  # weights[i][k]
        if use_alm and num_c > 0:
            assert self._alm_manager is not None
            for k in range(num_c):
                c_stack = torch.stack([c_per_z_detached[i][k] for i in range(N)])
                c_l2_detached.append(torch.sqrt(c_stack.pow(2).sum() + 1e-12))
            weights = [[torch.zeros((), device=self._device) for _ in range(num_c)] for _ in range(N)]
            for k, cat in enumerate(active_constraint_cats):
                c_name = self._total_loss._constraint_by_category[cat][0]
                idx = active_constraint_indices[k]
                lam = self._alm_manager.lamdas[idx].detach().squeeze()
                mu = self._alm_manager.mus[idx].detach().squeeze()
                grad_scale = self._norm_manager.get_grad_scale(f"alm_{c_name}")
                factor = (lam / c_l2_detached[k] + mu) * grad_scale  # scalar
                for i in range(N):
                    weights[i][k] = c_per_z_detached[i][k] * factor

        # Pass 2 — live forward + per-z backward with precomputed detached weights.
        for i in range(N):
            self._model.set_latent_vector(self._Z[i])
            if self._vis_manager is not None:
                self._vis_manager.set_active_slot(i)

            bundle = self._total_loss.compute(self._model, epoch_name=epoch)
            result = self._total_loss.weight(
                bundle, alm_manager=self._alm_manager, return_components=True, norm_manager=self._norm_manager
            )
            non_alm = sum(
                (v for k, v in result.components.items() if not k.startswith("alm_")),
                torch.tensor(0.0, device=self._device),
            )

            alm_term: Tensor = torch.tensor(0.0, device=self._device)
            if use_alm and num_c > 0:
                for k, cat in enumerate(active_constraint_cats):
                    c_name = self._total_loss._constraint_by_category[cat][0]
                    c_live_scaled = bundle.constraints[cat] * self._norm_manager.get_loss_scale(c_name)
                    alm_term = alm_term + weights[i][k] * c_live_scaled

            (alm_term + non_alm / N).backward()

            total_loss_val += result.log_data.get("raw_losses/total_loss", result.scalar.item())
            for log_k, log_v in result.log_data.items():
                summed_log_data[log_k] = summed_log_data.get(log_k, 0.0) + log_v
                if log_k.startswith("raw_losses/"):
                    per_slot_log_data[f"slots/slot_{i}/{log_k}"] = log_v

        if self._vis_manager is not None:
            self._vis_manager.set_active_slot(None)

        if self._print_timing:
            print(f"L2 two-pass backward: {time.perf_counter() - t0:.4f} seconds")

        aggregated_constraints = c_l2_detached if (use_alm and num_c > 0) else None
        avg_log_data = {k: v / N for k, v in summed_log_data.items()}
        return (
            total_loss_val / N,
            aggregated_constraints,
            avg_log_data,
            active_constraint_indices,
            {},
            None,
            per_slot_log_data,
            per_slot_grad_payloads,
        )

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    @record_function("DiversityManager.step")
    def step(self, epoch: int = -1) -> dict[str, float]:  # noqa: PLR0915, PLR0912
        """
        One full optimization step: Phase 1 -> ALM update -> Phase 2 -> opt.step -> log/render.
        Returns: dict of metrics for the caller (stripped prefix for convenience), containing:
            * "per_vector_loss": the average unweighted loss across the population (for tracking the main optimization objective)
            * "diversity_loss": the raw diversity loss (for tracking diversity independently of its weight)
            * "total_loss": the combined weighted loss (for tracking overall optimization progress)
            * any metric in avg_log_data with "raw_losses/" prefix, stripped in the returned dict for convenience (e.g. "chamfer", "constraint1", etc.)


        """
        if self._norm_manager.is_enabled() and not self._norm_manager.is_calibrated():
            self._calibrate()

        _stop_epoch = self._z_resample_stop_epoch
        _resample_allowed = (
            self._z_resample_cadence > 0
            and epoch > 0
            and epoch >= self._z_resample_start_epoch
            and (_stop_epoch < 0 or epoch < _stop_epoch)
            and epoch % self._z_resample_cadence == 0
        )
        if _resample_allowed:
            self._resample_population(epoch)
            print(
                "[GINN] Latent resampling triggered at "
                f"epoch={epoch} (mode={self._z_resample_mode}, cadence={self._z_resample_cadence})",
                flush=True,
            )
            # Slot identity changes on resample — clear accumulated history so the new window
            # starts fresh.  Epoch indices use real epoch numbers so the x-axis stays continuous
            # across windows (window-2 starts at epoch N, not at 0).
            self._per_slot_loss_history.clear()
            self._per_slot_epoch_indices.clear()

        capture_conflict = (
            self._conflict_tracker is not None and self._conflict_cadence > 0 and epoch % self._conflict_cadence == 0
        )
        diversity_loss_val: float = 0.0
        diversity_weighted_loss_val: float = 0.0
        diversity_log_data: dict[str, float] = {}
        conflict_vectors: dict[str, np.ndarray] = {}

        if isinstance(self._model_optimizer, torch.optim.LBFGS):
            # L-BFGS needs all gradient-accumulating passes inside a closure.
            # With max_iter=1 and no line search the closure runs exactly once per step,
            # so Phase 1 + Phase 2 results are captured via a dict for use outside.
            _captured: dict = {}

            def closure() -> torch.Tensor:
                self._model_optimizer.zero_grad()
                _captured["phase1"] = self._per_vector_backward(epoch)

                if self._diversity_manager is not None and self._diversity_manager.should_compute(self._population_size):
                    diversity_result = self._diversity_manager.compute(
                        self._model,
                        self._Z,
                        num_points=self._diversity_num_points,
                        bbox_lo=self._diversity_bbox_lo,
                        bbox_hi=self._diversity_bbox_hi,
                        norm_order=self._diversity_norm_order,
                        neighbor_agg_fn=self._diversity_neighbor_agg_fn,
                    )
                    diversity_result.weighted.backward(retain_graph=False)
                    _captured["diversity"] = diversity_result

                if self._gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._gradient_clip_norm)
                return torch.tensor(_captured["phase1"][0], dtype=torch.float32, device=self._device)

            self._model_optimizer.step(closure)
            (
                per_vector_loss_val,
                mean_constraints,
                avg_log_data,
                active_constraint_indices,
                per_loss_gradient_log_data,
                phase1_conflict_vectors,
                per_slot_log_data,
                per_slot_grad_payloads,
            ) = _captured["phase1"]
            conflict_vectors = dict(phase1_conflict_vectors or {})

            if "diversity" in _captured:
                div_result = _captured["diversity"]
                diversity_loss_val = float(div_result.raw.detach().item())
                diversity_weighted_loss_val = float(div_result.weighted.detach().item())
                diversity_log_data = self._diversity_manager.build_log_data(diversity_loss_val)  # type: ignore[union-attr]
            elif self._diversity_manager is not None and self._diversity_manager.enabled:
                per_loss_gradient_log_data[f"optimizer_gradients/{self._diversity_manager.loss_name}"] = 0.0

        else:
            self._model_optimizer.zero_grad()

            # Phase 1 — per-vector expensive backward
            (
                per_vector_loss_val,
                mean_constraints,
                avg_log_data,
                active_constraint_indices,
                per_loss_gradient_log_data,
                phase1_conflict_vectors,
                per_slot_log_data,
                per_slot_grad_payloads,
            ) = self._per_vector_backward(epoch)
            conflict_vectors = dict(phase1_conflict_vectors or {})

            # Phase 2 — cheap diversity backward
            if self._diversity_manager is not None and self._diversity_manager.should_compute(self._population_size):
                diversity_result = self._diversity_manager.compute(
                    self._model,
                    self._Z,
                    num_points=self._diversity_num_points,
                    bbox_lo=self._diversity_bbox_lo,
                    bbox_hi=self._diversity_bbox_hi,
                    norm_order=self._diversity_norm_order,
                    neighbor_agg_fn=self._diversity_neighbor_agg_fn,
                )

                trainable_params = [p for p in self._model.parameters() if p.requires_grad]
                prev_grad = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in trainable_params]
                diversity_result.weighted.backward(retain_graph=False)
                curr_grad = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in trainable_params]
                delta_norm = sum((c - pv).pow(2).sum().item() for c, pv in zip(curr_grad, prev_grad, strict=False)) ** 0.5
                diversity_grad_key = f"optimizer_gradients/{self._diversity_manager.loss_name}"
                per_loss_gradient_log_data[diversity_grad_key] = delta_norm

                if capture_conflict and self._conflict_tracker is not None:
                    diversity_delta = torch.cat(
                        [(c - pv).detach().cpu().flatten() for c, pv in zip(curr_grad, prev_grad, strict=False)]
                    )
                    conflict_vectors[self._diversity_manager.loss_name] = diversity_delta.numpy()

                diversity_loss_val = float(diversity_result.raw.detach().item())
                diversity_weighted_loss_val = float(diversity_result.weighted.detach().item())
                diversity_log_data = self._diversity_manager.build_log_data(diversity_loss_val)
            elif self._diversity_manager is not None and self._diversity_manager.enabled:
                per_loss_gradient_log_data[f"optimizer_gradients/{self._diversity_manager.loss_name}"] = 0.0

            if self._gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._gradient_clip_norm)

            self._model_optimizer.step()

        # External ALM update — only update multipliers for currently active constraints
        if mean_constraints is not None and self._alm_manager is not None:
            self._alm_manager.update_multipliers(mean_constraints, active_indices=active_constraint_indices)

        if capture_conflict and self._conflict_tracker is not None and conflict_vectors:
            self._conflict_tracker.save_from_grad_vectors(conflict_vectors, epoch)

        raw_metrics = {
            name.removeprefix("raw_losses/"): value for name, value in avg_log_data.items() if name.startswith("raw_losses/")
        }
        due_recalibration_names = self._norm_manager.get_recalibration_requests(epoch, raw_metrics)
        if due_recalibration_names:
            print(f"[LossNormalisation] Recalibrating after epoch {epoch}: {sorted(due_recalibration_names)}")
            self._calibrate()
            self._norm_manager.mark_recalibrated(due_recalibration_names, epoch)

        # Render (reference shape = Z[0])
        if self._population_size > 0:
            self._model.set_latent_vector(self._Z[0])
        self._render(epoch, per_vector_loss_val, per_slot_grad_payloads=per_slot_grad_payloads)

        # Log once per epoch
        self._log(avg_log_data, diversity_log_data, per_loss_gradient_log_data, epoch, per_slot_log_data)
        self._total_loss.finalize_epoch_mesh_stats()

        metrics: dict[str, float] = {
            "per_vector_loss": per_vector_loss_val,
            "diversity_loss": diversity_loss_val,
            "diversity_weighted_loss": diversity_weighted_loss_val,
            "total_loss": per_vector_loss_val + diversity_weighted_loss_val,
        }
        # Surface averaged raw metrics for the caller (stripped prefix for convenience)
        for k, v in avg_log_data.items():
            if k.startswith("raw_losses/"):
                metrics[k.removeprefix("raw_losses/")] = v
        return metrics

    def get_population(self) -> Tensor:
        return self._Z.detach().clone()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(
        self,
        avg_log_data: dict[str, float],
        diversity_log_data: dict[str, float],
        gradient_log_data: dict[str, float],
        epoch: int,
        per_slot_log_data: dict[str, float] | None = None,
    ) -> None:
        """Single wandb.log call with all averaged scalars + ALM multipliers + diversity + gradients + per-slot."""
        if not self._use_wandb or wandb.run is None:
            return

        log_dict: dict[str, float] = dict(avg_log_data)
        if self._alm_manager is not None:
            log_dict.update(self._alm_manager.get_telemetry())
        log_dict.update(self._norm_manager.get_telemetry())
        log_dict.update(get_optimizer_telemetry(self._model_optimizer))
        log_dict.update(diversity_log_data)
        log_dict.update(gradient_log_data)
        # Note: per-slot raw scalars are NOT added to log_dict to avoid hundreds of individual
        # wandb panels. They are instead aggregated into per-loss line charts below.

        # Competition charts (avg-of-population)
        raw_floats: dict[str, float] = {
            k.removeprefix("raw_losses/"): abs(v)
            for k, v in log_dict.items()
            if k.startswith("raw_losses/") and k != "raw_losses/total_loss"
        }
        normalized_floats: dict[str, float] = {
            k.removeprefix("normalized_losses/"): abs(v)
            for k, v in log_dict.items()
            if k.startswith("normalized_losses/") and k != "normalized_losses/total_loss"
        }
        weighted_floats: dict[str, float] = {
            k.removeprefix("weighted_losses/"): abs(v)
            for k, v in log_dict.items()
            if k.startswith("weighted_losses/") and k != "weighted_losses/total_loss"
        }
        optimizer_floats: dict[str, float] = {
            k.removeprefix("optimizer_losses/"): abs(v)
            for k, v in log_dict.items()
            if k.startswith("optimizer_losses/") and k != "optimizer_losses/total_loss"
        }
        competition_log = self._log_competition_charts(
            raw_floats,
            normalized_floats,
            weighted_floats,
            optimizer_floats,
            epoch,
        )

        # Per-slot line charts (one interactive line per slot, one chart per raw loss)
        slot_log = self._log_slot_line_charts(per_slot_log_data or {}, epoch)

        # Merge competition charts + slot charts into single log call to avoid WandB step corruption
        log_dict.update(competition_log)
        log_dict.update(slot_log)
        wandb.log(log_dict, step=epoch)

    def _log_slot_line_charts(
        self,
        per_slot_log_data: dict[str, float],
        epoch: int,
    ) -> dict[str, Any]:
        """Build one wandb.plot.line_series chart per raw loss, with one line per latent slot.

        Returns a dict ready to merge into the main wandb.log call.
        Silently returns {} when per-slot charting is disabled (resampling has fired)
        or when wandb is not active.
        """
        if not self._per_slot_charts_active or not per_slot_log_data:
            return {}
        if not self._use_wandb or wandb.run is None:
            return {}

        # Parse: "slots/slot_{i}/raw_losses/{loss_name}" -> {loss_name: {slot_idx: value}}
        slot_values: dict[str, dict[int, float]] = {}
        for key, val in per_slot_log_data.items():
            # Expected format: slots/slot_<i>/raw_losses/<loss_name>
            parts = key.split("/")
            if len(parts) < 4 or parts[0] != "slots" or not parts[1].startswith("slot_") or parts[2] != "raw_losses":
                continue
            try:
                slot_idx = int(parts[1].removeprefix("slot_"))
            except ValueError:
                continue
            loss_name = "/".join(parts[3:])
            slot_values.setdefault(loss_name, {})[slot_idx] = val

        if not slot_values:
            return {}

        # Build / extend per-slot history
        self._per_slot_epoch_indices.append(epoch)
        n_slots = self._population_size
        for loss_name, by_slot in slot_values.items():
            epoch_vals = [by_slot.get(i, float("nan")) for i in range(n_slots)]
            self._per_slot_loss_history.setdefault(loss_name, []).append(epoch_vals)

        xs = self._per_slot_epoch_indices
        keys = [f"slot_{i}" for i in range(n_slots)]
        result: dict[str, Any] = {}
        for loss_name, history in self._per_slot_loss_history.items():
            ys = [[history[t][i] for t in range(len(history))] for i in range(n_slots)]
            result[f"raw_losses_individual_slots/{loss_name}"] = wandb.plot.line_series(
                xs=xs,
                ys=ys,
                keys=keys,
                title=loss_name,
                xname="epoch",
            )
        return result

    def _log_competition_charts(  # noqa: PLR0915
        self,
        raw_floats: dict[str, float],
        normalized_floats: dict[str, float],
        weighted_floats: dict[str, float],
        optimizer_floats: dict[str, float],
        epoch: int,
    ) -> dict[str, Any]:
        """Generate pie/bar/radar/stacked-area charts of loss shares; return dict for WandB."""
        if not self._use_wandb or wandb.run is None:
            return {}
        matplotlib.use("Agg")
        self._raw_loss_history.append(dict(raw_floats))
        self._normalized_loss_history.append(dict(normalized_floats))
        self._weighted_loss_history.append(dict(weighted_floats))
        self._optimizer_loss_history.append(dict(optimizer_floats))

        competition_log: dict[str, Any] = {}

        def _safe_pie(values: list[float], labels: list[str], title: str) -> plt.Figure:
            fig, ax = plt.subplots(figsize=(6, 6))
            filtered = [
                (v, lbl)
                for v, lbl in zip(values, labels, strict=True)
                if isinstance(v, (int, float)) and math.isfinite(v) and v > 0
            ]
            if filtered:
                values = [v for v, _ in filtered]
                labels = [lbl for _, lbl in filtered]
            else:
                values, labels = [], []
            total = sum(values)
            if total == 0 or len(values) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.set_title(title)
            return fig

        def _safe_hbar(values: list[float], labels: list[str], title: str) -> plt.Figure:
            fig, ax = plt.subplots(figsize=(8, 1.8))
            total = sum(values)
            if total == 0 or len(values) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            else:
                pcts = [v / total * 100 for v in values]
                left = 0.0
                colors = matplotlib.colormaps["tab10"].colors  # type: ignore[attr-defined]
                for i, (pct, lbl) in enumerate(zip(pcts, labels, strict=True)):
                    ax.barh(0, pct, left=left, color=colors[i % len(colors)], edgecolor="white")
                    if pct > 3:
                        ax.text(
                            left + pct / 2,
                            0,
                            f"{lbl}\n{pct:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white",
                            fontweight="bold",
                        )
                    left += pct
            ax.set_xlim(0, 100)
            ax.set_xlabel("% of total")
            ax.set_yticks([])
            ax.set_title(title)
            fig.tight_layout()
            return fig

        raw_labels = list(raw_floats.keys())
        raw_vals = [raw_floats[k] for k in raw_labels]
        n_labels = list(normalized_floats.keys())
        n_vals = [normalized_floats[k] for k in n_labels]
        w_labels = list(weighted_floats.keys())
        w_vals = [weighted_floats[k] for k in w_labels]
        opt_labels = list(optimizer_floats.keys())
        opt_vals = [optimizer_floats[k] for k in opt_labels]

        fig_a = _safe_pie(raw_vals, raw_labels, f"Raw Losses (epoch {epoch})")
        competition_log["loss_competition/raw_pie"] = wandb.Image(fig_a)
        plt.close(fig_a)
        fig_b = _safe_pie(w_vals, w_labels, f"Weighted Losses (epoch {epoch})")
        competition_log["loss_competition/weighted_pie"] = wandb.Image(fig_b)
        plt.close(fig_b)
        fig_b2 = _safe_pie(n_vals, n_labels, f"Normalized Losses (epoch {epoch})")
        competition_log["loss_competition/normalized_pie"] = wandb.Image(fig_b2)
        plt.close(fig_b2)
        fig_b3 = _safe_pie(opt_vals, opt_labels, f"Optimizer Losses (epoch {epoch})")
        competition_log["loss_competition/optimizer_pie"] = wandb.Image(fig_b3)
        plt.close(fig_b3)
        fig_c = _safe_hbar(raw_vals, raw_labels, f"Raw Loss Shares % (epoch {epoch})")
        competition_log["loss_competition/raw_bar"] = wandb.Image(fig_c)
        plt.close(fig_c)
        fig_d = _safe_hbar(w_vals, w_labels, f"Weighted Loss Shares % (epoch {epoch})")
        competition_log["loss_competition/weighted_bar"] = wandb.Image(fig_d)
        plt.close(fig_d)
        fig_d2 = _safe_hbar(n_vals, n_labels, f"Normalized Loss Shares % (epoch {epoch})")
        competition_log["loss_competition/normalized_bar"] = wandb.Image(fig_d2)
        plt.close(fig_d2)
        fig_d3 = _safe_hbar(opt_vals, opt_labels, f"Optimizer Loss Shares % (epoch {epoch})")
        competition_log["loss_competition/optimizer_bar"] = wandb.Image(fig_d3)
        plt.close(fig_d3)

        # Radar: union of keys, each series normalized to its own max for comparable angles
        all_keys = list(dict.fromkeys(list(raw_floats.keys()) + list(weighted_floats.keys())))
        if len(all_keys) >= 2:
            angles = np.linspace(0, 2 * np.pi, len(all_keys), endpoint=False).tolist()
            angles += angles[:1]
            raw_vals_r = [raw_floats.get(k, 0.0) for k in all_keys]
            w_vals_r = [weighted_floats.get(k, 0.0) for k in all_keys]
            raw_max = max(raw_vals_r) or 1.0
            w_max = max(w_vals_r) or 1.0
            raw_norm = [v / raw_max for v in raw_vals_r] + [raw_vals_r[0] / raw_max]
            w_norm = [v / w_max for v in w_vals_r] + [w_vals_r[0] / w_max]

            fig_e, ax_e = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
            ax_e.plot(angles, raw_norm, "o--", color="steelblue", linewidth=1.5, label="raw (norm)")
            ax_e.fill(angles, raw_norm, color="steelblue", alpha=0.15)
            ax_e.plot(angles, w_norm, "o-", color="tomato", linewidth=2, label="weighted (norm)")
            ax_e.fill(angles, w_norm, color="tomato", alpha=0.15)
            ax_e.set_xticks(angles[:-1])
            ax_e.set_xticklabels(all_keys, fontsize=9)
            ax_e.set_title(f"Loss Radar (epoch {epoch})", pad=15)
            ax_e.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            fig_e.tight_layout()
            competition_log["loss_competition/radar"] = wandb.Image(fig_e)
            plt.close(fig_e)

        if len(self._raw_loss_history) >= 2:
            epochs_so_far = list(range(len(self._raw_loss_history)))

            def _stacked_area(history: list[dict[str, float]], ax: plt.Axes, title: str) -> None:
                all_k = list(dict.fromkeys(k for d in history for k in d))
                matrix = np.array([[d.get(k, 0.0) for k in all_k] for d in history])
                row_sums = matrix.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                matrix_pct = matrix / row_sums * 100
                tab_colors = matplotlib.colormaps["tab10"].colors  # type: ignore[attr-defined]
                ax.stackplot(
                    epochs_so_far,
                    matrix_pct.T,
                    labels=all_k,
                    colors=[tab_colors[i % len(tab_colors)] for i in range(len(all_k))],
                    alpha=0.85,
                )
                ax.set_ylabel("% share")
                ax.set_title(title)
                ax.legend(loc="upper left", fontsize=8, ncol=2)
                ax.set_ylim(0, 100)

            fig_f, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            _stacked_area(self._raw_loss_history, ax_top, "Raw Loss Competition Over Training")
            _stacked_area(self._weighted_loss_history, ax_bot, "Weighted Loss Competition Over Training")
            ax_bot.set_xlabel("Epoch")
            fig_f.tight_layout()
            competition_log["loss_competition/stacked_area"] = wandb.Image(fig_f)
            plt.close(fig_f)

        return competition_log

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render(  # noqa: PLR0912, PLR0915
        self,
        epoch: int,
        total_loss_val: float,
        per_slot_grad_payloads: list[tuple[int, dict[str, Any] | None]] | None = None,
    ) -> None:
        """Extract mesh from Z[0] and forward to either VisualisationManager or CAD fallback."""
        # vis_cadence_optimisation (from vis_config.yaml) takes precedence over legacy rendering_freq
        _cadence = self._vis_cadence_optimisation if self._vis_cadence_optimisation > 0 else self._rendering_freq
        if _cadence is None or epoch % _cadence != 0:
            return

        # Reuse the mesh already captured by the gradient pipeline; only
        # fall back to a fresh extraction when no grad payload is available
        # (calibration mode, first epoch before any extractor ran, etc.).
        grads_payload = self._total_loss.collect_vertex_grads_for_visualisation(
            weights_by_name=self._total_loss._manual_weights,
        )
        extra_state: dict[str, Any] = {}
        per_loss_meshes: dict[str, Any] = {}
        per_loss_grads: dict[str, Any] = {}
        if grads_payload is not None:
            verts = grads_payload["vertices"]
            faces = grads_payload["faces"]
            per_loss_grads = grads_payload["per_loss"]
            per_loss_meshes = grads_payload.get("per_loss_meshes", {})
            extra_state["vertice_grads_per_loss"] = per_loss_grads
            extra_state["vertice_meshes_per_loss"] = per_loss_meshes
        else:
            verts, faces = self._mesh_extractor.extract_mesh(sdf_model=self._model)
        # Visualization: write PLY for every population slot
        if self._vis_manager is not None:
            for i in range(self._population_size):
                # vis_strategy=static: evaluate mesh at original pretrained Z circle positions, not current optimizer Z
                if self._vis_strategy == "static" and self._Z_pretrained is not None:
                    self._model.set_latent_vector(self._Z_pretrained[i])
                else:
                    self._model.set_latent_vector(self._Z[i])
                _verts, _faces = self._vis_mesh_extractor.extract_mesh(sdf_model=self._model)
                self._vis_manager.set_active_slot(i)
                self._vis_manager.vis_mesh_shape(epoch, _verts, _faces)
                self._vis_manager.vis_mesh_assembly_v4(epoch, _verts, _faces)
                self._vis_manager.set_active_slot(None)
            # Restore model to Z[0] as reference shape
            self._model.set_latent_vector(self._Z[0])
            # Per-slot gradient vis (vis_all_losses=True): each slot gets its own
            # gradient PLY written under its slot subdirectory.
            if self._vis_all_losses and per_slot_grad_payloads:
                for slot_idx, slot_payload in per_slot_grad_payloads:
                    if slot_payload is None:
                        continue
                    slot_per_loss = slot_payload["per_loss"]
                    slot_meshes = slot_payload.get("per_loss_meshes", {})
                    self._vis_manager.set_active_slot(slot_idx)
                    for _loss_key, _grads in slot_per_loss.items():
                        _mesh_entry = slot_meshes.get(_loss_key)
                        if _mesh_entry is None:
                            continue
                        _v, _f = _mesh_entry
                        self._vis_manager.vis_mesh_gradient(epoch, _loss_key, _v, _f, _grads)
                    self._vis_manager.set_active_slot(None)
            else:
                # Default: reference-shape gradient (Z[0] perspective, no slot tag).
                for _loss_key, _grads in per_loss_grads.items():
                    _mesh_entry = per_loss_meshes.get(_loss_key)
                    if _mesh_entry is None:
                        continue
                    _v, _f = _mesh_entry
                    self._vis_manager.vis_mesh_gradient(epoch, _loss_key, _v, _f, _grads)

        # Circle survey: every N optimisation epochs, sweep the pretrained z-circle.
        if isinstance(self._vis_manager, CentralVisManagerV2):
            _survey_cfg: dict = self._vis_manager.latent_optimisation_cfg.get("vis_latent_circle_survey", {})
            _survey_enabled = _survey_cfg.get("enabled", False)
            _survey_cadence = int(_survey_cfg.get("cadence", 0))
            if _survey_enabled and _survey_cadence > 0 and epoch % _survey_cadence == 0:
                self._vis_manager.vis_latent_circle_survey(
                    model=self._model,
                    z_radius=float(_survey_cfg.get("z_radius", 2.0)),
                    n_samples=int(_survey_cfg.get("n_circle_samples", 90)),
                    device=str(self._device),
                    epoch=epoch,
                )

        if self._use_neural_renderer and self._renderer_manager is not None:
            try:
                self._renderer_manager.update(
                    {
                        "model": self._model,
                        "iteration": epoch,
                        "loss": total_loss_val,
                        "vertices": verts,
                        "faces": faces,
                        "assembly_id": None,
                        "bb_assembly_id": None,
                    }
                )
            except Exception as e:
                print(f"Renderer wrapper update failed: {e}")
        elif self._vis_manager is None:
            # Only fall back to CAD visualisation when no other renderer is
            # active; if vis_manager is set, it already exported PLYs above.
            self._run_cad_vis(verts, faces, epoch)

    def _run_cad_vis(self, verts: Any, faces: Any, epoch: int) -> None:
        """CAD rendering fallback: export STL, render to PNG via CADVisualizer, upload to wandb."""
        if not self._use_wandb or wandb.run is None:
            return

        verts_np = verts.detach().cpu().numpy() if torch.is_tensor(verts) else verts
        faces_np = faces.detach().cpu().numpy() if torch.is_tensor(faces) else faces
        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)

        self._log_dir.mkdir(parents=True, exist_ok=True)
        stl_path = self._log_dir / f"mesh_epoch_{epoch}.stl"
        image_path = self._log_dir / f"render_epoch_{epoch}.png"
        mesh.export(str(stl_path))

        visualizer = CADVisualizer(stl_path=str(stl_path), out_image_path=str(image_path))
        visualizer.render_until_blue_detected()

        if stl_path.exists():
            wandb.log({"mesh_3d": wandb.Object3D(str(stl_path))}, step=epoch)
            stl_path.unlink()
        if image_path.exists():
            wandb.log({"render_image": wandb.Image(str(image_path))}, step=epoch, commit=True)
            image_path.unlink()
