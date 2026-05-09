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
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager import (
    CentralVisManager,
)
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

    def __init__(
        self,
        model: SupportsLatentConcat,
        total_loss: TotalLossV2,
        alm_manager: ALMManager | None,
        population_size: int,
        z_dim: int,
        device: torch.device | str = "cuda",
        model_lr: float = 1e-4,
        diversity_manager: DiversityLossManager | None = None,
        diversity_num_points: int = 2048,
        diversity_bbox_scale_factor: float = 1.0,
        diversity_norm_order: int = 2,
        diversity_neighbor_agg_fn: str = "min",
        force_z0_to_origin: bool = True,
        z_resample_cadence: int = 0,  # so the original pretrained spots remain
        initial_Z: Tensor | None = None,
        z_resample_start_epoch: int = 0,
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
        vis_manager: CentralVisManager | None = None,
    ) -> None:
        """
        Args:
            model: SDF network supporting set_latent_vector / get_latent_vector.
            total_loss: Pure-computation loss manager (TotalLossV2).
            alm_manager: External ALM state manager. ``None`` -> manual constraint weights.
            population_size: Number of latents (N). Set to 1 for single-model training.
            z_dim: Dimensionality of each latent vector.
            device: Compute device.
            model_lr: Adam learning rate for model parameters.            gradient_clip_norm: If set, clips the total parameter gradient L2-norm to this
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
        self._renderer_manager: VisualisationManager | None = renderer_manager
        self._rendering_freq: int | None = rendering_freq
        self._use_neural_renderer: bool = use_neural_renderer

        self._use_wandb: bool = use_wandb
        self._log_dir: Path = log_dir or Path("subprojects/wp4_aeroshell_optimizer/v2/logs")
        self._vis_manager: CentralVisManager | None = vis_manager

        generator_device = self._device.type if self._device.type in {"cpu", "cuda"} else "cpu"
        self._resample_generator = torch.Generator(device=generator_device)

        # Fixed population of latent vectors (not trained). Z[0] optionally pinned to origin.
        if initial_Z is not None:
            expected_shape = (population_size, z_dim)
            if tuple(initial_Z.shape) != expected_shape:
                raise ValueError(
                    f"initial_Z shape {tuple(initial_Z.shape)} does not match expected "
                    f"{expected_shape}. "
                    f"Set population_size={initial_Z.shape[0]} in GINN.yaml to match the "
                    f"pretrained population, or re-run pretraining with "
                    f"population_size={population_size}."
                )
            self._Z: Tensor = initial_Z.clone().to(self._device)
        else:
            self._Z = torch.randn(population_size, z_dim, device=self._device, dtype=torch.float32)

        if self._force_z0_to_origin and self._population_size > 0:
            self._Z[0] = torch.zeros(z_dim, device=self._device, dtype=torch.float32)

        self._model_optimizer: Optimizer = torch.optim.Adam(self._model.parameters(), lr=model_lr)
        self._gradient_clip_norm: float | None = gradient_clip_norm if gradient_clip_norm and gradient_clip_norm > 0 else None

        # Rolling history for the stacked-area competition chart
        self._raw_loss_history: list[dict[str, float]] = []
        self._normalized_loss_history: list[dict[str, float]] = []
        self._weighted_loss_history: list[dict[str, float]] = []
        self._optimizer_loss_history: list[dict[str, float]] = []

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
        self._Z[start_idx:] = torch.randn(
            self._population_size - start_idx,
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
    # Phase 1 — per-vector backward
    # ------------------------------------------------------------------
    @record_function("DiversityManager._per_vector_backward")
    def _per_vector_backward(  # noqa: PLR0912, PLR0915
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
    ]:
        """
        Iterate over the population and immediately backward each z's loss.

        Returns mean scalar loss (for metrics), averaged mean constraints across the
        population (for ALM update; None if no ALM), averaged loss log_data,
        active constraint indices, averaged per-loss gradient log_data, optional
        conflict vectors for the first population member, and per-slot raw loss log_data.
        """
        use_alm: bool = self._alm_manager is not None
        all_alm_constraints: list[list[Tensor]] = []
        summed_log_data: dict[str, float] = {}
        summed_grad_log_data: dict[str, float] = {}
        per_slot_log_data: dict[str, float] = {}
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
            print(f"{mode_label} mode backward {n_losses} Losses: {time.perf_counter() - t0:.4f} seconds")

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
        )

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    @record_function("DiversityManager.step")
    def step(self, epoch: int = -1) -> dict[str, float]:
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

        if self._z_resample_cadence > 0 and epoch > 0 and epoch % self._z_resample_cadence == 0:
            self._resample_population(epoch)

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
        ) = self._per_vector_backward(epoch)

        # External ALM update — only update multipliers for currently active constraints
        if mean_constraints is not None and self._alm_manager is not None:
            self._alm_manager.update_multipliers(mean_constraints, active_indices=active_constraint_indices)

        capture_conflict = (
            self._conflict_tracker is not None and self._conflict_cadence > 0 and epoch % self._conflict_cadence == 0
        )

        # Phase 2 — cheap diversity backward
        diversity_loss_val: float = 0.0
        diversity_weighted_loss_val: float = 0.0
        diversity_log_data: dict[str, float] = {}
        conflict_vectors: dict[str, np.ndarray] = dict(phase1_conflict_vectors or {})
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

        if capture_conflict and self._conflict_tracker is not None and conflict_vectors:
            self._conflict_tracker.save_from_grad_vectors(conflict_vectors, epoch)

        if self._gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._gradient_clip_norm)

        self._model_optimizer.step()

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
        self._render(epoch, per_vector_loss_val)

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
        log_dict.update(diversity_log_data)
        log_dict.update(gradient_log_data)
        if per_slot_log_data:
            log_dict.update(per_slot_log_data)

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

        # Merge competition charts into single log call to avoid WandB step corruption
        log_dict.update(competition_log)
        wandb.log(log_dict, step=epoch)

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
    def _render(self, epoch: int, total_loss_val: float) -> None:
        """Extract mesh from Z[0] and forward to either VisualisationManager or CAD fallback."""
        if self._rendering_freq is None or epoch % self._rendering_freq != 0:
            return

        verts, faces = self._mesh_extractor.extract_mesh(sdf_model=self._model)

        # CentralVisManager path: write PLY for every population slot
        if self._vis_manager is not None:
            for i in range(self._population_size):
                self._model.set_latent_vector(self._Z[i])
                _verts, _faces = self._mesh_extractor.extract_mesh(sdf_model=self._model)
                self._vis_manager.save_ply(
                    f"mesh_shape/slot_{i:02d}",
                    epoch,
                    _verts.detach().cpu().numpy(),
                    _faces.detach().cpu().numpy(),
                )
            # Restore model to Z[0] as reference shape
            self._model.set_latent_vector(self._Z[0])
            return

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
        else:
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
