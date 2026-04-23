"""Modular total-loss pipeline for the aeroshell optimiser.

1. ``_compute_raw_losses``  - evaluate every active loss instance
2. ``_apply_weighting``     - independent objective & constraint weighting
3. ``_handle_telemetry``    - WandB logging, mesh rendering, memory diagnostics

**Weighting modes** - controlled by two independent flags in ``general_info``:

+-----------------------+----------------------------+------------------------------+
|                       | **CSE objectives**         | **Manual objectives**        |
|                       | ``CSE='True'``             | ``CSE`` absent / ``'False'`` |
+=======================+============================+==============================+
| **ALM constraints**   | ALM + CSE                  | ALM + manual obj weights     |
| ``ALM='True'``        |                            |                              |
+-----------------------+----------------------------+------------------------------+
| **Manual constraints** | Manual cstr + CSE          | Fully manual                |
| ``ALM`` absent/       |                            |                              |
| ``'False'``           |                            |                              |
+-----------------------+----------------------------+------------------------------+

**CSE prefactors**: When CSE is active each objective loss is multiplied by a
configurable prefactor before being fed to the cruising-speed solver::

    cse_prefactors:
      drag: 1.0
      solar: 1.0

**Per-constraint ALM initial values**: ``initial_lambda``, ``initial_mu`` and
``initial_nu`` are lists of length *n_constraints* (one entry per active constraint in
`[bb, eikonal, smoothness, connectedness, developability]` order). A single scalar is
broadcast to all.

**Category convention**:

* ``aero``, ``solar``                → *objective* losses
* ``bb``, ``eikonal``, ``smoothness``, ``connectedness``, ``developability`` → *constraint* losses
* everything else                    → *auxiliary* losses (always manual-weighted)
"""

from __future__ import annotations

import inspect
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeAlias

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

import wandb
from subprojects.wp3_solar_estimator.mesh_extractors.non_diff_algos.marching_cubes import (
    MeshExtractor,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.alm import ALMManager
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.cruising_speed import cruising_speed
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsModelContract
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.dynamic_loading import load_class_from_path
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.visualizer import CADVisualizer
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.visual_wrapper import VisualisationManager
from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor

TrainModel: TypeAlias = SupportsModelContract


class TotalLoss:
    """Aggregates, weights (ALM / manual x CSE / manual), and logs all loss components.

    Complies with the main training file which:
    * Calls ``get_total_loss(model, epoch_name)`` each epoch.
    * Iterates ``_last_weighted_losses`` for per-component ``.backward(retain_graph=False)``.
    * Reads ``_gradient_memory_diagnostics`` for optional per-backward memory logging.
    """

    # ------------------------------------------------------------------
    # initialisation
    # ------------------------------------------------------------------
    def __init__(
        self,
        dev: str | torch.device = "cuda",
        renderer_manager: VisualisationManager | None = None,
        external_alm: bool = False,  # Whether to do the ALM updates inside the loop or leave them to an external manager (like GINN's diversity manager)
    ) -> None:
        self._device: torch.device = torch.device(dev) if isinstance(dev, str) else dev
        self._renderer_manager: VisualisationManager | None = renderer_manager
        self._external_alm: bool = external_alm

        # Populated each call to get_total_loss(); maps component name → weighted
        # live tensor.  The training loop calls .backward() per entry.
        self._last_weighted_losses: dict[str, torch.Tensor] = {}
        # External ALM manager for ALM state
        self.alm_manager: ALMManager | None = None
        # Stores the last computed cruising speed (if CSE active) for telemetry.
        self._last_cruising_speed: torch.Tensor | None = None
        self._last_cruising_speed_diagnostics: dict | None = None
        # Rolling history for loss-competition stacked-area chart (one dict per epoch).
        self._raw_loss_history: list[dict[str, float]] = []
        self._weighted_loss_history: list[dict[str, float]] = []

        self._load_configs()

        general_info: dict[str, Any] = self.losses_config["general_info"]
        self._use_alm: bool = general_info.get("ALM") == "True"
        self._use_cse: bool = general_info.get("CSE") == "True"
        cse_params: dict[str, Any] = general_info.get("cse_params", {})
        self._cse_drag_category: str = cse_params.get("drag_category", "aero")
        self._cse_solar_category: str = cse_params.get("solar_category", "solar")
        self._cse_mass: float = float(cse_params.get("mass", 210))
        self._cse_c_losses: float = float(cse_params.get("c_losses", 20))
        self._cse_rolling_friction: float = float(cse_params.get("rolling_friction_coefficient", 0.006))
        self._cse_max_iter: int = int(cse_params.get("max_iter", 12))
        self._cse_tol: float = float(cse_params.get("tol", 1.0e-4))

        self._init_mesh_infrastructure()
        self._init_active_losses()  # must come before _init_alm_state (needs n_constraints)
        if self._use_alm:
            constraint_names: list[str] = [n for (n, _inst) in self._constraint_losses.values()]
            self.alm_manager = ALMManager(
                constraint_names,
                general_info,
                self._device,
                active_constraint_categories=list(self._constraint_losses.keys()),
                configured_constraint_categories=self._configured_constraint_categories,
            )

    def _load_configs(self) -> None:
        self._general_config: dict[str, Any] = load_config()
        self.losses_config: dict[str, Any] = load_config(
            v=self._general_config["general_info"]["version"],
            cfg_name=self._general_config["active_configs"]["losses"],
        )
        # NOTE: sweep overrides are applied automatically by the config
        # registry (see config_utils.py).  No manual wandb.config patching needed.

        wg: dict[str, Any] = self._general_config.get("wandb_general", {})
        self._use_wandb: bool = wg.get("use_wandb", True)
        self._use_renderer: bool = wg.get("use_renderer", True)
        self._use_neural_renderer: bool = wg.get("use_neural_renderer", False)
        self._rendering_freq: int | None = wg.get("rendering_frequency")

        _log_cfg: dict[str, Any] = load_config(v="v1", cfg_name="logging.yaml")
        self._memory_diagnostics: bool = _log_cfg.get("memory_diagnostics", False)
        self._mesh_statistics_enabled: bool = _log_cfg.get("mesh_statistics", False)
        self._gradient_memory_diagnostics: bool = _log_cfg.get("gradient_memory_diagnostics", False)

    def _init_mesh_infrastructure(self) -> None:
        dev_str = str(self._device)
        renderer_config: dict[str, Any] = load_config(v="v1", cfg_name="neural_renderer_v0.yaml")
        self._mesh_extraction_config: dict[str, Any] = renderer_config.get("mesh_extraction", {})

        self._mesh_extractor: MeshExtractor = MeshExtractor(
            algorithm=self._mesh_extraction_config.get("algorithm", "mc"),
            device=self._mesh_extraction_config.get("device", dev_str),
            verbose=self._mesh_extraction_config.get("verbose", False),
        )
        self._mesh_extractor.set_grid(
            bbox=self._mesh_extraction_config.get("bbox", [[-5000, 5000], [-5000, 5000], [-5000, 5000]]),
            resolution=self._mesh_extraction_config.get("resolution", 128),
        )
        # Import locally to avoid circular import during module initialization
        from subprojects.wp3_solar_estimator.validation.statistics import MeshStatistics  # noqa: PLC0415

        self._mesh_statistics = MeshStatistics(device=dev_str, use_wandb=self._use_wandb)

    def _init_active_losses(self) -> None:
        """Load every active loss and classify by YAML role metadata.

        Constraint ordering for ALM is the YAML category order.
        """
        project_root: Path = Path.cwd()
        losses_block: dict[str, Any] = self.losses_config.get("losses", {})

        # Flat ordered dict of *all* active losses (for _compute_raw_losses iteration)
        self._active_losses: dict[str, Any] = {}
        # Manual weights from config (second element of [compute, weight, path] tuple)
        self._manual_weights: dict[str, float] = {}
        # Role-classified subsets (references into _active_losses values)
        self._objective_losses: dict[str, tuple[str, Any]] = {}  # category -> (loss_name, instance)
        self._constraint_losses: dict[str, tuple[str, Any]] = {}  # category -> (loss_name, instance)
        self._configured_constraint_categories: list[str] = []
        self._objective_signs: dict[str, int] = {}  # category -> sign for manual objective mode
        self._auxiliary_losses: dict[str, Any] = {}  # name → instance

        for category, block in losses_block.items():
            if category == "total":
                continue
            if not isinstance(block, dict):
                continue
            role: str = block.get("_role", "aux")
            if role == "diversity":
                continue
            if role == "constraint":
                self._configured_constraint_categories.append(category)
            sign_val: int = -1 if block.get("_sign", "minimize") == "maximize" else 1
            for loss_name, entry in block.items():
                if loss_name.startswith("_"):
                    continue
                if not isinstance(entry, list) or len(entry) < 3:
                    continue  # skip non-loss metadata
                compute, weight, path = entry[0], entry[1], entry[2]
                if not compute:
                    continue
                abs_path: Path = project_root / Path(path)

                # Derive class name from loss_name (config key), not filename
                # This allows multiple losses to be defined in the same file (e.g., universal_null_losses.py)
                # Transform: "null_loss_solar_v1" → "NullLossSolarV1"
                family: str = loss_name.title().replace("_", "")

                loss_cls = load_class_from_path(str(abs_path), family)
                instance = loss_cls(
                    device=str(self._device),
                    loss_name=loss_name,
                    losses_config=self.losses_config,
                )
                self._active_losses[loss_name] = instance
                self._manual_weights[loss_name] = float(weight)

                # Classify by category
                if role == "objective":
                    self._objective_losses[category] = (loss_name, instance)
                    self._objective_signs[category] = sign_val
                elif role == "constraint":
                    self._constraint_losses[category] = (loss_name, instance)
                else:
                    self._auxiliary_losses[loss_name] = instance

    # ------------------------------------------------------------------
    # mesh extraction sharing
    # ------------------------------------------------------------------
    def _find_mesh_extractors(self) -> dict[str, MeshSDFExtractor]:
        """Discover MeshSDFExtractor instances on active loss objects (up to 1 level deep)."""
        extractors: dict[str, MeshSDFExtractor] = {}
        _check_attrs = ("mesh_extractor", "extractor")
        for name, instance in self._active_losses.items():
            # Direct check on loss instance
            for attr in _check_attrs:
                obj = getattr(instance, attr, None)
                if isinstance(obj, MeshSDFExtractor):
                    extractors[name] = obj
                    break
            else:
                # One level deeper: check public instance attributes for nested extractors
                for attr_name, attr_val in vars(instance).items():
                    if attr_name.startswith("_") or attr_val is None:
                        continue
                    for inner_attr in _check_attrs:
                        obj = getattr(attr_val, inner_attr, None)
                        if isinstance(obj, MeshSDFExtractor):
                            extractors[name] = obj
                            break
                    if name in extractors:
                        break
        return extractors

    def _inject_shared_meshes(self, model: TrainModel) -> None:
        """Pre-extract meshes for groups of losses that share the same extraction config.

        For each unique extraction key, one extractor performs the extraction and the
        result is injected into all other extractors in the group so they skip extraction.
        """
        extractors = self._find_mesh_extractors()
        print(f"[mesh-sharing] Found {len(extractors)} extractors: {list(extractors.keys())}")
        if len(extractors) < 2:
            return  # nothing to share

        # Group by extraction key
        groups: dict[tuple, list[tuple[str, MeshSDFExtractor]]] = defaultdict(list)
        for loss_name, ext in extractors.items():
            groups[ext.extraction_key()].append((loss_name, ext))

        for members in groups.values():
            if len(members) < 2:
                continue  # unique config, no sharing needed

            # Extract with the first extractor and inject into the rest
            primary_name, primary_ext = members[0]
            vertices, faces = primary_ext.extract_mesh(model)
            print(
                f"[mesh-sharing] Extracted {vertices.shape[0]} vertices via '{primary_name}', "
                f"sharing with {[n for n, _ in members[1:]]}"
            )
            for _name, ext in members[1:]:
                ext.set_shared_mesh(vertices, faces)

        self._shared_mesh_extractors = extractors

    def _clear_shared_meshes(self) -> None:
        """Clear all injected shared meshes after the loss computation loop."""
        for ext in getattr(self, "_shared_mesh_extractors", {}).values():
            ext.clear_shared_mesh()
        self._shared_mesh_extractors = {}

    # ------------------------------------------------------------------
    # loss computation
    # ------------------------------------------------------------------
    def _compute_raw_losses(self, model: TrainModel, epoch_name: int, skip_telemetry: bool = False) -> dict[str, torch.Tensor]:
        """Evaluate every active loss instance and return raw (unweighted) tensors."""
        self._inject_shared_meshes(model)
        raw: dict[str, torch.Tensor] = {}
        try:
            for name, instance in self._active_losses.items():
                sig = inspect.signature(instance.get_loss)
                if "skip_telemetry" in sig.parameters:
                    raw[name] = instance.get_loss(model, epoch_name, skip_telemetry=skip_telemetry)
                else:
                    raw[name] = instance.get_loss(model, epoch_name)
        finally:
            self._clear_shared_meshes()
        self.latest_raw_metrics = raw
        return raw

    def _extract_assembly_id_from_instance(self, instance: Any) -> int | None:
        """Extract assembly id from loss instance (supports public and mangled/private fields)."""
        candidates: list[Any] = [
            getattr(instance, "assembly_id", None),
            getattr(instance, "_assembly_id", None),
        ]

        for attr_name, attr_value in vars(instance).items():
            if attr_name.lower().endswith("assembly_id"):
                candidates.append(attr_value)

        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return int(candidate)
            except (TypeError, ValueError):
                continue

        return None

    def _active_assembly_id(self) -> int | None:
        """Best-effort extraction of runtime BB assembly id from active loss instances."""
        bb_entry = self._constraint_losses.get("bb")
        if bb_entry is not None:
            _bb_name, bb_instance = bb_entry
            bb_assembly_id = self._extract_assembly_id_from_instance(bb_instance)
            if bb_assembly_id is not None:
                return bb_assembly_id

        for instance in self._active_losses.values():
            runtime_assembly_id = self._extract_assembly_id_from_instance(instance)
            if runtime_assembly_id is not None:
                return runtime_assembly_id

        return None

    def _apply_weighting(
        self,
        raw_losses: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Weight objectives and constraints independently then sum.

        Objective weighting (``self._use_cse``):
          * **CSE**: combine aero + solar via cruising-speed estimator (with
            per-objective prefactors from config).
          * **Manual**: multiply each objective loss by its config weight.

        Constraint weighting (``self._use_alm``):
          * **ALM**: adaptive lambda*c + 0.5*mu*c^2 penalty per constraint.
          * **Manual**: multiply each constraint loss by its config weight.

        Auxiliary losses are always manual-weighted.

        Populates ``self._last_weighted_losses`` with independent-graph tensors
        suitable for per-component ``.backward(retain_graph=False)``.
        """
        gi: dict[str, Any] = self.losses_config["general_info"]
        self._last_weighted_losses = {}

        # ---- Objectives --------------------------------------------------
        if self._use_cse:
            prefactors: dict[str, float] = gi.get("cse_prefactors", {})
            if self._cse_drag_category not in self._objective_losses:
                raise KeyError(
                    f"CSE enabled but drag objective category '{self._cse_drag_category}' is not active. "
                    f"Active objective categories: {list(self._objective_losses.keys())}"
                )
            if self._cse_solar_category not in self._objective_losses:
                raise KeyError(
                    f"CSE enabled but solar objective category '{self._cse_solar_category}' is not active. "
                    f"Active objective categories: {list(self._objective_losses.keys())}"
                )

            drag_name: str = self._objective_losses[self._cse_drag_category][0]
            solar_name: str = self._objective_losses[self._cse_solar_category][0]
            drag_pf: float = float(prefactors.get("drag", 1.0))
            solar_pf: float = float(prefactors.get("solar", 1.0))
            drag: torch.Tensor = raw_losses[drag_name] * drag_pf
            solar: torch.Tensor = raw_losses[solar_name] * solar_pf

            v_cruise: torch.Tensor = cruising_speed(
                solar_power=solar,
                drag_force_at_25ms=drag,
                mass=self._cse_mass,
                c_losses=self._cse_c_losses,
                rolling_friction_coefficient=self._cse_rolling_friction,
                max_iter=self._cse_max_iter,
                tol=self._cse_tol,
            )
            self._last_cruising_speed = v_cruise
            self._last_cruising_speed_diagnostics = {
                "cruising_speed_ms": v_cruise.item(),
            }
            self._last_weighted_losses["cse_objective"] = -v_cruise
        else:
            self._last_cruising_speed = None
            self._last_cruising_speed_diagnostics = None
            for cat, (name, _instance) in self._objective_losses.items():
                w: float = self._manual_weights.get(name, 1.0)
                sign: float = float(self._objective_signs.get(cat, 1))
                self._last_weighted_losses[name] = sign * raw_losses[name] * w

        # ---- Constraints -------------------------------------------------
        constraint_names: list[str] = [n for (n, _inst) in self._constraint_losses.values()]
        constraints: list[torch.Tensor] = [raw_losses[n] for n in constraint_names]

        if self._use_alm:
            if not self._external_alm and self.alm_manager is not None:
                self.alm_manager.update_multipliers(constraints)
                alm_penalties = self.alm_manager.compute_penalty(constraints)
                self._last_weighted_losses.update(alm_penalties)
        else:
            for cname, raw_t in zip(constraint_names, constraints, strict=True):
                w_c: float = self._manual_weights.get(cname, 1.0)
                self._last_weighted_losses[cname] = raw_t * w_c

        # ---- Auxiliary (always manual-weighted) --------------------------
        for aux_name in self._auxiliary_losses:
            w_a: float = self._manual_weights.get(aux_name, 1.0)
            self._last_weighted_losses[aux_name] = raw_losses[aux_name] * w_a

        weighted_total: torch.Tensor = sum(self._last_weighted_losses.values())  # type: ignore[assignment]
        return weighted_total

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    def _handle_telemetry(  # noqa: PLR0912
        self,
        loss_log: dict[str, float],
        epoch_name: int,
        device: torch.device,
        model: TrainModel,
        total_loss: torch.Tensor,
        skip_gradients: bool = False,
    ) -> None:
        """WandB logging, mesh rendering, gradient logging, and memory diagnostics."""
        if not self._use_wandb:
            if not skip_gradients:
                # Still log gradients (they print to stdout when wandb is off)
                for inst in self._active_losses.values():
                    inst.log_gradients(epoch_name)
            self._log_memory_diagnostics(epoch_name, device)
            return

        # --- Build ONE log dict for all scalar metrics, then call wandb.log ONCE.
        # Multiple wandb.log(step=N) calls at the same step can prematurely
        # commit the step and corrupt wandb's internal step counter, causing
        # later media uploads (GIFs from finalize) to land at wrong steps.
        log_dict: dict[str, float] = {}

        # unweighted losses + total loss
        log_dict.update(loss_log)

        # individual weighted contributions
        for name, val in self._last_weighted_losses.items():
            # Negate cse_objective back to positive for physical meaning in logs
            # (we minimize -v_cruise internally, but log +v_cruise for readability)
            logged_val = (
                -val.item() if name == "cse_objective" else val.item()
            )  # the CSE is logged negatively, as we have a minimization scheme
            log_dict[f"weighted_losses/{name}"] = logged_val

        # cruising speed
        if self._last_cruising_speed is not None:
            log_dict["cruising_speed_ms"] = self._last_cruising_speed.item()

        # ALM multipliers (only in ALM mode)
        if self._use_alm and self.alm_manager is not None:
            log_dict.update(self.alm_manager.get_telemetry())

        wandb.log(log_dict, step=epoch_name)  # ein riesen log

        # --- Loss competition charts -------------------------------------
        # raw_floats: strip prefix, exclude total, take abs (handles negative cse_objective)
        raw_floats: dict[str, float] = {
            k.removeprefix("raw_losses/"): abs(v) for k, v in loss_log.items() if k != "raw_losses/total_loss"
        }
        # weighted_floats: abs of each weighted component.
        # When CSE is active, _last_weighted_losses contains cse_objective instead of
        # drag/solar, so the weighted charts naturally show cruising-speed competition.
        weighted_floats: dict[str, float] = {k: abs(v.item()) for k, v in self._last_weighted_losses.items()}
        self._log_loss_competition_charts(raw_floats, weighted_floats, epoch_name)

        # --- Mesh extraction & rendering ----------------------------------
        if self._use_renderer and self._renderer_manager is not None:
            verts, faces = self._mesh_extractor.extract_mesh(
                model=model,
                isovalue=self._mesh_extraction_config.get("isovalue", 0.0),
                batch_size=self._mesh_extraction_config.get("batch_size", 8192),
            )
            if self._mesh_statistics_enabled:
                self._mesh_statistics.log_basic_mesh_stats(verts, faces, epoch_name)
                self._mesh_statistics.log_solar_area_stats(verts, faces, epoch_name)

            if self._rendering_freq is not None and epoch_name % self._rendering_freq == 0:
                if self._use_neural_renderer and self._renderer_manager is not None:
                    try:
                        runtime_assembly_id = self._active_assembly_id()
                        self._renderer_manager.update(
                            {
                                "model": model,
                                "iteration": epoch_name,
                                "loss": total_loss.item(),
                                "vertices": verts,
                                "faces": faces,
                                "assembly_id": runtime_assembly_id,
                                "bb_assembly_id": runtime_assembly_id,
                            }
                        )
                    except Exception as e:
                        print(f"Renderer wrapper update failed: {e}")
                else:
                    self._run_cad_vis(verts, faces, epoch_name)
        else:
            print(f"Skipping rendering at epoch {epoch_name} due to rendering frequency setting.")

        # --- Gradient logging for each loss component ---------------------
        if not skip_gradients:
            for inst in self._active_losses.values():
                inst.log_gradients(epoch_name)

        # --- GPU memory diagnostics ---------------------------------------
        self._log_memory_diagnostics(epoch_name, device)

    def _run_cad_vis(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        epoch_name: int,
    ) -> None:
        """Render mesh via CADVisualizer, log to WandB, and clean up local files."""
        verts_np = verts.detach().cpu().numpy() if torch.is_tensor(verts) else verts
        faces_np = faces.detach().cpu().numpy() if torch.is_tensor(faces) else faces

        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
        stl_path = Path(f"subprojects/wp4_aeroshell_optimizer/v2/logs/mesh_epoch_{epoch_name}.stl")
        stl_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(stl_path))

        image_path = Path(f"subprojects/wp4_aeroshell_optimizer/v2/logs/render_epoch_{epoch_name}.png")
        visualizer = CADVisualizer(stl_path=str(stl_path), out_image_path=str(image_path))
        visualizer.render_until_blue_detected()

        if stl_path.exists():
            wandb.log({"mesh_3d": wandb.Object3D(str(stl_path))}, step=epoch_name)
            time.sleep(2)
            stl_path.unlink()
        if image_path.exists():
            wandb.log({"render_image": wandb.Image(str(image_path))}, step=epoch_name, commit=True)
            time.sleep(2)
            image_path.unlink()

    def _log_memory_diagnostics(self, epoch_name: int, device: torch.device) -> None:
        """Print and optionally log GPU memory stats (toggled by logging.yaml)."""
        if not self._memory_diagnostics:
            return
        if device.type != "cuda":
            return
        allocated_mb: float = torch.cuda.memory_allocated(device) / 1024**2
        reserved_mb: float = torch.cuda.memory_reserved(device) / 1024**2
        peak_mb: float = torch.cuda.max_memory_allocated(device) / 1024**2
        free_cached_mb: float = reserved_mb - allocated_mb
        print(
            f"[Epoch {epoch_name}] GPU Memory | "
            f"Allocated: {allocated_mb:.1f} MB | "
            f"Reserved: {reserved_mb:.1f} MB | "
            f"Peak: {peak_mb:.1f} MB | "
            f"Free (cached): {free_cached_mb:.1f} MB"
        )
        if self._use_wandb:
            wandb.log(
                {
                    "memory_stats/allocated_mb": allocated_mb,
                    "memory_stats/reserved_mb": reserved_mb,
                    "memory_stats/peak_allocated_mb": peak_mb,
                    "memory_stats/free_cached_mb": free_cached_mb,
                },
                step=epoch_name,
            )

    # ------------------------------------------------------------------
    # Loss competition charts
    # ------------------------------------------------------------------
    def _log_loss_competition_charts(  # noqa: PLR0915
        self,
        raw_floats: dict[str, float],
        weighted_floats: dict[str, float],
        epoch_name: int,
    ) -> None:
        """Generate 6 loss-competition visualizations and log to WandB under loss_competition/.

        (a) raw_pie          - pie chart of raw loss shares
        (b) weighted_pie     - pie chart of weighted loss shares
        (c) raw_bar          - normalized horizontal stacked bar, raw
        (d) weighted_bar     - normalized horizontal stacked bar, weighted
        (e) radar            - polar overlay of raw (dashed) vs weighted (solid), both normalized 0-1
        (f) stacked_area     - stacked area over training time (two subplots)

        When CSE is active, ``weighted_floats`` already contains ``cse_objective``
        in place of individual drag/solar entries, so the weighted charts naturally
        reflect cruising-speed competition without special-casing here.
        """
        matplotlib.use("Agg")  # headless cluster - no display server
        # Append to rolling history (feeds stacked-area chart)
        self._raw_loss_history.append(dict(raw_floats))
        self._weighted_loss_history.append(dict(weighted_floats))

        competition_log: dict = {}

        # ----------------------------------------------------------------
        # helpers
        # ----------------------------------------------------------------
        def _safe_pie(values: list[float], labels: list[str], title: str) -> plt.Figure:
            fig, ax = plt.subplots(figsize=(6, 6))
            # Filter non-finite and non-positive values (pie can't handle NaN/Inf/neg).
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
        w_labels = list(weighted_floats.keys())
        w_vals = [weighted_floats[k] for k in w_labels]

        # --- (a) Raw pie -------------------------------------------------
        fig_a = _safe_pie(raw_vals, raw_labels, f"Raw Losses (epoch {epoch_name})")
        competition_log["loss_competition/raw_pie"] = wandb.Image(fig_a)
        plt.close(fig_a)

        # --- (b) Weighted pie --------------------------------------------
        fig_b = _safe_pie(w_vals, w_labels, f"Weighted Losses (epoch {epoch_name})")
        competition_log["loss_competition/weighted_pie"] = wandb.Image(fig_b)
        plt.close(fig_b)

        # --- (c) Raw normalized horizontal bar ---------------------------
        fig_c = _safe_hbar(raw_vals, raw_labels, f"Raw Loss Shares % (epoch {epoch_name})")
        competition_log["loss_competition/raw_bar"] = wandb.Image(fig_c)
        plt.close(fig_c)

        # --- (d) Weighted normalized horizontal bar ----------------------
        fig_d = _safe_hbar(w_vals, w_labels, f"Weighted Loss Shares % (epoch {epoch_name})")
        competition_log["loss_competition/weighted_bar"] = wandb.Image(fig_d)
        plt.close(fig_d)

        # --- (e) Radar / spider chart ------------------------------------
        # Union of all loss keys; missing entries get 0 so both series share axes.
        all_keys = list(dict.fromkeys(list(raw_floats.keys()) + list(weighted_floats.keys())))
        n = len(all_keys)
        if n >= 2:
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
            angles += angles[:1]  # close the polygon

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
            ax_e.set_title(f"Loss Radar (epoch {epoch_name})", pad=15)
            ax_e.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            fig_e.tight_layout()
            competition_log["loss_competition/radar"] = wandb.Image(fig_e)
            plt.close(fig_e)

        # --- (f) Stacked area over training time -------------------------
        if len(self._raw_loss_history) >= 2:
            epochs_so_far = list(range(len(self._raw_loss_history)))

            def _stacked_area(history: list[dict[str, float]], ax: plt.Axes, title: str) -> None:
                all_k = list(dict.fromkeys(k for d in history for k in d))
                matrix = np.array([[d.get(k, 0.0) for k in all_k] for d in history])  # [T, K]
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

        wandb.log(competition_log, step=epoch_name)

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def get_cruising_speed_diagnostics(self) -> dict | None:
        """Return diagnostics from the last cruising speed solver call, or None if CSE is inactive."""
        return self._last_cruising_speed_diagnostics

    def get_total_loss(
        self, model: TrainModel, epoch_name: int = -1, skip_telemetry: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        """Initilizes unweighted losses computation, computes weighted total loss, logs to wandb, and returns the total loss tensor.

        If self._external_alm is True, returns a tuple of (weighted_objectives, raw_constraints_dict) so the caller
        can manually manage multiplier states across populations.

        After this call (for external_alm=False), ``self._last_weighted_losses`` contains per-component
        tensors with independent computation graphs suitable for individual
        ``.backward(retain_graph=False)`` calls.
        """
        device: torch.device = next(iter(model.parameters())).device

        # 1. Evaluate all active loss instances
        raw: dict[str, torch.Tensor] = self._compute_raw_losses(model, epoch_name, skip_telemetry=skip_telemetry)

        # 2. Apply weighting (populates _last_weighted_losses)
        total: torch.Tensor = self._apply_weighting(raw)
        self.latest_weighted_metrics = self._last_weighted_losses

        # 3. Build logging dict and dispatch telemetry
        if not skip_telemetry:
            loss_log: dict[str, float] = {f"raw_losses/{name}": t.item() for name, t in raw.items()}
            loss_log["raw_losses/total_loss"] = total.item()
            self._handle_telemetry(loss_log, epoch_name, device, model, total)

        if self._external_alm:
            constraint_names: list[str] = [n for (n, _inst) in self._constraint_losses.values()]
            raw_constraints_dict = {name: raw[name] for name in constraint_names}
            return total, raw_constraints_dict

        return total
