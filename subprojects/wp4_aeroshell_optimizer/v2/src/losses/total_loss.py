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
from torch.profiler import record_function

import wandb
from subprojects.wp3_solar_estimator.mesh_extractors.non_diff_algos.marching_cubes import (
    MeshExtractor,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.alm import ALMManager
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.cruising_speed import cruising_speed
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.loss_normalisation import LossNormalisationManager
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsModelContract
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import (
    NullVisManager,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import (
    CentralVisManagerV2,
)
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
        vis_manager: CentralVisManagerV2 | NullVisManager | None = None,
    ) -> None:
        self._device: torch.device = torch.device(dev) if isinstance(dev, str) else dev
        self._renderer_manager: VisualisationManager | None = renderer_manager
        self._external_alm: bool = external_alm
        self._vis_manager: CentralVisManagerV2 | NullVisManager | None = vis_manager

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
        self._normalized_loss_history: list[dict[str, float]] = []
        self._weighted_loss_history: list[dict[str, float]] = []
        self._optimizer_loss_history: list[dict[str, float]] = []

        self._load_configs()

        general_info: dict[str, Any] = self.losses_config["general_info"]
        self._use_alm: bool = general_info.get("ALM") == "True"
        self._use_cse: bool = general_info.get("CSE") == "True"
        self._deactivate_scheduler_mode: str = general_info["deactivate_scheduler_mode"]
        cse_params: dict[str, Any] = general_info.get("cse_params", {})
        self._cse_drag_category: str = cse_params.get("drag_category", "aero")
        self._cse_solar_category: str = cse_params.get("solar_category", "solar")
        self._cse_mass: float = float(cse_params.get("mass", 210))
        self._cse_c_losses: float = float(cse_params.get("c_losses", 20))
        self._cse_rolling_friction: float = float(cse_params.get("rolling_friction_coefficient", 0.006))
        self._cse_max_iter: int = int(cse_params.get("max_iter", 12))
        self._cse_tol: float = float(cse_params.get("tol", 1.0e-4))
        # Last-known raw (pre-prefactor) value per CSE objective category, used when that category
        # is scheduled out so the other objective can still run CSE with a meaningful partner value.
        self._cse_objective_cache: dict[str, torch.Tensor] = {}
        self._cse_weight: float = float(general_info.get("cse_weight", 1.0))

        # Visualisation state — populated each _compute_raw_losses() by
        # _extract_primary_visu_mesh + _capture_vertex_grads_via_autograd. Read by
        # the renderer via collect_vertex_grads_for_visualisation(). Mirrors V2.
        self._visu_primary_v: torch.Tensor | None = None
        self._visu_primary_f: torch.Tensor | None = None
        self._visu_per_loss_grads: dict[str, torch.Tensor] = {}

        self._init_mesh_infrastructure()
        self._init_active_losses()  # must come before _init_alm_state (needs n_constraints)
        # Propagate vis_manager to child loss instances (e.g. GaotLoss) so they can
        # route their own visualizations through CentralVisManager.
        if self._vis_manager is not None:
            for _loss_inst in self._active_losses.values():
                # add CVM to all losses
                _loss_inst._vis_manager = self._vis_manager
        if self._use_alm:
            constraint_names: list[str] = [n for (n, _inst) in self._constraint_losses.values()]
            self.alm_manager = ALMManager(
                constraint_names,
                general_info,
                self._device,
                active_constraint_categories=list(self._constraint_losses.keys()),
                configured_constraint_categories=self._configured_constraint_categories,
            )

        norm_cfg = self.losses_config.get("general_info", {}).get("loss_normalisation", {})
        self._norm_manager = LossNormalisationManager(norm_cfg)

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
        # category -> list of (start, end) inclusive epoch ranges where the loss is DISABLED
        self._disabled_epochs: dict[str, list[tuple[int, int]]] = {}
        # loss_name -> category (reverse map for scheduling in _compute_raw_losses)
        self._loss_name_to_category: dict[str, str] = {}

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

                self._loss_name_to_category[loss_name] = category

            disabled_raw = block.get("_disabled_epochs", [])
            self._disabled_epochs[category] = [(int(s), int(e)) for s, e in disabled_raw]

    def _warn_about_dead_weights(self) -> None:
        """Warn users about config weights that will be ignored due to ALM or CSE modes."""
        gi: dict[str, Any] = self.losses_config["general_info"]
        use_cse: bool = gi.get("CSE") == "True"
        use_alm: bool = gi.get("ALM") == "True"

        if use_cse:
            dead_objectives = [
                name
                for cat, (name, _) in self._objective_losses.items()
                if name in self._manual_weights and self._manual_weights[name] != 1.0
            ]
            if dead_objectives:
                print(
                    f"⚠️  WARNING: CSE is enabled (CSE=True). Individual objective weights for {dead_objectives} "
                    f"are dead code and will be ignored. Use 'cse_weight' config entry to scale CSE vs diversity."
                )

        if use_alm:
            dead_constraints = [
                name
                for cat, (name, _) in self._constraint_losses.items()
                if name in self._manual_weights and self._manual_weights[name] != 1.0
            ]
            if dead_constraints:
                print(
                    f"⚠️  WARNING: ALM is enabled (ALM=True). Constraint weights for {dead_constraints} "
                    f"are dead code and will be ignored. Use 'initial_lambda' and 'initial_mu' config entries to influence constraint strength."
                )

    def _is_category_active(self, category: str, epoch: int) -> bool:
        return all(not (start <= epoch and (end == -1 or epoch <= end)) for start, end in self._disabled_epochs.get(category, []))

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
    # per-loss vertex-gradient visualisation (mirrors TotalLossV2)
    # ------------------------------------------------------------------
    def _extract_primary_visu_mesh(self, model: TrainModel) -> None:
        """Extract one primary mesh for the visualiser via any available extractor.

        Reuses the result of `_inject_shared_meshes` when it ran (≥ 2 extractors).
        Otherwise calls `extract_mesh` on whichever extractor is registered and
        injects the result back into that extractor with `set_shared_mesh`, so the
        single mesh-using loss reuses the exact same `(verts, faces)` tensors —
        which means `autograd.grad(L_k, verts)` is well-defined for that loss.
        """
        self._visu_primary_v = None
        self._visu_primary_f = None
        extractors = self._find_mesh_extractors()
        if not extractors:
            return
        # If _inject_shared_meshes already extracted a primary, reuse it.
        for ext in extractors.values():
            shared = getattr(ext, "_shared_vertices", None)
            shared_f = getattr(ext, "_shared_faces", None)
            if shared is not None and shared_f is not None:
                self._visu_primary_v = shared
                self._visu_primary_f = shared_f
                return
        # Otherwise extract once with the first extractor and pin it.
        first_ext = next(iter(extractors.values()))
        verts, faces = first_ext.extract_mesh(model)
        first_ext.set_shared_mesh(verts, faces)
        self._shared_mesh_extractors = extractors  # ensure clear_shared_mesh() runs.
        self._visu_primary_v = verts
        self._visu_primary_f = faces

    def _capture_vertex_grads_via_autograd(self, raw_losses: dict[str, torch.Tensor]) -> None:
        """Collect dL/dv for every loss whose extractor produced one this step.

        `MeshSDFExtractor.extract_and_compute_loss` already runs
        `autograd.grad(loss, v_for_loss)` internally and stashes the result on
        the extractor as `last_dL_dv` (vertices match the shared primary mesh).
        We just harvest those tensors here, keyed by loss name. Losses that
        don't go through `extract_and_compute_loss` are absent from the result.

        The `raw_losses` argument is kept for API compatibility but unused.
        """
        del raw_losses
        self._visu_per_loss_grads = {}
        self._visu_per_loss_meshes: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for loss_name, ext in self._find_mesh_extractors().items():
            g = getattr(ext, "last_dL_dv", None)
            if g is None:
                continue
            self._visu_per_loss_grads[loss_name] = g.detach()
            v = getattr(ext, "last_verts", None)
            f = getattr(ext, "last_faces", None)
            if v is not None and f is not None:
                self._visu_per_loss_meshes[loss_name] = (v.detach(), f.detach())

    def collect_vertex_grads_for_visualisation(
        self,
        weights_by_name: dict[str, float] | None = None,
    ) -> dict[str, Any] | None:
        """Return per-loss dL/dv captured during the last `_compute_raw_losses`.

        Returns None if no loss had an autograd path to the primary mesh. Otherwise:
            {
              "vertices":         (V, 3) primary mesh vertices,
              "faces":            (F, 3) primary mesh faces,
              "per_loss":         {loss_name: (V, 3) dL/dv},
              "per_loss_meshes":  {loss_name: (verts, faces)} mesh for each loss,
            }

        The previous weighted-sum (`"total"`) entry was removed: it mixed losses
        defined on meshes with different resolutions and bbox extents, so the
        combined per-vertex gradient was meaningless.
        """
        if not getattr(self, "_visu_diag_logged", False):
            self._visu_diag_logged = True
            v = self._visu_primary_v
            v_state = "None" if v is None else f"shape={tuple(v.shape)} requires_grad={v.requires_grad} grad_fn={v.grad_fn is not None}"
            print(
                f"[VisuDiag V1] primary_mesh={v_state} "
                f"per_loss_grads_keys={list(self._visu_per_loss_grads.keys())}"
            )
        if not self._visu_per_loss_grads or self._visu_primary_v is None:
            return None
        del weights_by_name  # kept on the signature for backward compatibility
        return {
            "vertices": self._visu_primary_v,
            "faces": self._visu_primary_f,
            "per_loss": dict(self._visu_per_loss_grads),
            "per_loss_meshes": dict(getattr(self, "_visu_per_loss_meshes", {})),
        }


    # ------------------------------------------------------------------
    # loss computation
    # ------------------------------------------------------------------
    @record_function("TotalLossV1._compute_raw_losses")
    def _compute_raw_losses(self, model: TrainModel, epoch_name: int, calibration_mode: bool = False) -> dict[str, torch.Tensor]:
        """Evaluate every active loss instance and return raw (unweighted) tensors.

        Losses in a disabled epoch range are handled according to self._deactivate_scheduler_mode:
          "skip": not computed; tracked in self._skipped_categories for 0-logging.
          "zero_weighted": computed and included in raw; tracked in self._zero_weighted_categories.

        Args:
            calibration_mode: If True, bypass epoch scheduling and compute all losses regardless
                of _disabled_epochs. Used by _calibrate() to reach every loss before training starts.
        """
        self._skipped_categories: set[str] = set()
        self._zero_weighted_categories: set[str] = set()
        self._inject_shared_meshes(model)
        self._extract_primary_visu_mesh(model)
        raw: dict[str, torch.Tensor] = {}
        try:
            for name, instance in self._active_losses.items():
                category = self._loss_name_to_category.get(name, "")
                if not calibration_mode and not self._is_category_active(category, epoch_name):
                    if self._deactivate_scheduler_mode == "zero_weighted":
                        self._zero_weighted_categories.add(category)
                        sig = inspect.signature(instance.get_loss)
                        if "skip_telemetry" in sig.parameters:
                            raw[name] = instance.get_loss(model, epoch_name, skip_telemetry=True)
                        else:
                            raw[name] = instance.get_loss(model, epoch_name)
                    else:
                        self._skipped_categories.add(category)
                    continue
                sig = inspect.signature(instance.get_loss)
                if "skip_telemetry" in sig.parameters:
                    raw[name] = instance.get_loss(model, epoch_name, skip_telemetry=True)
                else:
                    raw[name] = instance.get_loss(model, epoch_name)
            self._capture_vertex_grads_via_autograd(raw)
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

    @record_function("TotalLossV1._apply_weighting")
    def _apply_weighting(  # noqa: PLR0912, PLR0915
        self,
        raw_losses: dict[str, torch.Tensor],
        calibration_pass: bool = False,
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
        skipped_cats: set[str] = getattr(self, "_skipped_categories", set())
        zero_weighted_cats: set[str] = getattr(self, "_zero_weighted_categories", set())

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
            inactive_obj_cats: set[str] = skipped_cats | zero_weighted_cats

            # Update cache for objectives that were computed this epoch (live or zero_weighted).
            # Zero_weighted: raw value exists in raw_losses but should not contribute gradient through CSE.
            if self._cse_drag_category not in skipped_cats:
                self._cse_objective_cache[self._cse_drag_category] = raw_losses[drag_name].detach().clone()
            if self._cse_solar_category not in skipped_cats:
                self._cse_objective_cache[self._cse_solar_category] = raw_losses[solar_name].detach().clone()

            # Resolve the tensor each CSE input will use.
            # Active: live tensor (gradient flows). Inactive: cached detached value (no gradient).
            # Cold-start fallback (cache empty, already off at epoch 0): zeros, like a weight-0 null loss.
            drag_is_live: bool = self._cse_drag_category not in inactive_obj_cats
            solar_is_live: bool = self._cse_solar_category not in inactive_obj_cats
            drag_raw: torch.Tensor = (
                raw_losses[drag_name]
                if drag_is_live
                else self._cse_objective_cache.get(self._cse_drag_category, torch.zeros(1, device=self._device))
            )
            solar_raw: torch.Tensor = (
                raw_losses[solar_name]
                if solar_is_live
                else self._cse_objective_cache.get(self._cse_solar_category, torch.zeros(1, device=self._device))
            )

            v_cruise: torch.Tensor = cruising_speed(
                solar_power=solar_raw * solar_pf,
                drag_force_at_25ms=drag_raw * drag_pf,
                mass=self._cse_mass,
                c_losses=self._cse_c_losses,
                rolling_friction_coefficient=self._cse_rolling_friction,
                max_iter=self._cse_max_iter,
                tol=self._cse_tol,
            )
            self._last_cruising_speed = v_cruise
            self._last_cruising_speed_diagnostics = {"cruising_speed_ms": v_cruise.item()}
            # Only backprop if at least one input is live; both-cached → v_cruise has no grad.
            if drag_is_live or solar_is_live:
                cse_scale = self._norm_manager.get_loss_scale("cse_objective") * self._norm_manager.get_grad_scale(
                    "cse_objective"
                )
                cse_weight = 1.0 if calibration_pass else self._cse_weight
                self._last_weighted_losses["cse_objective"] = -v_cruise * cse_scale * cse_weight
        else:
            self._last_cruising_speed = None
            self._last_cruising_speed_diagnostics = None
            for cat, (name, _instance) in self._objective_losses.items():
                if cat in skipped_cats:
                    continue
                if cat in zero_weighted_cats:
                    self._last_weighted_losses[name] = torch.zeros((), device=self._device)
                    continue
                w: float = 1.0 if calibration_pass else self._manual_weights.get(name, 1.0)
                sign: float = float(self._objective_signs.get(cat, 1))
                combined_scale = self._norm_manager.get_loss_scale(name) * self._norm_manager.get_grad_scale(name)
                self._last_weighted_losses[name] = sign * raw_losses[name] * w * combined_scale

        # ---- Constraints -------------------------------------------------
        # Inactive (skipped or zero_weighted) categories are excluded from ALM updates
        # so their multipliers are frozen while the loss is disabled.
        inactive_cats: set[str] = getattr(self, "_skipped_categories", set()) | getattr(self, "_zero_weighted_categories", set())
        all_constraint_cats: list[str] = list(self._constraint_losses.keys())
        active_constraint_cats: list[str] = [cat for cat in all_constraint_cats if cat not in inactive_cats]
        active_constraint_indices: list[int] = [i for i, cat in enumerate(all_constraint_cats) if cat not in inactive_cats]
        # Pre-scale each active constraint tensor by loss_scale so ALM operates in normalised space.
        # grad_scale is applied to the resulting penalty below to keep it consistent with v2 semantics.
        active_constraint_tensors: list[torch.Tensor] = [
            raw_losses[self._constraint_losses[cat][0]] * self._norm_manager.get_loss_scale(self._constraint_losses[cat][0])
            for cat in active_constraint_cats
        ]

        if self._use_alm:
            if not self._external_alm and self.alm_manager is not None:
                # Compute penalty with current-epoch multipliers first (for gradient), then update for next epoch.
                alm_penalties = self.alm_manager.compute_penalty(
                    active_constraint_tensors, active_indices=active_constraint_indices
                )
                self.alm_manager.update_multipliers(active_constraint_tensors, active_indices=active_constraint_indices)
                for key, t in alm_penalties.items():
                    # key = "alm_{constraint_name}"; grad_scale is keyed the same way
                    grad_scale = self._norm_manager.get_grad_scale(key)
                    self._last_weighted_losses[key] = t * grad_scale
                # zero_weighted constraints: overwrite with 0 so no gradient flows
                for cat in getattr(self, "_zero_weighted_categories", set()):
                    if cat in self._constraint_losses:
                        cname = self._constraint_losses[cat][0]
                        self._last_weighted_losses[f"alm_{cname}"] = torch.zeros(
                            1, device=str(self._device), requires_grad=False
                        ).squeeze()
        else:
            for cat, (cname, _) in self._constraint_losses.items():
                if cat in getattr(self, "_skipped_categories", set()):
                    continue  # skip entirely: no gradient, no entry in weighted losses
                w_c: float = 1.0 if calibration_pass else self._manual_weights.get(cname, 1.0)
                if cat in getattr(self, "_zero_weighted_categories", set()):
                    self._last_weighted_losses[cname] = torch.zeros((), device=self._device)
                    continue
                combined_scale = self._norm_manager.get_loss_scale(cname) * self._norm_manager.get_grad_scale(cname)
                self._last_weighted_losses[cname] = raw_losses[cname] * w_c * combined_scale

        # ---- Auxiliary (always manual-weighted) --------------------------
        for aux_name in self._auxiliary_losses:
            aux_cat: str = self._loss_name_to_category.get(aux_name, "")
            if aux_cat in getattr(self, "_skipped_categories", set()):
                continue
            if aux_cat in getattr(self, "_zero_weighted_categories", set()):
                self._last_weighted_losses[aux_name] = torch.zeros((), device=self._device)
                continue
            w_a: float = 1.0 if calibration_pass else self._manual_weights.get(aux_name, 1.0)
            combined_scale = self._norm_manager.get_loss_scale(aux_name) * self._norm_manager.get_grad_scale(aux_name)
            self._last_weighted_losses[aux_name] = raw_losses[aux_name] * w_a * combined_scale

        weighted_total: torch.Tensor = sum(self._last_weighted_losses.values())  # type: ignore[assignment]
        return weighted_total

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    def _handle_telemetry(  # noqa: PLR0912, PLR0915
        self,
        loss_log: dict[str, float],
        epoch_name: int,
        device: torch.device,
        model: TrainModel,
        total_loss: torch.Tensor,
    ) -> None:
        """WandB logging, mesh rendering, and memory diagnostics."""
        if not self._use_wandb:
            self._print_memory_diagnostics(epoch_name, device)
            return

        # --- Build ONE log dict for all scalar metrics, then call wandb.log ONCE.
        # Multiple wandb.log(step=N) calls at the same step can prematurely
        # commit the step and corrupt wandb's internal step counter, causing
        # later media uploads (GIFs from finalize) to land at wrong steps.
        log_dict: dict[str, float] = {}

        # unweighted losses + total loss
        log_dict.update(loss_log)

        # normalized_losses: raw * loss_scale only (no grad_scale, no manual weight)
        for name, val in loss_log.items():
            if not name.startswith("raw_losses/") or name == "raw_losses/total_loss":
                continue
            loss_name = name.removeprefix("raw_losses/")
            log_dict[f"normalized_losses/{loss_name}"] = val * self._norm_manager.get_loss_scale(loss_name)
        if self._use_cse and self._last_cruising_speed is not None:
            cse_loss_scale = self._norm_manager.get_loss_scale("cse_objective")
            log_dict["normalized_losses/cse_objective"] = self._last_cruising_speed.item() * cse_loss_scale

        # weighted_losses: legacy compat — positive values for readability
        for name, tensor_val in self._last_weighted_losses.items():
            logged_val = -tensor_val.item() if name == "cse_objective" else tensor_val.item()
            log_dict[f"weighted_losses/{name}"] = logged_val

        # optimizer_losses: exact signed value entering the optimizer sum
        for name, tensor_val in self._last_weighted_losses.items():
            log_dict[f"optimizer_losses/{name}"] = tensor_val.item()
        obj_sign_legacy: dict[str, float] = {
            self._objective_losses[cat][0]: float(self._objective_signs.get(cat, 1)) for cat in self._objective_losses
        }
        if self._use_cse:
            obj_sign_legacy["cse_objective"] = -1.0
        log_dict["normalized_losses/total_loss"] = sum(
            obj_sign_legacy.get(k.removeprefix("normalized_losses/"), 1.0) * v
            for k, v in log_dict.items()
            if k.startswith("normalized_losses/") and k != "normalized_losses/total_loss"
        )
        log_dict["optimizer_losses/total_loss"] = total_loss.item()

        # cruising speed
        if self._last_cruising_speed is not None:
            log_dict["cruising_speed_ms"] = self._last_cruising_speed.item()

        # ALM multipliers + normalization scale constants
        if self._use_alm and self.alm_manager is not None:
            log_dict.update(self.alm_manager.get_telemetry())
        log_dict.update(self._norm_manager.get_telemetry())

        # --- Loss competition charts -------------------------------------
        raw_floats: dict[str, float] = {
            k.removeprefix("raw_losses/"): abs(v) for k, v in loss_log.items() if k != "raw_losses/total_loss"
        }
        normalized_floats: dict[str, float] = {
            k.removeprefix("normalized_losses/"): abs(v)
            for k, v in log_dict.items()
            if k.startswith("normalized_losses/") and k != "normalized_losses/total_loss"
        }
        weighted_floats: dict[str, float] = {
            k: abs(v.item()) for k, v in self._last_weighted_losses.items() if k != "weighted_losses/total_loss"
        }
        optimizer_floats: dict[str, float] = {
            k.removeprefix("optimizer_losses/"): abs(v)
            for k, v in log_dict.items()
            if k.startswith("optimizer_losses/") and k != "optimizer_losses/total_loss"
        }
        competition_log = self._log_loss_competition_charts(
            raw_floats, normalized_floats, weighted_floats, optimizer_floats, epoch_name
        )

        # Merge competition charts into single log call to avoid WandB step corruption
        log_dict.update(competition_log)

        # --- GPU memory diagnostics (add to log_dict before single wandb.log call) -------
        if self._memory_diagnostics and device.type == "cuda":
            allocated_mb: float = torch.cuda.memory_allocated(device) / 1024**2
            reserved_mb: float = torch.cuda.memory_reserved(device) / 1024**2
            peak_mb: float = torch.cuda.max_memory_allocated(device) / 1024**2
            free_cached_mb: float = reserved_mb - allocated_mb
            log_dict["memory_stats/allocated_mb"] = allocated_mb
            log_dict["memory_stats/reserved_mb"] = reserved_mb
            log_dict["memory_stats/peak_allocated_mb"] = peak_mb
            log_dict["memory_stats/free_cached_mb"] = free_cached_mb

        # Defer the actual wandb.log call — model_baseclass.py will merge gradient metrics
        # into this dict and fire a single wandb.log per epoch after backward passes complete.
        self._pending_wandb_log: dict = log_dict

        # --- Mesh extraction & rendering ----------------------------------
        # CentralVisManager path: independent of renderer_manager. This handles
        # the PLY/GIF export pipeline. The neural-renderer branch below runs
        # *in addition* so that detached modes (MeshSDF_grads_*) still receive
        # per-loss vertex gradient updates — they live inside
        # `_renderer_manager` and were otherwise being skipped entirely when
        # `_vis_manager` was active.
        _rendering_active = self._rendering_freq is None or epoch_name % self._rendering_freq == 0
        if self._vis_manager is not None and _rendering_active:
            verts_vm, faces_vm = self._mesh_extractor.extract_mesh(
                model=model,
                isovalue=self._mesh_extraction_config.get("isovalue", 0.0),
                batch_size=self._mesh_extraction_config.get("batch_size", 8192),
            )
            if self._mesh_statistics_enabled:
                self._mesh_statistics.log_basic_mesh_stats(verts_vm, faces_vm, epoch_name)
                self._mesh_statistics.log_solar_area_stats(verts_vm, faces_vm, epoch_name)
            _verts_np = verts_vm.detach().cpu().numpy() if torch.is_tensor(verts_vm) else verts_vm
            _faces_np: np.ndarray | None = (
                faces_vm.detach().cpu().numpy()
                if torch.is_tensor(faces_vm)
                else (faces_vm if isinstance(faces_vm, np.ndarray) else None)
            )
            epoch_int = int(epoch_name) if isinstance(epoch_name, (int, float)) else 0
            self._vis_manager.vis_mesh_shape(epoch_int, verts_vm, faces_vm)
            # Per-loss gradient PLYs go through the central manager. This is
            # decoupled from the neural-renderer branch so the gradient panels
            # work even when `use_neural_renderer=False`.
            try:
                _grads_payload = self.collect_vertex_grads_for_visualisation(
                    weights_by_name=self._manual_weights,
                )
            except Exception as _exc:
                print(f"collect_vertex_grads_for_visualisation failed: {_exc}")
                _grads_payload = None
            if _grads_payload is not None:
                _per_loss_grads = _grads_payload["per_loss"]
                _per_loss_meshes = _grads_payload.get("per_loss_meshes", {})
                for _loss_key, _grads in _per_loss_grads.items():
                    _mesh_entry = _per_loss_meshes.get(_loss_key)
                    if _mesh_entry is None:
                        continue
                    _v, _f = _mesh_entry
                    self._vis_manager.vis_mesh_gradient(epoch_int, _loss_key, _v, _f, _grads)

        # Neural-renderer / CAD fallback path. Runs whether or not the central
        # vis_manager is active, so detached MeshSDF_grads_* modes keep getting
        # per-loss vertex gradient frames.
        if self._use_renderer and self._renderer_manager is not None and _rendering_active:
            verts, faces = self._mesh_extractor.extract_mesh(
                model=model,
                isovalue=self._mesh_extraction_config.get("isovalue", 0.0),
                batch_size=self._mesh_extraction_config.get("batch_size", 8192),
            )
            if self._mesh_statistics_enabled:
                self._mesh_statistics.log_basic_mesh_stats(verts, faces, epoch_name)
                self._mesh_statistics.log_solar_area_stats(verts, faces, epoch_name)

            if self._rendering_freq is not None and epoch_name % self._rendering_freq == 0:
                if self._use_neural_renderer:
                    try:
                        runtime_assembly_id = self._active_assembly_id()
                        # Per-loss dL/dv from the most recent _compute_raw_losses
                        # (mirrors diversity_manager._render injection for the concat path).
                        grads_payload = self.collect_vertex_grads_for_visualisation(
                            weights_by_name=self._manual_weights,
                        )
                        extra_state: dict[str, Any] = {}
                        per_loss_grads: dict[str, Any] = {}
                        per_loss_meshes: dict[str, Any] = {}
                        if grads_payload is not None:
                            verts = grads_payload["vertices"]
                            faces = grads_payload["faces"]
                            per_loss_grads = grads_payload["per_loss"]
                            per_loss_meshes = grads_payload.get("per_loss_meshes", {})
                            extra_state["vertice_grads_per_loss"] = per_loss_grads
                            extra_state["vertice_meshes_per_loss"] = per_loss_meshes
                        self._renderer_manager.update(
                            {
                                "model": model,
                                "iteration": epoch_name,
                                "loss": total_loss.item(),
                                "vertices": verts,
                                "faces": faces,
                                "assembly_id": runtime_assembly_id,
                                "bb_assembly_id": runtime_assembly_id,
                                **extra_state,
                            }
                        )
                    except Exception as e:
                        print(f"Renderer wrapper update failed: {e}")
                else:
                    self._run_cad_vis(verts, faces, epoch_name)

        # --- GPU memory diagnostics (print only; logging is merged in _handle_telemetry) ----
        self._print_memory_diagnostics(epoch_name, device)

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

    def _print_memory_diagnostics(self, epoch_name: int, device: torch.device) -> None:
        """Print GPU memory stats to stdout (toggled by logging.yaml). WandB logging is handled in _handle_telemetry."""
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

    # ------------------------------------------------------------------
    # Loss competition charts
    # ------------------------------------------------------------------
    def _log_loss_competition_charts(  # noqa: PLR0915
        self,
        raw_floats: dict[str, float],
        normalized_floats: dict[str, float],
        weighted_floats: dict[str, float],
        optimizer_floats: dict[str, float],
        epoch_name: int,
    ) -> dict[str, Any]:
        """Generate loss-competition visualizations and return as dict for WandB.

        Four views (raw / normalized / weighted / optimizer) each get a pie and bar chart.
        A radar overlay (raw vs weighted) and a stacked-area timeline are also produced.
        """
        matplotlib.use("Agg")  # headless cluster - no display server
        self._raw_loss_history.append(dict(raw_floats))
        self._normalized_loss_history.append(dict(normalized_floats))
        self._weighted_loss_history.append(dict(weighted_floats))
        self._optimizer_loss_history.append(dict(optimizer_floats))

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
        n_labels = list(normalized_floats.keys())
        n_vals = [normalized_floats[k] for k in n_labels]
        w_labels = list(weighted_floats.keys())
        w_vals = [weighted_floats[k] for k in w_labels]
        opt_labels = list(optimizer_floats.keys())
        opt_vals = [optimizer_floats[k] for k in opt_labels]

        fig_a = _safe_pie(raw_vals, raw_labels, f"Raw Losses (epoch {epoch_name})")
        competition_log["loss_competition/raw_pie"] = wandb.Image(fig_a)
        plt.close(fig_a)
        fig_b = _safe_pie(n_vals, n_labels, f"Normalized Losses (epoch {epoch_name})")
        competition_log["loss_competition/normalized_pie"] = wandb.Image(fig_b)
        plt.close(fig_b)
        fig_c = _safe_pie(w_vals, w_labels, f"Weighted Losses (epoch {epoch_name})")
        competition_log["loss_competition/weighted_pie"] = wandb.Image(fig_c)
        plt.close(fig_c)
        fig_d = _safe_pie(opt_vals, opt_labels, f"Optimizer Losses (epoch {epoch_name})")
        competition_log["loss_competition/optimizer_pie"] = wandb.Image(fig_d)
        plt.close(fig_d)
        fig_e = _safe_hbar(raw_vals, raw_labels, f"Raw Loss Shares % (epoch {epoch_name})")
        competition_log["loss_competition/raw_bar"] = wandb.Image(fig_e)
        plt.close(fig_e)
        fig_f = _safe_hbar(n_vals, n_labels, f"Normalized Loss Shares % (epoch {epoch_name})")
        competition_log["loss_competition/normalized_bar"] = wandb.Image(fig_f)
        plt.close(fig_f)
        fig_g = _safe_hbar(w_vals, w_labels, f"Weighted Loss Shares % (epoch {epoch_name})")
        competition_log["loss_competition/weighted_bar"] = wandb.Image(fig_g)
        plt.close(fig_g)
        fig_h = _safe_hbar(opt_vals, opt_labels, f"Optimizer Loss Shares % (epoch {epoch_name})")
        competition_log["loss_competition/optimizer_bar"] = wandb.Image(fig_h)
        plt.close(fig_h)

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

            fig_radar, ax_radar = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
            ax_radar.plot(angles, raw_norm, "o--", color="steelblue", linewidth=1.5, label="raw (norm)")
            ax_radar.fill(angles, raw_norm, color="steelblue", alpha=0.15)
            ax_radar.plot(angles, w_norm, "o-", color="tomato", linewidth=2, label="weighted (norm)")
            ax_radar.fill(angles, w_norm, color="tomato", alpha=0.15)
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(all_keys, fontsize=9)
            ax_radar.set_title(f"Loss Radar (epoch {epoch_name})", pad=15)
            ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            fig_radar.tight_layout()
            competition_log["loss_competition/radar"] = wandb.Image(fig_radar)
            plt.close(fig_radar)

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

            fig_area, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            _stacked_area(self._raw_loss_history, ax_top, "Raw Loss Competition Over Training")
            _stacked_area(self._weighted_loss_history, ax_bot, "Weighted Loss Competition Over Training")
            ax_bot.set_xlabel("Epoch")
            fig_area.tight_layout()
            competition_log["loss_competition/stacked_area"] = wandb.Image(fig_area)
            plt.close(fig_area)

        return competition_log

    # ------------------------------------------------------------------
    # Loss & gradient normalisation calibration
    # ------------------------------------------------------------------
    def _calibrate(  # noqa: PLR0912, PLR0915
        self,
        model: TrainModel,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Calibrate loss and gradient scale factors before training begins.

        Runs one full forward pass with calibration_mode=True (bypasses epoch scheduling),
        then one backward pass to measure per-component gradient norms on top of loss-scaled
        tensors. Null losses (requires_grad=False) are skipped — their scales stay at 1.0.
        Optimizer gradients are zeroed before and after to leave no trace.
        """
        print("[LossNormalisation] Starting calibration pass...")
        optimizer.zero_grad()

        loss_scales: dict[str, float] = {}
        grad_scales: dict[str, float] = {}
        calibrated_names: set[str] = set()

        active_names: set[str] = set(self._active_losses.keys())
        if self._use_cse:
            drag_name = self._objective_losses.get(self._cse_drag_category, (None,))[0]
            solar_name = self._objective_losses.get(self._cse_solar_category, (None,))[0]
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
            optimizer.zero_grad()
            print("[LossNormalisation] Calibration skipped (all active components use fixed loss+grad scales).")
            return

        # --- Loss scale calibration (forward only) ---
        raw = self._compute_raw_losses(model, epoch_name=0, calibration_mode=True)
        for name, tensor in raw.items():
            fixed_loss = self._norm_manager.get_fixed_loss_scale(name)
            if fixed_loss is not None:
                loss_scales[name] = fixed_loss
                calibrated_names.add(name)
                print(f"  Loss scale for {name}: fixed={fixed_loss:.6g}")
                continue
            mean_val = abs(tensor.detach().item())
            loss_scales[name] = self._norm_manager._target_loss / mean_val if mean_val > 0 else 1.0
            calibrated_names.add(name)
            print(f"  Loss scale for {name}: mean_raw={mean_val:.6g}, scale={loss_scales[name]:.6g}")

        # CSE: scale is derived from the raw v_cruise output, not individual drag/solar.
        if self._use_cse:
            print("  CSE mode detected: computing cse_objective scale...")
            # Temporarily record loss_scales (no grad_scales yet) so _apply_weighting can run.
            # cse_scale will be 1.0*1.0=1.0 here (grad_scales empty → get_grad_scale returns 1.0).
            self._norm_manager.record_calibration(loss_scales, {}, calibrated_names)
            fixed_cse_loss = self._norm_manager.get_fixed_loss_scale("cse_objective")
            if fixed_cse_loss is not None:
                loss_scales["cse_objective"] = fixed_cse_loss
                calibrated_names.add("cse_objective")
                print(f"  Loss scale for cse_objective: fixed={fixed_cse_loss:.6g}")
            else:
                self._apply_weighting(raw, calibration_pass=True)
                if "cse_objective" in self._last_weighted_losses:
                    # loss_scale=1.0 and grad_scale=1.0 at this point, so the value equals -v_cruise
                    raw_cse_val = abs(self._last_weighted_losses["cse_objective"].detach().item())
                    loss_scales["cse_objective"] = self._norm_manager._target_loss / raw_cse_val if raw_cse_val > 0 else 1.0
                    calibrated_names.add("cse_objective")
                    print(f"  Loss scale for cse_objective: raw={raw_cse_val:.6g}, scale={loss_scales['cse_objective']:.6g}")
            drag_name = self._objective_losses.get(self._cse_drag_category, (None,))[0]
            solar_name = self._objective_losses.get(self._cse_solar_category, (None,))[0]
            for merged_name in (drag_name, solar_name):
                if merged_name:
                    print(f"  Excluding {merged_name} (merged into cse_objective)")
                    calibrated_names.discard(merged_name)
                    loss_scales.pop(merged_name, None)

        # Store loss scales so _apply_weighting applies them during gradient calibration.
        self._norm_manager.record_calibration(loss_scales, {}, calibrated_names)

        # --- Gradient scale calibration (backward pass) ---
        print("[LossNormalisation] Computing gradient scales...")
        optimizer.zero_grad()
        raw = self._compute_raw_losses(model, epoch_name=0, calibration_mode=True)
        self._apply_weighting(raw, calibration_pass=True)  # loss_scales now active; populates _last_weighted_losses
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        prev_grad = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in trainable_params]

        components_with_grad = [(name, t) for name, t in self._last_weighted_losses.items() if t.requires_grad]
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
            if fixed_grad is not None and name not in grad_scales:
                grad_scales[name] = fixed_grad

        self._norm_manager.record_calibration(loss_scales, grad_scales, calibrated_names)
        optimizer.zero_grad()

        summary_items = [
            f"{n}(loss={loss_scales.get(n, 1.0):.3g}, grad={grad_scales.get(n, 1.0):.3g})" for n in sorted(calibrated_names)
        ]
        print(f"[LossNormalisation] Calibration complete. {len(calibrated_names)} losses calibrated:")
        for item in summary_items:
            print(f"  ✓ {item}")

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
        raw: dict[str, torch.Tensor] = self._compute_raw_losses(model, epoch_name)

        # 2. Apply weighting (populates _last_weighted_losses)
        total: torch.Tensor = self._apply_weighting(raw)
        self.latest_weighted_metrics = self._last_weighted_losses

        # 3. Build logging dict and dispatch telemetry
        if not skip_telemetry:
            loss_log: dict[str, float] = {f"raw_losses/{name}": t.item() for name, t in raw.items()}
            # Add 0 entries for skipped losses
            for cat in getattr(self, "_skipped_categories", set()):
                for role_dict in (self._objective_losses, self._constraint_losses):
                    if cat in role_dict:
                        loss_log[f"raw_losses/{role_dict[cat][0]}"] = 0.0
                if cat in self._auxiliary_losses:
                    loss_log[f"raw_losses/{cat}"] = 0.0
            raw_total = 0.0
            for name, t in raw.items():
                category = self._loss_name_to_category.get(name, "")
                sign = float(self._objective_signs.get(category, 1))
                raw_total += sign * t.item()
            loss_log["raw_losses/total_loss"] = raw_total
            self._handle_telemetry(loss_log, epoch_name, device, model, total)

        if self._external_alm:
            constraint_names: list[str] = [n for (n, _inst) in self._constraint_losses.values()]
            raw_constraints_dict = {name: raw[name] for name in constraint_names}
            return total, raw_constraints_dict

        return total
