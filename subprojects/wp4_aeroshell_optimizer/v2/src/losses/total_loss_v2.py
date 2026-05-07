"""Clean loss pipeline for non-legacy training loops (single- or population-based).

Responsibilities are strictly limited to loss computation and weighting:

* compute(model): evaluate all active loss instances.
* weight(bundle, alm_manager=None): CSE/manual objectives + ALM/manual constraints + manual aux.

No wandb, no mesh extraction, no matplotlib, and no ALM state updates happen here.
The caller (for example DiversityTrainer) owns logging, rendering, ALM multiplier
updates, and any other telemetry.

Role classification is driven from YAML via a _role metadata key per category
block (objective, constraint, aux). _sign: maximize flips the sign of an objective
in manual (non-CSE) mode. Keys starting with "_" are metadata and never loaded as
losses. Constraint ordering for ALM follows YAML category order.
"""

from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.profiler import record_function

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.alm import ALMManager
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.cruising_speed import cruising_speed
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.loss_normalisation import LossNormalisationManager
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsModelContract
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.dynamic_loading import load_class_from_path
from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor

_LOGGER = logging.getLogger("total_loss_v2")


@dataclass
class RawLossBundle:
    """Raw, unweighted loss tensors grouped by YAML role.

    Keys are category names (the YAML block names: "aero", "bb", ...). Each category
    has exactly one active loss (the one with compute=1 in its block).

    ``zero_weighted`` holds tensors for losses that are scheduled-inactive but still
    computed (deactivate_scheduler_mode="zero_weighted"). They are logged at their real value
    but contribute 0 to the gradient.

    ``side_data`` is a forward-compat slot: individual losses may write extra
    rendering/telemetry data here. TotalLossV2 does not populate or read it today;
    the caller decides whether to consume it.
    """

    # Use fields for mypy related stuff, has to do with the @dataclass decorator
    objectives: dict[str, torch.Tensor] = field(default_factory=dict)
    constraints: dict[str, torch.Tensor] = field(default_factory=dict)
    aux: dict[str, torch.Tensor] = field(default_factory=dict)
    zero_weighted: dict[str, torch.Tensor] = field(default_factory=dict)
    side_data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        def fmt_dict(d: dict[str, torch.Tensor]) -> str:
            return ", ".join(f"{k}: {v.item():.4e}" for k, v in d.items())

        return (
            "RawLossBundle: "
            f"Objectives: {{{fmt_dict(self.objectives)}}},\n"
            f"Constraints: {{{fmt_dict(self.constraints)}}},\n"
            f"Aux: {{{fmt_dict(self.aux)}}},\n"
            f"Zero-weighted: {{{fmt_dict(self.zero_weighted)}}},\n"
            f"Side-data keys: {list(self.side_data.keys())}"
        )


@dataclass
class WeightedLossResult:
    """Output of ``TotalLossV2.weight``.

    ``scalar`` keeps the live autograd graph for backward(). ``log_data`` holds
    detached (host-side) floats suitable for averaging across a population and
    feeding directly into wandb.log. ``components`` holds per-loss weighted tensors
    (with live graphs) when return_components=True.
    """

    scalar: torch.Tensor
    log_data: dict[str, float] = field(default_factory=dict)
    components: dict[str, torch.Tensor] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"WeightedLossResult content: \n Scalar: {self.scalar.item():.4e},\nLog data: {self.log_data},\nComponents: {{{', '.join(f'{k}: {v.item():.4e}' for k, v in self.components.items())}}}"


class TotalLossV2:
    """Aggregates and weights loss components — compute + weight, nothing else."""

    def __init__(self, device: str | torch.device = "cuda") -> None:
        self._device: torch.device = torch.device(device) if isinstance(device, str) else device

        general_config: dict[str, Any] = load_config()
        self.losses_config: dict[str, Any] = load_config(
            v=general_config["general_info"]["version"],
            cfg_name=general_config["active_configs"]["losses"],
        )

        gi: dict[str, Any] = self.losses_config["general_info"]
        self.use_cse: bool = gi.get("CSE") == "True"
        self._deactivate_scheduler_mode: str = gi["deactivate_scheduler_mode"]

        # CSE physical constants — defaults match historical hardcoded values
        cse_params: dict[str, Any] = gi.get("cse_params", {})
        self._cse_drag_category: str = cse_params.get("drag_category", "aero")
        self._cse_solar_category: str = cse_params.get("solar_category", "solar")
        self._cse_mass: float = float(cse_params.get("mass", 210))
        self._cse_c_losses: float = float(cse_params.get("c_losses", 20))
        self._cse_rolling_friction: float = float(cse_params.get("rolling_friction_coefficient", 0.006))
        self._cse_max_iter: int = int(cse_params.get("max_iter", 12))
        self._cse_tol: float = float(cse_params.get("tol", 1e-4))
        # Last-known raw (pre-prefactor) value per CSE objective category, used when that category
        # is scheduled out so the other objective can still run CSE with a meaningful partner value.
        self.solar_default_value: float = gi["cse_defaults"]["solar"]
        self.drag_default_value: float = gi["cse_defaults"]["drag"]
        self._cse_prefactors: dict[str, float] = gi.get("cse_prefactors", {})
        self._cse_weight: float = float(gi.get("cse_weight", 1.0))

        self._init_active_losses()
        self._warn_about_dead_weights()

    def _init_active_losses(self) -> None:  # noqa: PLR0912, PLR0915
        """Instantiate active losses and build role-grouped internal lookup maps.

        Sets up `_objective_by_category`, `_constraint_by_category`, `_aux_by_category`,
        `_objective_signs`, and `_manual_weights` from the configured YAML loss blocks.

        Validation is strict: every non-"total" category must define a valid `_role`.
        Roles objective/constraint/aux must each have exactly one enabled loss entry
        (`compute=1`) with a valid `[compute, weight, path]` structure. Additional
        roles (for example diversity) are ignored by TotalLossV2.
        """
        project_root: Path = Path.cwd()
        losses_block_raw: Any = self.losses_config.get("losses")
        if not isinstance(losses_block_raw, dict):
            raise TypeError("Invalid losses config: expected top-level 'losses' to be a dictionary.")
        losses_block: dict[str, Any] = losses_block_raw
        if not losses_block:
            raise ValueError("Invalid losses config: no categories defined under 'losses'.")

        # Category -> (loss_name, instance); dict preserves insertion order on Python 3.7+
        self._objective_by_category: dict[str, tuple[str, Any]] = {}
        self._constraint_by_category: dict[str, tuple[str, Any]] = {}
        self._aux_by_category: dict[str, tuple[str, Any]] = {}
        self._objective_signs: dict[str, int] = {}  # only used in manual (non-CSE) objective mode
        self._manual_weights: dict[str, float] = {}  # loss_name -> weight from YAML [compute, weight, path]
        # category -> list of (start, end) inclusive epoch ranges where the loss is DISABLED
        self._disabled_epochs: dict[str, list[tuple[int, int]]] = {}

        for category, block in losses_block.items():
            if category == "total":
                continue
            if not isinstance(block, dict):
                raise TypeError(f"Invalid category '{category}': expected a dictionary block with _role and loss entries.")

            role: str | None = block.get("_role")
            if role is None:
                raise ValueError(f"Missing _role for category '{category}'. Expected one of ['objective', 'constraint', 'aux'].")
            if role not in {"objective", "constraint", "aux", "diversity"}:
                raise ValueError(
                    f"Invalid _role '{role}' for category '{category}'. Expected one of ['aux', 'constraint', 'diversity', 'objective']."
                )
            if role == "diversity":  # we skip diversity role in this class, it's handled separately in DiversityLossManager
                continue

            sign_val: int = -1 if block.get("_sign", "minimize") == "maximize" else 1

            # Find exactly one active loss in this category (compute=1)
            enabled_entries: list[tuple[str, float, str]] = []
            for loss_name, entry in block.items():
                if loss_name.startswith("_"):
                    continue
                if not isinstance(entry, list) or len(entry) < 3:
                    raise ValueError(
                        f"Invalid entry for '{category}.{loss_name}': expected [compute, weight, path], got {entry!r}."
                    )
                compute, weight, path = entry[0], entry[1], entry[2]
                if isinstance(compute, bool):
                    is_enabled = compute
                elif isinstance(compute, int):
                    if compute not in {0, 1}:
                        raise ValueError(
                            f"Invalid compute flag for '{category}.{loss_name}': expected 0/1 or bool, got {compute!r}."
                        )
                    is_enabled = bool(compute)
                else:
                    raise TypeError(f"Invalid compute flag for '{category}.{loss_name}': expected 0/1 or bool, got {compute!r}.")

                try:
                    weight_float = float(weight)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid weight for '{category}.{loss_name}': expected numeric value, got {weight!r}."
                    ) from exc

                if not isinstance(path, str) or not path.strip():
                    raise ValueError(f"Invalid path for '{category}.{loss_name}': expected non-empty string, got {path!r}.")

                if not is_enabled:
                    continue

                enabled_entries.append((loss_name, weight_float, path))

            if not enabled_entries:
                raise ValueError(f"No enabled loss entry for category '{category}'. Set exactly one entry to compute=1.")
            if len(enabled_entries) > 1:
                enabled_names = [name for name, _w, _p in enabled_entries]
                raise ValueError(
                    f"Multiple enabled loss entries for category '{category}': {enabled_names}. "
                    "Set exactly one entry to compute=1."
                )

            loss_name, weight_float, path = enabled_entries[0]

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
            self._manual_weights[loss_name] = weight_float
            active_entry = (loss_name, instance)

            if role == "objective":
                self._objective_by_category[category] = active_entry
                self._objective_signs[category] = sign_val
            elif role == "constraint":
                self._constraint_by_category[category] = active_entry
            else:  # aux
                self._aux_by_category[category] = active_entry

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
                for cat, (name, _) in self._objective_by_category.items()
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
                for cat, (name, _) in self._constraint_by_category.items()
                if name in self._manual_weights and self._manual_weights[name] != 1.0
            ]
            if dead_constraints:
                print(
                    f"⚠️  WARNING: ALM is enabled (ALM=True). Constraint weights for {dead_constraints} "
                    f"are dead code and will be ignored. Use 'initial_lambda' and 'initial_mu' config entries to influence constraint strength."
                )

    def _is_category_active(self, category: str, epoch: int) -> bool:
        return all(
            not (start <= epoch and (end == -1 or epoch <= end)) for start, end in self._disabled_epochs.get(category, [])
        )  # Ruff mumbo jumbo instead of for loop here

    # ------------------------------------------------------------------
    # Mesh extraction sharing
    # ------------------------------------------------------------------
    def _all_loss_instances(self) -> dict[str, Any]:
        """Flat view of all active loss instances keyed by loss_name."""
        out: dict[str, Any] = {}
        out.update(dict(self._objective_by_category.values()))
        out.update(dict(self._constraint_by_category.values()))
        out.update(dict(self._aux_by_category.values()))
        return out

    def _find_mesh_extractors(self) -> dict[str, MeshSDFExtractor]:
        """Discover MeshSDFExtractor instances on active loss objects (up to 1 level deep)."""
        extractors: dict[str, MeshSDFExtractor] = {}
        _check_attrs = ("mesh_extractor", "extractor")
        for name, instance in self._all_loss_instances().items():
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

    def _inject_shared_meshes(self, model: SupportsModelContract) -> None:
        """Pre-extract meshes for groups of losses that share the same extraction config."""
        extractors = self._find_mesh_extractors()
        if len(extractors) < 2:
            return

        groups: dict[tuple, list[tuple[str, MeshSDFExtractor]]] = defaultdict(list)
        for loss_name, ext in extractors.items():
            groups[ext.extraction_key()].append((loss_name, ext))

        for members in groups.values():
            if len(members) < 2:
                continue
            primary_name, primary_ext = members[0]
            vertices, faces = primary_ext.extract_mesh(model)
            _LOGGER.debug(
                "mesh-sharing: %d vertices via '%s', sharing with %s",
                int(vertices.shape[0]),
                primary_name,
                [n for n, _ in members[1:]],
            )
            for _name, ext in members[1:]:
                ext.set_shared_mesh(vertices, faces)

        self._shared_mesh_extractors = extractors

    def _clear_shared_meshes(self) -> None:
        """Clear all injected shared meshes after the loss computation loop."""
        for ext in getattr(self, "_shared_mesh_extractors", {}).values():
            ext.clear_shared_mesh()
        self._shared_mesh_extractors = {}

    def finalize_epoch_mesh_stats(self) -> None:
        """Emit one compact mesh-extraction summary per epoch and reset counters."""
        seen: set[int] = set()
        for ext in self._find_mesh_extractors().values():
            ext_id = id(ext)
            if ext_id in seen:
                continue
            seen.add(ext_id)

            stats = ext.get_stats()
            if int(stats.get("calls", 0)) == 0:
                continue

            ext.log_stats_summary(level=logging.INFO)
            ext.reset_stats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def constraint_names(self) -> list[str]:
        """Active constraint loss names, in YAML category order (= ALM vector order)."""
        return [name for (name, _inst) in self._constraint_by_category.values()]

    @property
    def constraint_category_order(self) -> list[str]:
        """Category names for active constraints, in YAML order."""
        return list(self._constraint_by_category.keys())

    @record_function("TotalLossV2.compute")
    def compute(
        self,
        model: SupportsModelContract,
        epoch_name: int = 0,
        calibration_mode: bool = False,
    ) -> RawLossBundle:
        """Evaluate every active loss instance and return raw (unweighted) tensors.

        Args:
            model: The trainable model passed to each active loss instance.
            epoch_name: Training step/epoch identifier forwarded to each loss call.
            calibration_mode: If True, bypass epoch scheduling and compute all losses regardless
                of _disabled_epochs. Used by LossNormalisationManager to calibrate scale factors
                for losses that are not yet active at epoch 0.

        Returns:
            RawLossBundle with unweighted objective, constraint, and aux tensors
            grouped by category.
        """
        self._inject_shared_meshes(model)
        bundle = RawLossBundle()
        self._skipped_log_zeros: dict[str, float] = {}
        try:
            for role_dict, target_dict in (
                (self._objective_by_category, bundle.objectives),
                (self._constraint_by_category, bundle.constraints),
                (self._aux_by_category, bundle.aux),
            ):
                for category, (loss_name, instance) in role_dict.items():
                    if not calibration_mode and not self._is_category_active(category, epoch_name):
                        is_cse_managed_objective = (
                            self.use_cse
                            and role_dict is self._objective_by_category
                            and category in {self._cse_drag_category, self._cse_solar_category}
                        )
                        if self._deactivate_scheduler_mode == "zero_weighted":
                            with record_function(f"loss_{loss_name}"):
                                sig = inspect.signature(instance.get_loss)
                                if "skip_telemetry" in sig.parameters:
                                    bundle.zero_weighted[category] = instance.get_loss(model, epoch_name, skip_telemetry=True)
                                else:
                                    bundle.zero_weighted[category] = instance.get_loss(model, epoch_name)
                        elif not is_cse_managed_objective:
                            self._skipped_log_zeros[f"raw_losses/{loss_name}"] = 0.0
                        continue
                    with record_function(f"loss_{loss_name}"):
                        sig = inspect.signature(instance.get_loss)
                        if "skip_telemetry" in sig.parameters:
                            target_dict[category] = instance.get_loss(model, epoch_name, skip_telemetry=True)
                        else:
                            target_dict[category] = instance.get_loss(model, epoch_name)
        finally:
            self._clear_shared_meshes()

        return bundle

    @record_function("TotalLossV2.weight")
    def weight(  # noqa: PLR0912, PLR0915
        self,
        bundle: RawLossBundle,
        alm_manager: ALMManager | None = None,
        return_components: bool = False,
        norm_manager: LossNormalisationManager | None = None,
        calibration_pass: bool = False,
    ) -> WeightedLossResult:
        """Combine raw losses into a backpropable scalar plus a detached log dict.

        * Objectives: CSE (drag+solar -> cruising speed) if ``self.use_cse``, else manual weights with _sign.
        * Constraints: ALM ``lambda*c + 0.5*mu*c^2`` if ``alm_manager`` is given, else manual weights.
        * Auxiliaries: always manual weights.

        When norm_manager is provided, both loss_scale and grad_scale are baked into each component
        tensor before it is returned (and before backward is called by the caller).

        Args:
            bundle: Raw unweighted losses grouped by role/category from `compute`.
            alm_manager: Optional ALM manager for constraint penalty weighting.
            return_components: If True, populate WeightedLossResult.components with per-loss tensors
                (with live autograd graphs) for gradient conflict analysis. No extra computation.
            norm_manager: Optional normalisation manager; applies calibrated loss and gradient scales.

                Returns:
            WeightedLossResult where `scalar` is used for backward, `log_data` contains detached
            numeric values for telemetry, and `components` holds per-loss tensors if return_components=True.

                Telemetry namespaces:
                - raw_losses/*: direct loss outputs (pre-weight, pre-normalization).
                - normalized_losses/*: raw value after loss-scale only (for interpretability).
                - optimizer_losses/*: exact per-component value entering the optimizer sum
                    (after sign/weights/ALM and any grad-scale).
                - weighted_losses/*: legacy namespace retained for dashboard compatibility.

        ALM multiplier updates are NOT performed here; the caller owns that lifecycle.
        """
        log_data: dict[str, float] = {}
        weighted_components: dict[str, torch.Tensor] = {}

        loss_scale = (
            (lambda component_name, raw_value: raw_value * norm_manager.get_loss_scale(component_name))
            if norm_manager
            else (lambda _component_name, raw_value: raw_value)
        )

        # Raw logging — active losses by loss_name, zero_weighted at real value, skipped as 0
        for category, t in bundle.objectives.items():
            objective_name = self._objective_by_category[category][0]
            raw_value = t.item()
            log_data[f"raw_losses/{objective_name}"] = raw_value
            log_data[f"normalized_losses/{objective_name}"] = loss_scale(objective_name, raw_value)
        for category, t in bundle.constraints.items():
            constraint_name = self._constraint_by_category[category][0]
            raw_value = t.item()
            log_data[f"raw_losses/{constraint_name}"] = raw_value
            log_data[f"normalized_losses/{constraint_name}"] = loss_scale(constraint_name, raw_value)
        for category, t in bundle.aux.items():
            aux_name = self._aux_by_category[category][0]
            raw_value = t.item()
            log_data[f"raw_losses/{aux_name}"] = raw_value
            log_data[f"normalized_losses/{aux_name}"] = loss_scale(aux_name, raw_value)
        for category, t in bundle.zero_weighted.items():
            lookup = (
                self._objective_by_category.get(category)
                or self._constraint_by_category.get(category)
                or self._aux_by_category.get(category)
            )
            assert lookup is not None, (
                f"Category '{category}' in zero_weighted but not found in objectives, constraints, or aux lookups."
            )
            raw_name = lookup[0]
            raw_value = t.item()
            log_data[f"raw_losses/{raw_name}"] = raw_value
            log_data[f"normalized_losses/{raw_name}"] = loss_scale(raw_name, raw_value)
            if not (
                self.use_cse
                and category in {self._cse_drag_category, self._cse_solar_category}
                and category in self._objective_by_category
            ):
                log_data[f"weighted_losses/{lookup[0]}"] = 0.0
                log_data[f"optimizer_losses/{lookup[0]}"] = 0.0
        log_data.update(getattr(self, "_skipped_log_zeros", {}))

        # --- Objectives ---------------------------------------------------
        if self.use_cse:
            if self._cse_drag_category not in self._objective_by_category:
                raise KeyError(
                    f"CSE active but drag category '{self._cse_drag_category}' is not configured. "
                    f"Configured objective categories: {list(self._objective_by_category.keys())}"
                )
            if self._cse_solar_category not in self._objective_by_category:
                raise KeyError(
                    f"CSE active but solar category '{self._cse_solar_category}' is not configured. "
                    f"Configured objective categories: {list(self._objective_by_category.keys())}"
                )
            drag_cat = self._cse_drag_category
            solar_cat = self._cse_solar_category
            drag_pf: float = float(self._cse_prefactors.get("drag", 1.0))
            solar_pf: float = float(self._cse_prefactors.get("solar", 1.0))

            # Resolve the tensor each CSE input will use.
            # Active: live tensor (gradient flows). Inactive: cached detached value (no gradient).
            # Cold-start fallback (cache empty, already off at epoch 0): zeros, like a weight-0 null loss.
            drag_is_live: bool = drag_cat in bundle.objectives
            solar_is_live: bool = solar_cat in bundle.objectives
            drag_raw: torch.Tensor = (
                bundle.objectives[drag_cat] if drag_is_live else self.drag_default_value * torch.ones(1, device=self._device)
            )
            solar_raw: torch.Tensor = (
                bundle.objectives[solar_cat] if solar_is_live else self.solar_default_value * torch.ones(1, device=self._device)
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
            log_data["cruising_speed_ms"] = v_cruise.item()
            log_data["cse/raw_drag_force_n"] = float(drag_raw.detach().item())
            log_data["cse/raw_solar_power_w"] = float(solar_raw.detach().item())
            log_data["cse/cruising_speed_ms"] = v_cruise.item()
            cse_loss_scale = norm_manager.get_loss_scale("cse_objective") if norm_manager else 1.0
            log_data["normalized_losses/cse_objective"] = v_cruise.item() * cse_loss_scale
            # Only backprop if at least one input is live; both dead → v_cruise has no grad.
            if drag_is_live or solar_is_live:
                # Minimize -v_cruise to maximize physical speed
                cse_scale = (
                    (norm_manager.get_loss_scale("cse_objective") * norm_manager.get_grad_scale("cse_objective"))
                    if norm_manager
                    else 1.0
                )
                cse_weight = 1.0 if calibration_pass else self._cse_weight
                weighted_components["cse_objective"] = -v_cruise * cse_scale * cse_weight
                log_data["weighted_losses/cse_objective"] = (v_cruise * cse_loss_scale * cse_weight).item()
                log_data["optimizer_losses/cse_objective"] = weighted_components["cse_objective"].detach().item()
        else:
            for category, t in bundle.objectives.items():
                name: str = self._objective_by_category[category][0]
                w: float = 1.0 if calibration_pass else self._manual_weights.get(name, 1.0)
                sign: float = float(self._objective_signs.get(category, 1))
                combined_scale = (norm_manager.get_loss_scale(name) * norm_manager.get_grad_scale(name)) if norm_manager else 1.0
                wt: torch.Tensor = sign * w * t * combined_scale
                weighted_components[name] = wt
                log_data[f"weighted_losses/{name}"] = wt.item()
                log_data[f"optimizer_losses/{name}"] = wt.item()

        # --- Constraints --------------------------------------------------
        # Derive active constraint indices relative to the full YAML-ordered constraint list,
        # so that ALM multipliers for inactive (scheduled-out) constraints are frozen.
        all_constraint_cats = self.constraint_category_order
        active_constraint_cats = [cat for cat in all_constraint_cats if cat in bundle.constraints]
        active_constraint_indices = [i for i, cat in enumerate(all_constraint_cats) if cat in bundle.constraints]
        active_constraint_tensors: list[torch.Tensor] = []
        for cat in active_constraint_cats:
            c_tensor = bundle.constraints[cat]
            if norm_manager is not None:
                c_name = self._constraint_by_category[cat][0]
                c_tensor = c_tensor * norm_manager.get_loss_scale(c_name)
            active_constraint_tensors.append(c_tensor)

        if alm_manager is not None:
            penalties: dict[str, torch.Tensor] = alm_manager.compute_penalty(
                active_constraint_tensors, active_indices=active_constraint_indices
            )
            for key, t in penalties.items():
                grad_scale = norm_manager.get_grad_scale(key) if norm_manager else 1.0
                weighted_components[key] = t * grad_scale
                log_data[f"weighted_losses/{key}"] = t.item()
                log_data[f"optimizer_losses/{key}"] = weighted_components[key].detach().item()
        else:
            for category, t in bundle.constraints.items():
                name = self._constraint_by_category[category][0]
                w = 1.0 if calibration_pass else self._manual_weights.get(name, 1.0)
                combined_scale = (norm_manager.get_loss_scale(name) * norm_manager.get_grad_scale(name)) if norm_manager else 1.0
                wt = w * t * combined_scale
                weighted_components[name] = wt
                log_data[f"weighted_losses/{name}"] = wt.item()
                log_data[f"optimizer_losses/{name}"] = wt.item()

        # --- Auxiliaries (always manual) ----------------------------------
        for category, t in bundle.aux.items():
            name = self._aux_by_category[category][0]
            w = 1.0 if calibration_pass else self._manual_weights.get(name, 1.0)
            combined_scale = (norm_manager.get_loss_scale(name) * norm_manager.get_grad_scale(name)) if norm_manager else 1.0
            wt = w * t * combined_scale
            weighted_components[name] = wt
            log_data[f"weighted_losses/{name}"] = wt.item()
            log_data[f"optimizer_losses/{name}"] = wt.item()

        # --- Final scalar -------------------------------------------------
        if weighted_components:
            scalar: torch.Tensor = sum(weighted_components.values())  # type: ignore[assignment]
        else:
            scalar = torch.tensor(0.0, device=self._device)

        raw_total = 0.0
        for category, t in bundle.objectives.items():
            sign = float(self._objective_signs.get(category, 1))
            raw_total += sign * float(t.detach().item())
        for source in (bundle.constraints, bundle.aux):
            raw_total += sum(float(t.detach().item()) for t in source.values())
        for category, t in bundle.zero_weighted.items():
            sign = float(self._objective_signs.get(category, 1)) if category in self._objective_by_category else 1.0
            raw_total += sign * float(t.detach().item())
        log_data["raw_losses/total_loss"] = raw_total

        obj_sign: dict[str, float] = {
            self._objective_by_category[cat][0]: float(self._objective_signs.get(cat, 1)) for cat in self._objective_by_category
        }
        if self.use_cse:
            obj_sign["cse_objective"] = -1.0
        log_data["normalized_losses/total_loss"] = sum(
            obj_sign.get(k.removeprefix("normalized_losses/"), 1.0) * v
            for k, v in log_data.items()
            if k.startswith("normalized_losses/") and k != "normalized_losses/total_loss"
        )
        log_data["optimizer_losses/total_loss"] = scalar.detach().item()
        return_value = WeightedLossResult(
            scalar=scalar,
            log_data=log_data,
            components=weighted_components if return_components else {},
        )

        return return_value
