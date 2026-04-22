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

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.alm import ALMManager
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.cruising_speed import cruising_speed
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

    ``side_data`` is a forward-compat slot: individual losses may write extra
    rendering/telemetry data here. TotalLossV2 does not populate or read it today;
    the caller decides whether to consume it.
    """

    objectives: dict[str, torch.Tensor] = field(default_factory=dict)
    constraints: dict[str, torch.Tensor] = field(default_factory=dict)
    aux: dict[str, torch.Tensor] = field(default_factory=dict)
    side_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightedLossResult:
    """Output of ``TotalLossV2.weight``.

    ``scalar`` keeps the live autograd graph for backward(). ``log_data`` holds
    detached (host-side) floats suitable for averaging across a population and
    feeding directly into wandb.log.
    """

    scalar: torch.Tensor
    log_data: dict[str, float] = field(default_factory=dict)


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

        # CSE physical constants — defaults match historical hardcoded values
        cse_params: dict[str, Any] = gi.get("cse_params", {})
        self._cse_drag_category: str = cse_params.get("drag_category", "aero")
        self._cse_solar_category: str = cse_params.get("solar_category", "solar")
        self._cse_mass: float = float(cse_params.get("mass", 210))
        self._cse_c_losses: float = float(cse_params.get("c_losses", 20))
        self._cse_rolling_friction: float = float(cse_params.get("rolling_friction_coefficient", 0.006))
        self._cse_max_iter: int = int(cse_params.get("max_iter", 12))
        self._cse_tol: float = float(cse_params.get("tol", 1e-4))
        self._cse_prefactors: dict[str, float] = gi.get("cse_prefactors", {})

        self._init_active_losses()

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
            family: str = Path(path).stem.title().replace("_", "")
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

    def compute(self, model: SupportsModelContract, epoch_name: int = 0) -> RawLossBundle:
        """Evaluate every active loss instance and return raw (unweighted) tensors.

        Args:
            model: The trainable model passed to each active loss instance.
            epoch_name: Training step/epoch identifier forwarded to each loss call.

        Returns:
            RawLossBundle with unweighted objective, constraint, and aux tensors
            grouped by category.
        """
        self._inject_shared_meshes(model)
        bundle = RawLossBundle()
        try:
            for category, (_loss_name, instance) in self._objective_by_category.items():
                sig = inspect.signature(instance.get_loss)
                if "skip_telemetry" in sig.parameters:
                    bundle.objectives[category] = instance.get_loss(model, epoch_name, skip_telemetry=True)
                else:
                    bundle.objectives[category] = instance.get_loss(model, epoch_name)
            for category, (_loss_name, instance) in self._constraint_by_category.items():
                sig = inspect.signature(instance.get_loss)
                if "skip_telemetry" in sig.parameters:
                    bundle.constraints[category] = instance.get_loss(model, epoch_name, skip_telemetry=True)
                else:
                    bundle.constraints[category] = instance.get_loss(model, epoch_name)
            for category, (_loss_name, instance) in self._aux_by_category.items():
                sig = inspect.signature(instance.get_loss)
                if "skip_telemetry" in sig.parameters:
                    bundle.aux[category] = instance.get_loss(model, epoch_name, skip_telemetry=True)
                else:
                    bundle.aux[category] = instance.get_loss(model, epoch_name)
        finally:
            self._clear_shared_meshes()
        return bundle

    def weight(  # noqa: PLR0912
        self,
        bundle: RawLossBundle,
        alm_manager: ALMManager | None = None,
    ) -> WeightedLossResult:
        """Combine raw losses into a backpropable scalar plus a detached log dict.

        * Objectives: CSE (drag+solar -> cruising speed) if ``self.use_cse``, else manual weights with _sign.
        * Constraints: ALM ``lambda*c + 0.5*mu*c^2`` if ``alm_manager`` is given, else manual weights.
        * Auxiliaries: always manual weights.

        Args:
            bundle: Raw unweighted losses grouped by role/category from `compute`.
            alm_manager: Optional ALM manager for constraint penalty weighting.

        Returns:
            WeightedLossResult where `scalar` is used for backward and `log_data`
            contains detached numeric values for telemetry.

        ALM multiplier updates are NOT performed here; the caller owns that lifecycle.
        """
        log_data: dict[str, float] = {}
        weighted_components: dict[str, torch.Tensor] = {}

        # Raw logging (all active losses, by loss_name)
        for category, t in bundle.objectives.items():
            log_data[f"raw_losses/{self._objective_by_category[category][0]}"] = t.item()
        for category, t in bundle.constraints.items():
            log_data[f"raw_losses/{self._constraint_by_category[category][0]}"] = t.item()
        for category, t in bundle.aux.items():
            log_data[f"raw_losses/{self._aux_by_category[category][0]}"] = t.item()

        # --- Objectives ---------------------------------------------------
        if self.use_cse:
            if self._cse_drag_category not in bundle.objectives:
                raise KeyError(
                    f"CSE active but drag category '{self._cse_drag_category}' is not an active objective. "
                    f"Available objective categories: {list(bundle.objectives.keys())}"
                )
            if self._cse_solar_category not in bundle.objectives:
                raise KeyError(
                    f"CSE active but solar category '{self._cse_solar_category}' is not an active objective. "
                    f"Available objective categories: {list(bundle.objectives.keys())}"
                )
            drag_pf: float = float(self._cse_prefactors.get("drag", 1.0))
            solar_pf: float = float(self._cse_prefactors.get("solar", 1.0))
            drag: torch.Tensor = bundle.objectives[self._cse_drag_category] * drag_pf
            solar: torch.Tensor = bundle.objectives[self._cse_solar_category] * solar_pf

            v_cruise: torch.Tensor = cruising_speed(
                solar_power=solar,
                drag_force_at_25ms=drag,
                mass=self._cse_mass,
                c_losses=self._cse_c_losses,
                rolling_friction_coefficient=self._cse_rolling_friction,
                max_iter=self._cse_max_iter,
                tol=self._cse_tol,
            )
            # Minimize -v_cruise to maximize physical speed
            weighted_components["cse_objective"] = -v_cruise
            # Log positive v_cruise for readability (matches legacy dashboards)
            log_data["cruising_speed_ms"] = v_cruise.item()
            log_data["weighted_losses/cse_objective"] = v_cruise.item()
        else:
            for category, t in bundle.objectives.items():
                name: str = self._objective_by_category[category][0]
                w: float = self._manual_weights.get(name, 1.0)
                sign: float = float(self._objective_signs.get(category, 1))
                wt: torch.Tensor = sign * w * t
                weighted_components[name] = wt
                log_data[f"weighted_losses/{name}"] = wt.item()

        # --- Constraints --------------------------------------------------
        if alm_manager is not None:
            # Order matches alm_manager.constraint_names (both come from YAML order)
            constraint_tensors: list[torch.Tensor] = list(bundle.constraints.values())
            penalties: dict[str, torch.Tensor] = alm_manager.compute_penalty(constraint_tensors)
            for key, t in penalties.items():
                weighted_components[key] = t
                log_data[f"weighted_losses/{key}"] = t.item()
        else:
            for category, t in bundle.constraints.items():
                name = self._constraint_by_category[category][0]
                w = self._manual_weights.get(name, 1.0)
                wt = w * t
                weighted_components[name] = wt
                log_data[f"weighted_losses/{name}"] = wt.item()

        # --- Auxiliaries (always manual) ----------------------------------
        for category, t in bundle.aux.items():
            name = self._aux_by_category[category][0]
            w = self._manual_weights.get(name, 1.0)
            wt = w * t
            weighted_components[name] = wt
            log_data[f"weighted_losses/{name}"] = wt.item()

        # --- Final scalar -------------------------------------------------
        if weighted_components:
            scalar: torch.Tensor = sum(weighted_components.values())  # type: ignore[assignment]
        else:
            scalar = torch.tensor(0.0, device=self._device)
        log_data["raw_losses/total_loss"] = scalar.item()

        return WeightedLossResult(scalar=scalar, log_data=log_data)
