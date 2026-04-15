"""Clean loss pipeline for non-legacy training loops (single- or population-based).

Responsibilities are strictly limited to loss computation and weighting:

* ``compute(model)``                                      - evaluate all active loss instances
* ``weight(bundle, alm_manager=None)`` - CSE/manual objectives + ALM/manual constraints + manual aux

NO wandb, NO mesh extraction, NO matplotlib, NO ALM state here. The caller (e.g. DiversityTrainer)
owns logging, rendering, ALM multiplier updates, and any other telemetry.

Role classification is driven from YAML via a ``_role`` metadata key per category block
(``objective``, ``constraint``, ``aux``). ``_sign: maximize`` flips the sign of an objective
in manual (non-CSE) mode. Keys starting with ``_`` are metadata and never loaded as losses.
Constraint ordering for ALM follows the YAML category order.
"""

from __future__ import annotations

import inspect
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.alm import ALMManager
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.cruising_speed import cruising_speed
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsModelContract
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.dynamic_loading import load_class_from_path

TrainModel: TypeAlias = SupportsModelContract

_ROLE_OBJECTIVE = "objective"
_ROLE_CONSTRAINT = "constraint"
_ROLE_AUX = "aux"
_VALID_ROLES = frozenset({_ROLE_OBJECTIVE, _ROLE_CONSTRAINT, _ROLE_AUX})
_METADATA_PREFIX = "_"


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
    constraints: OrderedDict[str, torch.Tensor] = field(default_factory=OrderedDict)
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
        self._use_cse: bool = gi.get("CSE") == "True"

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

    def _init_active_losses(self) -> None:  # noqa: PLR0912
        """Instantiate the active loss of every category and classify by ``_role``."""
        project_root: Path = Path.cwd()
        losses_block: dict[str, Any] = self.losses_config.get("losses", {})

        # Category -> (loss_name, instance); OrderedDict preserves YAML order for ALM alignment
        self._objective_by_category: OrderedDict[str, tuple[str, Any]] = OrderedDict()
        self._constraint_by_category: OrderedDict[str, tuple[str, Any]] = OrderedDict()
        self._aux_by_category: OrderedDict[str, tuple[str, Any]] = OrderedDict()
        self._objective_signs: dict[str, int] = {}  # only used in manual (non-CSE) objective mode
        self._manual_weights: dict[str, float] = {}  # loss_name -> weight from YAML [compute, weight, path]

        for category, block in losses_block.items():
            if category == "total":
                continue
            if not isinstance(block, dict):
                continue

            role: str | None = block.get("_role")
            if role is None:
                continue  # category without _role: silently skip (legacy/commented blocks)
            if role not in _VALID_ROLES:
                raise ValueError(f"Invalid _role '{role}' for category '{category}'. Expected one of {sorted(_VALID_ROLES)}.")

            sign_val: int = -1 if block.get("_sign", "minimize") == "maximize" else 1

            # Find the single active loss in this category (first entry with compute=1)
            active_entry: tuple[str, Any] | None = None
            for loss_name, entry in block.items():
                if loss_name.startswith(_METADATA_PREFIX):
                    continue
                if not isinstance(entry, list) or len(entry) < 3:
                    continue
                compute, weight, path = entry[0], entry[1], entry[2]
                if not compute:
                    continue

                abs_path: Path = project_root / Path(path)
                family: str = Path(path).stem.title().replace("_", "")
                loss_cls = load_class_from_path(str(abs_path), family)
                instance = loss_cls(
                    device=str(self._device),
                    loss_name=loss_name,
                    losses_config=self.losses_config,
                )
                self._manual_weights[loss_name] = float(weight)
                active_entry = (loss_name, instance)
                break

            if active_entry is None:
                continue

            if role == _ROLE_OBJECTIVE:
                self._objective_by_category[category] = active_entry
                self._objective_signs[category] = sign_val
            elif role == _ROLE_CONSTRAINT:
                self._constraint_by_category[category] = active_entry
            else:  # aux
                self._aux_by_category[category] = active_entry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def use_cse(self) -> bool:
        return self._use_cse

    @property
    def constraint_names(self) -> list[str]:
        """Active constraint loss names, in YAML category order (= ALM vector order)."""
        return [name for (name, _inst) in self._constraint_by_category.values()]

    @property
    def constraint_category_order(self) -> list[str]:
        """Category names for active constraints, in YAML order."""
        return list(self._constraint_by_category.keys())

    def compute(self, model: TrainModel, epoch_name: int = 0) -> RawLossBundle:
        """Evaluate every active loss instance and return raw (unweighted) tensors."""
        bundle = RawLossBundle()
        for category, (_loss_name, instance) in self._objective_by_category.items():
            bundle.objectives[category] = self._call_loss(instance, model, epoch_name)
        for category, (_loss_name, instance) in self._constraint_by_category.items():
            bundle.constraints[category] = self._call_loss(instance, model, epoch_name)
        for category, (_loss_name, instance) in self._aux_by_category.items():
            bundle.aux[category] = self._call_loss(instance, model, epoch_name)
        return bundle

    @staticmethod
    def _call_loss(instance: Any, model: TrainModel, epoch_name: int) -> torch.Tensor:
        sig = inspect.signature(instance.get_loss)
        if "skip_telemetry" in sig.parameters:
            return instance.get_loss(model, epoch_name, skip_telemetry=True)
        return instance.get_loss(model, epoch_name)

    def weight(  # noqa: PLR0912
        self,
        bundle: RawLossBundle,
        alm_manager: ALMManager | None = None,
    ) -> WeightedLossResult:
        """Combine raw losses into a backpropable scalar plus a detached log dict.

        * Objectives: CSE (drag+solar -> cruising speed) if ``self.use_cse``, else manual weights with _sign.
        * Constraints: ALM ``lambda*c + 0.5*mu*c^2`` if ``alm_manager`` is given, else manual weights.
        * Auxiliaries: always manual weights.

        ALM multiplier updates are NOT performed here — the caller owns that lifecycle.
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
        if self._use_cse:
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
