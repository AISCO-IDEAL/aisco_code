from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.weights import weights_alm
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.dynamic_loading import load_class_from_path


@dataclass
class DiversityLossComputeResult:
    raw: Tensor
    weighted: Tensor  # ALM penalty: λ·c_div + 0.5·μ·c_div² (or linear formula)
    c_div: Tensor  # constraint violation: max(0, D - max_div); zero when satisfied or no max_div
    active: bool = True  # False while inside warmup window


class DiversityLossManager:
    """Population-level diversity loss manager with ALM-based constraint enforcement.

    Owns diversity loss instantiation and treats the max_div saturation as an ALM
    inequality constraint (D ≤ max_div), maintaining its own dual variables (λ, μ, ν).
    This replaces the old fixed-weight + hard-clamp + diversity_grad_ratio approach.

    The constraint violation c_div = max(0, D - max_div) is zero when diversity is
    satisfied. When violated, the ALM penalty grows via dual-variable updates using
    the same weights_alm() formula as the geometric constraints in ALMManager.
    """

    def __init__(self, losses_config: dict[str, Any], device: str | torch.device = "cuda") -> None:
        self._device: torch.device = torch.device(device) if isinstance(device, str) else device
        self._losses_config: dict[str, Any] = losses_config

        self._enabled: bool = False
        self._loss_name: str = "diversity"
        self._loss_instance: Any | None = None

        # ALM dual variables (scalar tensors, shape (1,1) to match weights_alm expectations)
        self._lam: Tensor = torch.zeros(1, 1, device=self._device)
        self._mu: Tensor = torch.ones(1, 1, device=self._device)
        self._nu: Tensor = torch.ones(1, 1, device=self._device)
        self._alpha: Tensor = torch.ones(1, 1, device=self._device)

        self._max_div: float | None = None
        # Optional curriculum on max_div — see `max_div_schedule` config block.
        # When disabled (default), self._max_div is used unchanged for the whole run.
        self._max_div_schedule_enabled: bool = False
        self._max_div_schedule_end: float = 0.0
        self._max_div_schedule_start_epoch: int = 0
        self._max_div_schedule_end_epoch: int = 1
        self._max_div_schedule_mode: str = "linear"
        # Cached max_div from the most recent compute() call; consumed by build_log_data.
        self._last_max_div_used: float | None = None

        self._gamma: float = 0.01
        self._gamma_aggressive: float = 1.1
        self._alm_formula: str = "quadratic_augmented"

        self._init_from_config()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def loss_name(self) -> str:
        return self._loss_name

    def _init_from_config(self) -> None:  # noqa: PLR0912, PLR0915
        losses_block_raw: Any = self._losses_config.get("losses")
        if not isinstance(losses_block_raw, dict):
            return

        diversity_block_raw: Any = losses_block_raw.get("diversity")
        if diversity_block_raw is None:
            return
        if not isinstance(diversity_block_raw, dict):
            raise TypeError("Invalid losses config: 'losses.diversity' must be a dictionary block.")

        role = diversity_block_raw.get("_role")
        if role != "diversity":
            raise ValueError(
                "Invalid losses config: 'losses.diversity._role' must be set to 'diversity' when the block is present."
            )

        enabled_entries: list[tuple[str, str]] = []
        for loss_name, entry in diversity_block_raw.items():
            if loss_name.startswith("_"):
                continue
            if not isinstance(entry, list) or len(entry) < 3:
                raise ValueError(f"Invalid entry for 'diversity.{loss_name}': expected [compute, weight, path], got {entry!r}.")

            compute, _weight, path = entry[0], entry[1], entry[2]
            if isinstance(compute, bool):
                is_enabled = compute
            elif isinstance(compute, int):
                if compute not in {0, 1}:
                    raise ValueError(f"Invalid compute flag for 'diversity.{loss_name}': expected 0/1 or bool, got {compute!r}.")
                is_enabled = bool(compute)
            else:
                raise TypeError(f"Invalid compute flag for 'diversity.{loss_name}': expected 0/1 or bool, got {compute!r}.")

            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"Invalid path for 'diversity.{loss_name}': expected non-empty string, got {path!r}.")

            if is_enabled:
                enabled_entries.append((loss_name, path))

        if not enabled_entries:
            return
        if len(enabled_entries) > 1:
            enabled_names = [name for name, _path in enabled_entries]
            raise ValueError(
                f"Multiple enabled diversity losses: {enabled_names}. Set exactly one entry to compute=1 in losses.diversity."
            )

        self._loss_name, path = enabled_entries[0]
        abs_path = Path.cwd() / Path(path)
        family: str = self._loss_name.title().replace("_", "")
        loss_cls = load_class_from_path(str(abs_path), family)
        self._loss_instance = loss_cls(
            device=str(self._device),
            loss_name=self._loss_name,
            losses_config=self._losses_config,
        )
        self._enabled = True

        # Read ALM params from the loss's own config block (e.g. field_diversity_surface_v0_config)
        loss_cfg: dict[str, Any] = self._losses_config.get(f"{self._loss_name}_config", {}) or {}

        max_div_raw = loss_cfg.get("max_div")
        if max_div_raw is not None and str(max_div_raw).strip().lower() not in {"none", "null"}:
            self._max_div = float(max_div_raw)

        schedule_cfg = loss_cfg.get("max_div_schedule") or {}
        if isinstance(schedule_cfg, dict) and bool(schedule_cfg.get("enabled", False)):
            if self._max_div is None:
                raise ValueError("max_div_schedule.enabled=true requires max_div to be set (it's the starting value).")
            if "end_value" not in schedule_cfg:
                raise ValueError("max_div_schedule.enabled=true requires 'end_value'.")
            self._max_div_schedule_end = float(schedule_cfg["end_value"])
            self._max_div_schedule_start_epoch = int(schedule_cfg.get("start_epoch", 0))
            self._max_div_schedule_end_epoch = int(schedule_cfg.get("end_epoch", 0))
            if self._max_div_schedule_end_epoch <= self._max_div_schedule_start_epoch:
                raise ValueError(
                    f"max_div_schedule: end_epoch ({self._max_div_schedule_end_epoch}) must be > "
                    f"start_epoch ({self._max_div_schedule_start_epoch})."
                )
            mode = str(schedule_cfg.get("mode", "linear")).lower()
            if mode not in ("linear", "smoothstep"):
                raise ValueError(f"max_div_schedule.mode '{mode}' not recognised. Expected 'linear' or 'smoothstep'.")
            self._max_div_schedule_mode = mode
            self._max_div_schedule_enabled = True

        self._last_max_div_used = self._max_div

        initial_lambda = float(loss_cfg.get("initial_lambda", 0.01))
        initial_mu = float(loss_cfg.get("initial_mu", 1.0))
        initial_nu = float(loss_cfg.get("initial_nu", 1.0))
        alpha = float(loss_cfg.get("alpha", 0.99))
        self._gamma = float(loss_cfg.get("gamma", 0.01))
        self._gamma_aggressive = float(loss_cfg.get("gamma_aggressive", 1.1))

        alm_formula = str(loss_cfg.get("alm_formula", "quadratic_augmented"))
        if alm_formula not in ("quadratic_augmented", "linear_augmented"):
            raise ValueError(
                f"alm_formula '{alm_formula}' not recognised for diversity. Expected 'quadratic_augmented' or 'linear_augmented'."
            )
        self._alm_formula = alm_formula

        self._lam = torch.tensor([[initial_lambda]], device=self._device, dtype=torch.float32)
        self._mu = torch.tensor([[initial_mu]], device=self._device, dtype=torch.float32)
        self._nu = torch.tensor([[initial_nu]], device=self._device, dtype=torch.float32)
        self._alpha = torch.tensor([[alpha]], device=self._device, dtype=torch.float32)

    def should_compute(self, population_size: int) -> bool:
        return self._enabled and self._loss_instance is not None and population_size > 1

    def _current_max_div(self, epoch: int) -> float | None:
        """Resolve the active max_div for this epoch, applying the schedule if enabled.

        Returns the scalar `self._max_div` unchanged when the schedule is disabled,
        when max_div is null, or for any `epoch <= start_epoch` (including the -1
        sentinel from compute's default arg). Linear/smoothstep interpolation between
        start and end_value across [start_epoch, end_epoch], then held at end_value.
        """
        if self._max_div is None or not self._max_div_schedule_enabled:
            return self._max_div
        if epoch < 0 or epoch <= self._max_div_schedule_start_epoch:
            return self._max_div
        if epoch >= self._max_div_schedule_end_epoch:
            return self._max_div_schedule_end
        span = self._max_div_schedule_end_epoch - self._max_div_schedule_start_epoch
        t = (epoch - self._max_div_schedule_start_epoch) / span
        if self._max_div_schedule_mode == "smoothstep":
            t = t * t * (3.0 - 2.0 * t)
        return self._max_div + t * (self._max_div_schedule_end - self._max_div)

    def compute(
        self,
        model: SupportsLatentConcat,
        Z: Tensor,
        num_points: int,
        bbox_lo: list[float] | None = None,
        bbox_hi: list[float] | None = None,
        norm_order: int = 2,
        neighbor_agg_fn: str = "min",
        epoch: int = -1,
        leinster_temp: float | None = None,
        leinster_q: float | None = None,
    ) -> DiversityLossComputeResult:
        """Compute diversity loss and ALM penalty.

        Returns:
            DiversityLossComputeResult with:
                raw: raw Leinster diversity D (negative, more negative = more diverse)
                c_div: constraint violation max(0, D - max_div); zero when diversity is satisfied
                weighted: ALM penalty λ·c_div + 0.5·μ·c_div² (the tensor to backward through)
                active: False inside the warmup window (no backward should fire)
        """
        zero = torch.zeros((), device=Z.device, dtype=torch.float32)
        if self._loss_instance is None:
            return DiversityLossComputeResult(raw=zero, weighted=zero, c_div=zero)

        raw_loss = self._loss_instance.get_loss(
            model,
            Z,
            num_points=num_points,
            bbox_lo=bbox_lo,
            bbox_hi=bbox_hi,
            norm_order=norm_order,
            neighbor_agg_fn=neighbor_agg_fn,
            epoch=epoch,
            leinster_temp=leinster_temp,
            leinster_q=leinster_q,
        )
        if raw_loss is None:
            return DiversityLossComputeResult(raw=zero, weighted=zero, c_div=zero, active=False)

        current_max_div = self._current_max_div(epoch)
        self._last_max_div_used = current_max_div
        if current_max_div is not None:  # noqa: SIM108 weird readabilitly rule
            c_div = torch.clamp(raw_loss - current_max_div, min=0.0)
        else:
            # No saturation threshold — treat the raw loss directly as the violation
            # (pushes diversity without bound; useful for early training or debugging)
            c_div = raw_loss

        lam = self._lam.detach().squeeze()
        mu = self._mu.detach().squeeze()
        if self._alm_formula == "linear_augmented":  # noqa: SIM108 weird readabilitly rule
            alm_penalty = (lam + 0.5 * mu) * c_div
        else:
            alm_penalty = lam * c_div + 0.5 * mu * c_div**2

        return DiversityLossComputeResult(raw=raw_loss, weighted=alm_penalty, c_div=c_div)

    def update_multipliers(self, c_div: Tensor) -> None:
        """Update ALM dual variables based on the observed constraint violation.

        Should be called with c_div.detach() after the Phase-2 backward.
        When c_div = 0 (diversity satisfied), multipliers are frozen.
        """
        weights_alm(
            lamdas=self._lam,
            mus=self._mu,
            nus=self._nu,
            alphas=self._alpha,
            gamma=self._gamma,
            gamma_aggressive=self._gamma_aggressive,
            constrained_losses=c_div.detach().view(1),
            use_alm_aggressive=False,
        )

    def build_log_data(self, raw_loss_value: float) -> dict[str, float]:
        lam_val = self._lam.item()
        mu_val = self._mu.item()
        nu_val = self._nu.item()

        active_max_div = self._last_max_div_used if self._last_max_div_used is not None else self._max_div
        c_div_val = max(0.0, raw_loss_value - active_max_div) if active_max_div is not None else raw_loss_value
        if self._alm_formula == "linear_augmented":
            weighted_val = (lam_val + 0.5 * mu_val) * c_div_val
        else:
            weighted_val = lam_val * c_div_val + 0.5 * mu_val * c_div_val**2

        return {
            f"raw_losses/{self._loss_name}": raw_loss_value,
            f"normalized_losses/{self._loss_name}": raw_loss_value,
            f"weighted_losses/{self._loss_name}": weighted_val,
            f"optimizer_losses/{self._loss_name}": weighted_val,
            "diversity/loss": raw_loss_value,
            "diversity/weighted_loss": weighted_val,
            "diversity/c_div": c_div_val,
            "diversity/max_div_current": active_max_div if active_max_div is not None else 0.0,
            f"alm_params/lambda_{self._loss_name}": lam_val,
            f"alm_params/mu_{self._loss_name}": mu_val,
            f"alm_params/nu_{self._loss_name}": nu_val,
        }
