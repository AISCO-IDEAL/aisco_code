"""Stores calibrated scale factors for losses and gradients.

Calibration runs once before epoch 0 and computes:
- loss_scale per component: scales raw loss so its initial magnitude ≈ target_loss_value
- grad_scale per component: scales the normalised loss so its gradient L2 norm ≈ target_gradient_norm

The manager also supports delayed recalibration rules for individual losses, so a loss can be
re-normalised later when it first becomes nonzero or on a periodic cadence.

When disabled, all getters return 1.0.

Args:
    config: The loss_normalization block from general_info in the losses config.
"""

from dataclasses import dataclass


@dataclass
class RecalibrationRule:
    """Trigger metadata for delayed loss recalibration."""

    loss_name: str
    min_abs_value: float = 0.0
    consecutive_epochs: int = 1
    warmup_epochs: int = 0
    cooldown_epochs: int = 0
    max_recalibrations: int | None = 1
    fixed_loss_scale: float | None = None
    fixed_grad_scale: float | None = None
    _current_streak: int = 0
    _last_recalibration_epoch: int | None = None
    _recalibration_count: int = 0

    @classmethod
    def from_config(cls, loss_name: str, config: dict) -> "RecalibrationRule":
        max_recalibrations_raw = config.get("max_recalibrations", 1)
        if max_recalibrations_raw in {None, "none", "null", "off"}:
            max_recalibrations: int | None = None
        else:
            max_recalibrations = max(1, int(max_recalibrations_raw))

        return cls(
            loss_name=loss_name,
            min_abs_value=float(config.get("min_abs_value", 0.0)),
            consecutive_epochs=max(1, int(config.get("consecutive_epochs", 1))),
            warmup_epochs=max(0, int(config.get("warmup_epochs", 0))),
            cooldown_epochs=max(0, int(config.get("cooldown_epochs", 0))),
            max_recalibrations=max_recalibrations,
            fixed_loss_scale=(float(config["fixed_loss_scale"]) if config.get("fixed_loss_scale") is not None else None),
            fixed_grad_scale=(float(config["fixed_grad_scale"]) if config.get("fixed_grad_scale") is not None else None),
        )

    def should_trigger(self, epoch: int, raw_value: float) -> bool:
        if epoch < self.warmup_epochs:
            return False
        if self.max_recalibrations is not None and self._recalibration_count >= self.max_recalibrations:
            return False

        if abs(raw_value) >= self.min_abs_value:
            self._current_streak += 1
        else:
            self._current_streak = 0

        if self._current_streak < self.consecutive_epochs:
            return False

        if self._last_recalibration_epoch is None:
            return True

        epoch_delta = epoch - self._last_recalibration_epoch
        return not (self.cooldown_epochs > 0 and epoch_delta < self.cooldown_epochs)

    def mark_recalibrated(self, epoch: int) -> None:
        self._last_recalibration_epoch = epoch
        self._recalibration_count += 1
        self._current_streak = 0


class LossNormalisationManager:
    def __init__(self, config: dict) -> None:
        self._enabled: bool = config.get("enabled", False)
        self._target_loss: float = float(config.get("target_loss_value", 1.0))
        self._target_grad: float = float(config.get("target_gradient_norm", 1.0))
        self._calibrated_names: set[str] = set()
        self._loss_scales: dict[str, float] = {}
        self._grad_scales: dict[str, float] = {}
        self._is_calibrated: bool = False
        self._recalibration_rules: dict[str, RecalibrationRule] = {}

        recalibration_rules = config.get("recalibration_rules", {})
        if isinstance(recalibration_rules, dict):
            for loss_name, rule_config in recalibration_rules.items():
                if isinstance(rule_config, dict):
                    self._recalibration_rules[loss_name] = RecalibrationRule.from_config(loss_name, rule_config)

    def is_enabled(self) -> bool:
        return self._enabled

    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def get_fixed_loss_scale(self, name: str) -> float | None:
        rule = self._recalibration_rules.get(name)
        if rule is None:
            return None
        return rule.fixed_loss_scale

    def get_fixed_grad_scale(self, name: str) -> float | None:
        rule = self._recalibration_rules.get(name) or self._recalibration_rules.get(name.removeprefix("alm_"))
        if rule is None:
            return None
        return rule.fixed_grad_scale

    def has_fixed_loss_scale(self, name: str) -> bool:
        return self.get_fixed_loss_scale(name) is not None

    def has_fixed_grad_scale(self, name: str) -> bool:
        return self.get_fixed_grad_scale(name) is not None

    def has_both_fixed_scales(self, name: str) -> bool:
        return self.has_fixed_loss_scale(name) and self.has_fixed_grad_scale(name)

    def get_loss_scale(self, name: str) -> float:
        """Returns the calibrated loss scale for a component, or 1.0 if not calibrated."""
        if not self._enabled or not self._is_calibrated:
            return 1.0
        return self._loss_scales.get(name, 1.0)

    def get_grad_scale(self, name: str) -> float:
        """Returns the calibrated gradient scale for a component, or 1.0 if not calibrated."""
        if not self._enabled or not self._is_calibrated:
            return 1.0
        if name in self._grad_scales:
            return self._grad_scales[name]
        fixed = self.get_fixed_grad_scale(name)
        return fixed if fixed is not None else 1.0

    def record_calibration(
        self,
        loss_scales: dict[str, float],
        grad_scales: dict[str, float],
        calibrated_names: set[str],
    ) -> None:
        self._loss_scales = loss_scales
        self._grad_scales = grad_scales
        self._calibrated_names = calibrated_names
        self._is_calibrated = True

    def get_telemetry(self) -> dict[str, float]:
        """Returns scale factors for wandb logging. Empty dict when disabled or uncalibrated."""
        if not self._enabled or not self._is_calibrated:
            return {}
        telemetry: dict[str, float] = {}
        for name, scale in self._loss_scales.items():
            telemetry[f"normalization/loss_scale_{name}"] = scale
        for name, scale in self._grad_scales.items():
            telemetry[f"normalization/grad_scale_{name}"] = scale
        return telemetry

    def get_recalibration_requests(self, epoch: int, raw_values: dict[str, float]) -> set[str]:
        """Return the configured loss names that should be recalibrated this epoch."""
        if not self._enabled or not self._is_calibrated or not self._recalibration_rules:
            return set()

        due: set[str] = set()
        for loss_name, rule in self._recalibration_rules.items():
            if rule.fixed_loss_scale is not None and rule.fixed_grad_scale is not None:
                continue
            raw_value = raw_values.get(loss_name)
            if raw_value is None:
                continue
            if rule.should_trigger(epoch, float(raw_value)):
                due.add(loss_name)
        return due

    def mark_recalibrated(self, loss_names: set[str], epoch: int) -> None:
        """Update the internal trigger state after a recalibration pass."""
        for loss_name in loss_names:
            rule = self._recalibration_rules.get(loss_name)
            if rule is not None:
                rule.mark_recalibrated(epoch)
