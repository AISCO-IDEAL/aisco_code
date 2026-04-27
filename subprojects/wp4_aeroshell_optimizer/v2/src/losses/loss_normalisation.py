class LossNormalisationManager:
    """Stores calibrated scale factors for losses and gradients.

    Calibration runs once before epoch 0 (via DiversityTrainer._calibrate()) and computes:
    - loss_scale per component: scales raw loss so its initial magnitude ≈ target_loss_value
    - grad_scale per component: scales the normalised loss so its gradient L2 norm ≈ target_gradient_norm

    Both scales are baked into the component tensors inside TotalLossV2.weight(), so the backward
    pass and optimizer step need no changes. When disabled, all getters return 1.0.

    Args:
        config: The loss_normalization block from general_info in the losses config.
    """

    def __init__(self, config: dict) -> None:
        self._enabled: bool = config.get("enabled", False)
        self._target_loss: float = float(config.get("target_loss_value", 1.0))
        self._target_grad: float = float(config.get("target_gradient_norm", 1.0))
        self._calibrated_names: set[str] = set()
        self._loss_scales: dict[str, float] = {}
        self._grad_scales: dict[str, float] = {}
        self._is_calibrated: bool = False

    def is_enabled(self) -> bool:
        return self._enabled

    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def get_loss_scale(self, name: str) -> float:
        """Returns the calibrated loss scale for a component, or 1.0 if not calibrated."""
        if not self._enabled or not self._is_calibrated:
            return 1.0
        return self._loss_scales.get(name, 1.0)

    def get_grad_scale(self, name: str) -> float:
        """Returns the calibrated gradient scale for a component, or 1.0 if not calibrated."""
        if not self._enabled or not self._is_calibrated:
            return 1.0
        return self._grad_scales.get(name, 1.0)

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
