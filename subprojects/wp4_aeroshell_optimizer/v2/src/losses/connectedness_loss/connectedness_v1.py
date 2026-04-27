from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss


class ConnectednessV1(ParentLoss):
    def _setup(self) -> None:
        try:
            self._cripser: Any = importlib.import_module("cripser")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ConnectednessV1 requires the optional Python package 'cripser'. "
                "Either install 'cripser' in the active environment or disable "
                "losses.connectedness.connectedness_v1 in "
                "subprojects/wp4_aeroshell_optimizer/v2/configs/losses_v0_config.yaml."
            ) from exc

        connectedness_config: dict[str, Any] = self.losses_config.get("connectedness_config", {})
        get_points_config: dict[str, Any] = self.losses_config.get("get_points_params", {})

        self.grid_dim: int = int(connectedness_config.get("grid_dim", 3))
        self.grid_resolution: int = int(connectedness_config.get("grid_resolution", 40))
        self.iso_level: float = float(connectedness_config.get("iso_level", 0.0))
        self.target_betti_dim0: int = int(connectedness_config.get("target_betti_dim0", 1))
        self.maxdim: int = int(connectedness_config.get("maxdim", 0))
        self.eval_period: int = max(1, int(connectedness_config.get("eval_period", 5)))
        self.return_cached_on_skip: bool = self._as_bool(connectedness_config.get("return_cached_on_skip", "False"))
        self.persistence_min_lifetime: float = float(connectedness_config.get("persistence_min_lifetime", 0.0))
        self.max_extra_components: int = max(0, int(connectedness_config.get("max_extra_components", 0)))
        self.penalty_mode: str = str(connectedness_config.get("penalty_mode", "quadratic")).strip().lower()
        self.huber_delta: float = float(connectedness_config.get("huber_delta", 0.1))
        self.normalize_mode: str = str(connectedness_config.get("normalize_mode", "sum")).strip().lower()
        self.hysteresis_enabled: bool = self._as_bool(connectedness_config.get("hysteresis_enabled", "False"))
        self.success_patience_evals: int = max(1, int(connectedness_config.get("success_patience_evals", 3)))
        self.freeze_loss_threshold: float = float(connectedness_config.get("freeze_loss_threshold", 1.0e-4))
        self.reactivate_loss_threshold: float = float(connectedness_config.get("reactivate_loss_threshold", 5.0e-4))

        violation_cap_raw = connectedness_config.get("violation_cap")
        if violation_cap_raw is None or (
            isinstance(violation_cap_raw, str) and violation_cap_raw.strip().lower() in {"none", "null", "off"}
        ):
            self.violation_cap = None
        else:
            self.violation_cap = float(violation_cap_raw)

        self.ph_smoothing_kernel: int = max(1, int(connectedness_config.get("ph_smoothing_kernel", 1)))
        if self.ph_smoothing_kernel % 2 == 0:
            self.ph_smoothing_kernel += 1

        default_bbox_size: float = float(get_points_config.get("bbox_size", 10.0))
        default_bbox: list[list[float]] = [[-0.5 * default_bbox_size, 0.5 * default_bbox_size] for _ in range(self.grid_dim)]
        bbox: list[list[float]] = connectedness_config.get("bbox", default_bbox)
        if len(bbox) != self.grid_dim:
            raise ValueError(f"connectedness_config.bbox must have {self.grid_dim} entries, got {len(bbox)}.")

        self.grid_shape: tuple[int, ...] = (self.grid_resolution,) * self.grid_dim
        axes = [
            torch.linspace(float(axis_bounds[0]), float(axis_bounds[1]), self.grid_resolution, device=self.device)
            for axis_bounds in bbox
        ]
        mesh = torch.meshgrid(*axes, indexing="ij")
        self.grid_points: torch.Tensor = torch.stack(mesh, dim=-1).reshape(-1, self.grid_dim)

        self._cached_loss: torch.Tensor | None = None
        self._is_frozen: bool = False
        self._success_streak_evals: int = 0

    def _loss(self, model) -> torch.Tensor:  # noqa: PLR0912
        model_device = next(iter(model.parameters())).device
        if self.grid_points.device != model_device:
            self.grid_points = self.grid_points.to(model_device)

        epoch_name: int = int(getattr(self, "epoch_name", -1))
        if epoch_name % self.eval_period != 0:
            return self._inactive_loss()

        with torch.no_grad():
            model_out = model(self.grid_points)
            if isinstance(model_out, tuple):
                model_out = model_out[0]
            field_flat = model_out.reshape(-1)
            field_for_ph = field_flat
            if self.grid_dim == 3 and self.ph_smoothing_kernel > 1:
                grid_t = field_flat.reshape(1, 1, *self.grid_shape)
                grid_t = torch.nn.functional.avg_pool3d(
                    grid_t,
                    kernel_size=self.ph_smoothing_kernel,
                    stride=1,
                    padding=self.ph_smoothing_kernel // 2,
                )
                field_for_ph = grid_t.reshape(-1)
            field_np = field_for_ph.detach().cpu().numpy().reshape(self.grid_shape)

        ph: np.ndarray = self._cripser.computePH(field_np, maxdim=self.maxdim)
        critical_linear_indices = self._extract_dim0_death_indices(ph)
        if critical_linear_indices.size == 0:
            self._cached_loss = torch.tensor(0.0, device=self.grid_points.device)
            return self._cached_loss

        critical_idx_t = torch.from_numpy(critical_linear_indices).to(device=self.grid_points.device, dtype=torch.long)
        critical_points = self.grid_points[critical_idx_t]

        model_values = model(critical_points)
        if isinstance(model_values, tuple):
            model_values = model_values[0]
        model_values = model_values.reshape(-1)

        loss = self._build_penalty(model_values)

        loss_val = float(loss.detach().item())
        if self.hysteresis_enabled:
            if self._is_frozen:
                if loss_val >= self.reactivate_loss_threshold:
                    self._is_frozen = False
                    self._success_streak_evals = 0
                else:
                    loss = torch.tensor(0.0, device=self.grid_points.device)
            if not self._is_frozen:
                if loss_val <= self.freeze_loss_threshold:
                    self._success_streak_evals += 1
                    if self._success_streak_evals >= self.success_patience_evals:
                        self._is_frozen = True
                        loss = torch.tensor(0.0, device=self.grid_points.device)
                else:
                    self._success_streak_evals = 0

        self._cached_loss = loss.detach()
        return loss

    def _build_penalty(self, model_values: torch.Tensor) -> torch.Tensor:
        violation = torch.clamp(model_values - self.iso_level, min=0.0)
        if self.violation_cap is not None:
            violation = torch.clamp(violation, max=self.violation_cap)

        if self.penalty_mode == "huber":
            abs_v = violation.abs()
            delta = max(self.huber_delta, 1.0e-12)
            quad = torch.minimum(abs_v, torch.tensor(delta, device=violation.device, dtype=violation.dtype))
            linear = abs_v - quad
            loss_vec = 0.5 * quad.pow(2) + delta * linear
        else:
            loss_vec = violation.pow(2)

        if loss_vec.numel() == 0:
            return torch.tensor(0.0, device=self.grid_points.device)
        if self.normalize_mode == "mean":
            return loss_vec.mean()
        if self.normalize_mode in {"sqrt", "sum_div_sqrt_n"}:
            return loss_vec.sum() / torch.sqrt(torch.tensor(float(loss_vec.numel()), device=loss_vec.device))
        return loss_vec.sum()

    def _extract_dim0_death_indices(self, ph: np.ndarray) -> np.ndarray:
        if ph.size == 0:
            return np.empty((0,), dtype=np.int64)

        ph_dim0 = ph[ph[:, 0] == 0]
        if ph_dim0.size == 0:
            return np.empty((0,), dtype=np.int64)

        lifetime = ph_dim0[:, 2] - ph_dim0[:, 1]
        sorting_idx = np.argsort(lifetime)[::-1]
        ph_dim0 = ph_dim0[sorting_idx]

        crosses_iso = (ph_dim0[:, 1] < -self.iso_level) & (ph_dim0[:, 2] > self.iso_level)
        ph_dim0 = ph_dim0[crosses_iso]
        if ph_dim0.size == 0:
            return np.empty((0,), dtype=np.int64)

        if self.persistence_min_lifetime > 0.0:
            lifetime_iso = ph_dim0[:, 2] - ph_dim0[:, 1]
            ph_dim0 = ph_dim0[lifetime_iso >= self.persistence_min_lifetime]
            if ph_dim0.size == 0:
                return np.empty((0,), dtype=np.int64)

        if ph_dim0.shape[0] <= self.target_betti_dim0:
            return np.empty((0,), dtype=np.int64)

        extra_components = ph_dim0[self.target_betti_dim0 :]
        if self.max_extra_components > 0:
            extra_components = extra_components[: self.max_extra_components]
        death_coords = extra_components[:, 6 : 6 + self.grid_dim].astype(np.int64)
        death_coords = np.clip(death_coords, 0, self.grid_resolution - 1)

        linear_indices = np.ravel_multi_index(death_coords.T, self.grid_shape)
        linear_indices = np.unique(linear_indices)
        return linear_indices.astype(np.int64)

    def _inactive_loss(self) -> torch.Tensor:
        if self.return_cached_on_skip and self._cached_loss is not None:
            return self._cached_loss
        return torch.tensor(0.0, device=self.grid_points.device)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
