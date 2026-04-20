from __future__ import annotations

from typing import Any

import cripser
import numpy as np
import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss


class ConnectednessV1(ParentLoss):
    def _setup(self) -> None:
        connectedness_config: dict[str, Any] = self.losses_config.get("connectedness_config", {})
        get_points_config: dict[str, Any] = self.losses_config.get("get_points_params", {})

        self.grid_dim: int = int(connectedness_config.get("grid_dim", 3))
        self.grid_resolution: int = int(connectedness_config.get("grid_resolution", 40))
        self.iso_level: float = float(connectedness_config.get("iso_level", 0.0))
        self.target_betti_dim0: int = int(connectedness_config.get("target_betti_dim0", 1))
        self.maxdim: int = int(connectedness_config.get("maxdim", 0))
        self.warmup_epochs: int = int(connectedness_config.get("warmup_epochs", 50))
        self.eval_period: int = max(1, int(connectedness_config.get("eval_period", 5)))
        self.return_cached_on_skip: bool = self._as_bool(connectedness_config.get("return_cached_on_skip", "False"))

        default_bbox_size: float = float(get_points_config.get("bbox_size", 10.0))
        default_bbox: list[list[float]] = [[-0.5 * default_bbox_size, 0.5 * default_bbox_size] for _ in range(self.grid_dim)]
        bbox: list[list[float]] = connectedness_config.get("bbox", default_bbox)
        if len(bbox) != self.grid_dim:
            raise ValueError(
                f"connectedness_config.bbox must have {self.grid_dim} entries, got {len(bbox)}."
            )

        self.grid_shape: tuple[int, ...] = (self.grid_resolution,) * self.grid_dim
        axes = [
            torch.linspace(float(axis_bounds[0]), float(axis_bounds[1]), self.grid_resolution, device=self.device)
            for axis_bounds in bbox
        ]
        mesh = torch.meshgrid(*axes, indexing="ij")
        self.grid_points: torch.Tensor = torch.stack(mesh, dim=-1).reshape(-1, self.grid_dim)

        self._cached_loss: torch.Tensor | None = None

    def _loss(self, model) -> torch.Tensor:
        model_device = next(iter(model.parameters())).device
        if self.grid_points.device != model_device:
            self.grid_points = self.grid_points.to(model_device)

        epoch_name: int = int(getattr(self, "epoch_name", -1))
        if epoch_name < self.warmup_epochs:
            return self._inactive_loss()

        if (epoch_name - self.warmup_epochs) % self.eval_period != 0:
            return self._inactive_loss()

        with torch.no_grad():
            model_out = model(self.grid_points)
            if isinstance(model_out, tuple):
                model_out = model_out[0]
            field_flat = model_out.reshape(-1)
            field_np = field_flat.detach().cpu().numpy().reshape(self.grid_shape)

        ph: np.ndarray = cripser.computePH(field_np, maxdim=self.maxdim)
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

        loss = torch.clamp(self.iso_level - model_values, max=0.0).pow(2).sum()
        self._cached_loss = loss.detach()
        return loss

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
        if ph_dim0.shape[0] <= self.target_betti_dim0:
            return np.empty((0,), dtype=np.int64)

        extra_components = ph_dim0[self.target_betti_dim0 :]
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
