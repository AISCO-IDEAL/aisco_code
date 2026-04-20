from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.dynamic_loading import load_class_from_path


@dataclass
class DiversityLossComputeResult:
    raw: Tensor
    weighted: Tensor


class DiversityLossManager:
    """Population-level diversity loss manager.

    Owns diversity loss instantiation from losses config and applies fixed weighting.
    Runtime sampling parameters are provided by the caller at compute time.
    """

    def __init__(self, losses_config: dict[str, Any], device: str | torch.device = "cuda") -> None:
        self._device: torch.device = torch.device(device) if isinstance(device, str) else device
        self._losses_config: dict[str, Any] = losses_config

        self._enabled: bool = False
        self._weight: float = 0.0
        self._loss_name: str = "diversity"
        self._loss_instance: Any | None = None

        self._init_from_config()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def loss_name(self) -> str:
        return self._loss_name

    def _init_from_config(self) -> None:  # noqa: PLR0912
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

        enabled_entries: list[tuple[str, float, str]] = []
        for loss_name, entry in diversity_block_raw.items():
            if loss_name.startswith("_"):
                continue
            if not isinstance(entry, list) or len(entry) < 3:
                raise ValueError(f"Invalid entry for 'diversity.{loss_name}': expected [compute, weight, path], got {entry!r}.")

            compute, weight, path = entry[0], entry[1], entry[2]
            if isinstance(compute, bool):
                is_enabled = compute
            elif isinstance(compute, int):
                if compute not in {0, 1}:
                    raise ValueError(f"Invalid compute flag for 'diversity.{loss_name}': expected 0/1 or bool, got {compute!r}.")
                is_enabled = bool(compute)
            else:
                raise TypeError(f"Invalid compute flag for 'diversity.{loss_name}': expected 0/1 or bool, got {compute!r}.")

            try:
                weight_float = float(weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid weight for 'diversity.{loss_name}': expected numeric value, got {weight!r}.") from exc

            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"Invalid path for 'diversity.{loss_name}': expected non-empty string, got {path!r}.")

            if is_enabled:
                enabled_entries.append((loss_name, weight_float, path))

        if not enabled_entries:
            return
        if len(enabled_entries) > 1:
            enabled_names = [name for name, _weight, _path in enabled_entries]
            raise ValueError(
                f"Multiple enabled diversity losses: {enabled_names}. Set exactly one entry to compute=1 in losses.diversity."
            )

        self._loss_name, self._weight, path = enabled_entries[0]
        abs_path = Path.cwd() / Path(path)
        family: str = Path(path).stem.title().replace("_", "")
        loss_cls = load_class_from_path(str(abs_path), family)
        self._loss_instance = loss_cls(
            device=str(self._device),
            loss_name=self._loss_name,
            losses_config=self._losses_config,
        )
        self._enabled = True

    def should_compute(self, population_size: int) -> bool:
        return self._enabled and self._loss_instance is not None and self._weight > 0.0 and population_size > 1

    def compute(
        self,
        model: SupportsLatentConcat,
        Z: Tensor,
        num_points: int,
        bbox_size: float,
        norm_order: int,
        neighbor_agg_fn: str,
    ) -> DiversityLossComputeResult:
        if self._loss_instance is None:
            zero = torch.zeros((), device=Z.device, dtype=torch.float32)
            return DiversityLossComputeResult(raw=zero, weighted=zero)

        raw_loss = self._loss_instance.get_loss(
            model,
            Z,
            num_points=num_points,
            bbox_size=bbox_size,
            norm_order=norm_order,
            neighbor_agg_fn=neighbor_agg_fn,
        )
        weighted_loss = self._weight * raw_loss
        return DiversityLossComputeResult(raw=raw_loss, weighted=weighted_loss)

    def build_log_data(self, raw_loss_value: float) -> dict[str, float]:
        weighted = self._weight * raw_loss_value
        return {
            f"raw_losses/{self._loss_name}": raw_loss_value,
            f"weighted_losses/{self._loss_name}": weighted,
            "diversity/loss": raw_loss_value,
            "diversity/weighted_loss": weighted,
            "diversity/weight": self._weight,
        }
