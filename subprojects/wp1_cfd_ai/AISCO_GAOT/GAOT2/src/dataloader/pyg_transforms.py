# src/trainer/utils/pyg_transforms.py

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.src.utils.scale import rescale, rescale_new

EPSILON = 1e-10


class RescalePosition(BaseTransform):
    """Rescales node positions 'pos' to a specified range (default: [-1, 1])."""

    def __init__(self, lims=(-1.0, 1.0)):
        self.lims = lims

    def forward(self, data: Data) -> Data:
        if hasattr(data, "pos") and data.pos is not None:
            data.pos = rescale(data.pos, lims=self.lims)
        else:
            print("Warning: RescalePosition transform called but data has no 'pos' attribute.")
        return data

    def __call__(self, data: Data) -> Data:
        if hasattr(data, "pos") and data.pos is not None:
            data.pos = rescale(data.pos, lims=self.lims)
        else:
            print("Warning: RescalePosition transform called but data has no 'pos' attribute.")
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lims={self.lims})"


class RescalePositionNew(BaseTransform):
    """Rescales node positions 'pos' to a specified range (default: [-1, 1])."""

    def __init__(self, lims=(-1.0, 1.0), phy_domain=([-3e3, -3e3, -3e3], [3e3, 3e3, 3e3])):
        self.lims = lims
        self.phy_domain = phy_domain

    def forward(self, data: Data) -> Data:
        if hasattr(data, "pos") and data.pos is not None:
            data.pos = rescale_new(data.pos, lims=self.lims, phys_domain=self.phy_domain)
        else:
            print("Warning: RescalePosition transform called but data has no 'pos' attribute.")
        return data

    def __call__(self, data: Data) -> Data:
        if hasattr(data, "pos") and data.pos is not None:
            data.pos = rescale_new(data.pos, lims=self.lims, phys_domain=self.phy_domain)
        else:
            print("Warning: RescalePosition transform called but data has no 'pos' attribute.")
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lims={self.lims})"


class NormalizeFeatures(BaseTransform):
    """Normalizes node features 'y' (or per-head 'force'/'cd') and optionally 'c' using pre-computed mean and std."""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        c_mean: torch.Tensor | None = None,
        c_std: torch.Tensor | None = None,
        per_head_stats: dict | None = None,
    ):
        self.mean = mean.detach()
        self.std = std.detach()

        self.c_mean = c_mean.detach() if c_mean is not None else None
        self.c_std = c_std.detach() if c_std is not None else None

        # Per-head stats for multi-head format (keys: "force", "cd")
        self.per_head_stats = per_head_stats or {}

    def _normalize_targets(self, data: Data) -> Data:
        """Normalize target values: either data.y (old format) or data.force/data.cd (multi-head)."""
        if hasattr(data, "y") and data.y is not None:
            mean_dev = self.mean.to(data.y.device)
            std_dev = self.std.to(data.y.device)
            data.y = (data.y - mean_dev) / (std_dev + EPSILON)
        else:
            # Multi-head format: normalize force and cd separately using per-head stats
            if hasattr(data, "force") and data.force is not None and "force" in self.per_head_stats:
                f_mean = self.per_head_stats["force"]["mean"].to(data.force.device)
                f_std = self.per_head_stats["force"]["std"].to(data.force.device)
                # Only normalize non-NaN values
                valid = ~torch.isnan(data.force)
                if valid.any():
                    data.force = data.force.clone()
                    data.force[valid] = (data.force[valid] - f_mean) / (f_std + EPSILON)

            if hasattr(data, "cd") and data.cd is not None and "cd" in self.per_head_stats:
                cd_mean = self.per_head_stats["cd"]["mean"].to(data.cd.device)
                cd_std = self.per_head_stats["cd"]["std"].to(data.cd.device)
                valid = ~torch.isnan(data.cd)
                if valid.any():
                    data.cd = data.cd.clone()
                    data.cd[valid] = (data.cd[valid] - cd_mean) / (cd_std + EPSILON)
        return data

    def _normalize_c(self, data: Data) -> Data:
        """Normalize 'c' features if stats are available."""
        if hasattr(data, "c") and data.c is not None and self.c_mean is not None and self.c_std is not None:
            c_mean_dev = self.c_mean.to(data.c.device)
            c_std_dev = self.c_std.to(data.c.device)
            data.c = (data.c - c_mean_dev) / (c_std_dev + EPSILON)
        return data

    def forward(self, data: Data) -> Data:
        data = self._normalize_targets(data)
        data = self._normalize_c(data)
        return data

    def __call__(self, data: Data) -> Data:
        data = self._normalize_targets(data)
        data = self._normalize_c(data)
        return data

    def __repr__(self) -> str:
        has_c = self.c_mean is not None and self.c_std is not None
        heads = list(self.per_head_stats.keys()) if self.per_head_stats else []
        return f"{self.__class__.__name__}(mean=..., std=..., has_c_norm={has_c}, heads={heads})"
