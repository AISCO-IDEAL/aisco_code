"""Scalar encoder: maps per-sample scalar features (e.g. geometric scaling
factors, flow speed) to the same latent dimension used by the MAGNO encoder,
then fuses them with the spatial token features via concatenation + MLP.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch

# Registry of known scalar feature names → their dimensionality.
# Extend this dict when introducing new scalar features.
SCALAR_FEATURE_DIMS: dict[str, int] = {
    "scale_factors": 3,  # per-axis physical extent [x, y, z]
    "v_inf": 1,  # freestream velocity (scalar)
}


class ScalarEncoder(nn.Module):
    """Encode per-sample scalars and broadcast onto the latent token grid.

    Parameters
    ----------
    feature_names : list[str]
        Ordered list of ``data.<name>`` attributes to concatenate.
    latent_dim : int
        Output dimension (must match ``lifting_channels`` of the MAGNO encoder).
    hidden_dim : int
        Hidden width of the encoder MLP.
    """

    def __init__(self, feature_names: list[str], latent_dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.feature_names = list(feature_names)
        self.latent_dim = latent_dim

        total_input_dim = sum(SCALAR_FEATURE_DIMS[n] for n in self.feature_names)
        hidden_dim = hidden_dim or latent_dim

        # Encoder: scalars → latent_dim   (2-layer MLP, GELU)
        self.encoder = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Fusion: concat(token_features, scalar_features) → latent_dim  (1-layer, GELU)
        self.fusion = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.GELU(),
        )

    # ──────────────────────────────────────────────────────────────────
    def _gather_scalars(self, batch: Batch) -> torch.Tensor:
        """Collect requested scalar features from the batch into one vector per graph.

        Returns shape ``[B, total_input_dim]``.
        """
        parts: list[torch.Tensor] = []
        for name in self.feature_names:
            feat = getattr(batch, name, None)
            if feat is None:
                raise AttributeError(f"ScalarEncoder requires '{name}' on each sample but it was not found in the batch.")
            # Ensure shape is [B, D] (some attrs might be [B] or [B,1]).
            if feat.ndim == 1:
                feat = feat.unsqueeze(-1)
            parts.append(feat)
        return torch.cat(parts, dim=-1)  # [B, total_input_dim]

    # ──────────────────────────────────────────────────────────────────
    def forward(self, token_features: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Fuse scalar information into the latent token grid.

        Parameters
        ----------
        token_features : Tensor
            Shape ``[B, M, C]`` — output of the MAGNO encoder.
        batch : Batch
            PyG Batch carrying the scalar attributes listed in
            ``self.feature_names``.

        Returns
        -------
        Tensor
            Shape ``[B, M, C]`` — fused token features.
        """
        scalars = self._gather_scalars(batch)  # [B, total_input_dim]
        encoded = self.encoder(scalars)  # [B, latent_dim]

        # Broadcast [B, latent_dim] → [B, M, latent_dim]
        num_tokens = token_features.shape[1]
        encoded_broadcast = encoded.unsqueeze(1).expand(-1, num_tokens, -1)

        # Cache norms for W&B diagnostics (detached, no graph)
        with torch.no_grad():
            self._last_token_norm = token_features.norm(dim=-1).mean().item()
            self._last_scalar_norm = encoded_broadcast.norm(dim=-1).mean().item()

        # Concat + fusion MLP
        fused = torch.cat([token_features, encoded_broadcast], dim=-1)  # [B, M, 2*latent_dim]
        return self.fusion(fused)  # [B, M, latent_dim]
