"""Scalar encoder: maps per-sample scalar features (e.g. geometric scaling
factors, bounding-box extents, flow speed) to the same latent dimension used
by the MAGNO encoder, then fuses them with the spatial token features.

Supports two fusion modes (config key ``mode``):
    ``"concat"``  - original additive fusion via concatenation + MLP.
    ``"film"``    - FiLM conditioning: produces per-channel gamma and beta from
                    scalars and applies ``gamma * tokens + beta``.  Optionally
                    position-aware when ``position_aware: true``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch

# Registry of known scalar feature names -> their dimensionality.
# Extend this dict when introducing new scalar features.
SCALAR_FEATURE_DIMS: dict[str, int] = {
    "scale_factors": 3,  # per-axis physical extent [x, y, z]
    "bbox_min": 3,  # per-axis min coord in physical space (metres)
    "bbox_max": 3,  # per-axis max coord in physical space (metres)
    "bbox_center": 3,  # per-axis center coord in physical space (metres)
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
    hidden_dim : int | None
        Hidden width of the encoder MLP.
    mode : str
        ``"concat"`` (default, backward-compatible) or ``"film"``.
    position_aware : bool
        If True **and** mode is ``"film"``, gamma/beta are conditioned on each
        token's grid position so different spatial locations receive different
        modulation.  Requires ``token_positions`` to be passed in ``forward()``.
    num_film_layers : int
        Depth of the FiLM generator MLP (default 2).
    film_init_scale : float
        Kept for config backward-compatibility; no longer used.
        gamma is now bounded to (0, 2) via ``1 + tanh(head(trunk))`` with
        zero-initialised weights, guaranteeing identity at the start of training.
    """

    def __init__(
        self,
        feature_names: list[str],
        latent_dim: int,
        hidden_dim: int | None = None,
        mode: str = "concat",
        position_aware: bool = False,
        num_film_layers: int = 2,
        film_init_scale: float = 0.1,  # noqa: ARG002
    ):
        super().__init__()
        self.feature_names = list(feature_names)
        self.latent_dim = latent_dim
        self.mode = mode
        self.position_aware = position_aware and (mode == "film")

        total_input_dim = sum(SCALAR_FEATURE_DIMS[n] for n in self.feature_names)
        hidden_dim = hidden_dim or latent_dim

        if mode == "concat":
            # ── original concat path (unchanged) ─────────────────────────
            self.encoder = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.fusion = nn.Sequential(
                nn.Linear(2 * latent_dim, latent_dim),
                nn.GELU(),
            )

        elif mode == "film":
            # ── FiLM path ────────────────────────────────────────────────
            coord_dim = 3 if self.position_aware else 0
            film_in = total_input_dim + coord_dim

            # Normalise FiLM inputs so the trunk always sees O(1) activations
            # regardless of the physical scale of scalar features (e.g. metres
            # vs mm) - this prevents the trunk from driving large gamma/beta early in
            # training even with small film_init_scale.
            self.film_input_norm = nn.LayerNorm(film_in)

            layers: list[nn.Module] = []
            in_dim = film_in
            for _ in range(num_film_layers - 1):
                layers += [nn.Linear(in_dim, hidden_dim), nn.GELU()]
                in_dim = hidden_dim
            self.film_trunk = nn.Sequential(*layers)

            # gamma and beta heads - separate so we can control init independently
            self.gamma_head = nn.Linear(in_dim, latent_dim)
            self.beta_head = nn.Linear(in_dim, latent_dim)

            # gamma = 1 + tanh(gamma_head), so zero-init -> exact identity at start.
            # tanh bounds gamma to (0, 2) regardless of training stage, preventing
            # runaway amplification.
            nn.init.zeros_(self.gamma_head.weight)
            nn.init.zeros_(self.gamma_head.bias)
            nn.init.zeros_(self.beta_head.weight)
            nn.init.zeros_(self.beta_head.bias)
        else:
            raise ValueError(f"ScalarEncoder mode must be 'concat' or 'film', got '{mode}'")

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
    def forward(
        self,
        token_features: torch.Tensor,
        batch: Batch,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse scalar information into the latent token grid.

        Parameters
        ----------
        token_features : Tensor [B, M, C]
        batch : Batch
        token_positions : Tensor [M, 3] or None
            Latent-token grid coordinates (only used when position_aware=True).

        Returns
        -------
        Tensor [B, M, C]
        """
        scalars = self._gather_scalars(batch)  # [B, S]
        B, M, _C = token_features.shape

        # ── cache scale_factors stats for the optimizer's W&B logging ────
        if "scale_factors" in self.feature_names:
            sf = getattr(batch, "scale_factors", None)
            if sf is not None:
                sf_flat = sf.detach().float()
                self._last_sf = {
                    "mean": sf_flat.mean().item(),
                    "std": sf_flat.std().item(),
                    "min": sf_flat.min().item(),
                    "max": sf_flat.max().item(),
                }

        if self.mode == "concat":
            encoded = self.encoder(scalars)  # [B, C]
            encoded_broadcast = encoded.unsqueeze(1).expand(-1, M, -1)
            with torch.no_grad():
                self._last_token_norm = token_features.norm(dim=-1).mean().item()
                self._last_scalar_norm = encoded_broadcast.norm(dim=-1).mean().item()
            fused = torch.cat([token_features, encoded_broadcast], dim=-1)
            return self.fusion(fused)

        # ── FiLM path ────────────────────────────────────────────────────
        if self.position_aware:
            assert token_positions is not None, (
                "ScalarEncoder(position_aware=True) requires token_positions to be passed; got None."
            )
            # token_positions: [M, 3] -> broadcast to [B, M, 3]
            tp = token_positions.unsqueeze(0).expand(B, -1, -1)  # [B, M, 3]
            # scalars: [B, S] -> [B, M, S]
            sc = scalars.unsqueeze(1).expand(-1, M, -1)
            film_in = torch.cat([sc, tp], dim=-1)  # [B, M, S+3]
            film_in = self.film_input_norm(film_in)  # normalise to O(1)
            trunk_out = self.film_trunk(film_in)  # [B, M, H]
            gamma = 1.0 + torch.tanh(self.gamma_head(trunk_out))  # [B, M, C] ∈ (0, 2)
            beta = self.beta_head(trunk_out)  # [B, M, C]
        else:
            film_in = self.film_input_norm(scalars)  # normalise to O(1)
            trunk_out = self.film_trunk(film_in)  # [B, H]
            gamma = (1.0 + torch.tanh(self.gamma_head(trunk_out))).unsqueeze(1)  # [B, 1, C] ∈ (0, 2)
            beta = self.beta_head(trunk_out).unsqueeze(1)  # [B, 1, C]

        with torch.no_grad():
            self._last_token_norm = token_features.norm(dim=-1).mean().item()
            self._last_gamma_mean = gamma.mean().item()
            self._last_gamma_std = gamma.std().item()
            self._last_beta_mean = beta.mean().item()
            # Relative perturbation: |(gamma-1)*x + beta| / |x|
            perturb = ((gamma - 1.0) * token_features + beta).norm(dim=-1)
            denom = token_features.norm(dim=-1).clamp(min=1e-8)
            self._last_film_perturbation = (perturb / denom).mean().item()

        return gamma * token_features + beta
