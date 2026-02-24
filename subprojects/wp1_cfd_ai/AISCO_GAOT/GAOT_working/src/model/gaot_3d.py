from typing import Optional, Union

import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from torch import nn
from torch_geometric.data import Batch

from .layers.attn import Transformer
from .layers.magno import MAGNOEncoder


def configloader() -> dict:
    with open("subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT_working/config/config.yaml") as f:
        config = yaml.safe_load(f)
    return config


class GAOT3D(nn.Module):
    """
    Geometry Aware Operator Transformer.

    All configuration is passed as a single ``config`` dict (typically loaded
    from a YAML file).  Expected top-level keys::

        config.magno          – MAGNOEncoder parameters (plain dict)
        config.transformer    – Transformer parameters (plain dict)
        config.latent_tokens  – [D, H, W]
        config.norm_domain    – [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        config.input_size     – int, default 6
        config.output_size    – int, default 1
        config.decoder        – {hidden_features, num_shared_layers, output_heads}
    """

    def __init__(self, config: dict = configloader()):
        super().__init__()
        self.config = config

        # ── top-level scalars ────────────────────────────────────────────
        self.input_size = config.get("input_size", 6)
        self.output_size = config.get("output_size", 1)

        magno_cfg = config["magno"]
        transformer_cfg = config["transformer"]

        self.node_latent_size = magno_cfg["lifting_channels"]
        self.patch_size = transformer_cfg["patch_size"]

        latent_tokens = config.get("latent_tokens", [32, 32, 32])
        self.D = int(latent_tokens[0])
        self.H = int(latent_tokens[1])
        self.W = int(latent_tokens[2])
        self.num_latent_tokens = self.D * self.H * self.W
        self.coord_dim = magno_cfg.get("gno_coord_dim", 3)

        # ── norm domain → latent token grid ──────────────────────────────
        norm_domain = config.get("norm_domain", [[-1, -1, -1], [1, 1, 1]])
        x_min, y_min, z_min = norm_domain[0]
        x_max, y_max, z_max = norm_domain[1]
        meshgrid = torch.meshgrid(
            torch.linspace(x_min, x_max, self.D),
            torch.linspace(y_min, y_max, self.H),
            torch.linspace(z_min, z_max, self.W),
            indexing="ij",
        )
        internal_latent_tokens = torch.stack(meshgrid, dim=-1).reshape(-1, self.coord_dim)
        self.register_buffer("latent_tokens", internal_latent_tokens)

        # ── sub-modules ──────────────────────────────────────────────────
        self.encoder = self._init_encoder(magno_cfg)
        self.processor = self._init_processor(self.config)
        self.attn_pool, self.decoder = self._init_decoder(config.get("decoder", {}))

    # ─── encoder ─────────────────────────────────────────────────────────
    def _init_encoder(self, magno_cfg: dict) -> MAGNOEncoder:
        return MAGNOEncoder(
            in_channels=self.input_size,
            out_channels=self.node_latent_size,
            magno_config=magno_cfg,
        )

    # ─── processor (vision transformer) ─────────────────────────────────
    def _init_processor(self, transformer_cfg: dict) -> Transformer:
        P = self.patch_size
        C = self.node_latent_size
        patch_dim = P * P * P * C

        self.patch_linear = nn.Linear(patch_dim, patch_dim)
        self.positional_embedding_name = transformer_cfg.get("positional_embedding", "rope")
        self.positions = self._get_patch_positions()

        # Inject grid dimensions into attn_config
        attn_cfg = dict(transformer_cfg.get("attn_config", {}))
        attn_cfg["D"] = self.D
        attn_cfg["H"] = self.H
        attn_cfg["W"] = self.W

        proc_cfg = dict(transformer_cfg)
        proc_cfg["attn_config"] = attn_cfg

        return Transformer(
            input_size=patch_dim,
            output_size=patch_dim,
            config=proc_cfg,
        )

    # ─── decoder ─────────────────────────────────────────────────────────
    def _init_decoder(self, decoder_cfg: dict):
        C = self.node_latent_size
        attn_pool = MultiStatPooling(embed_dim=C)

        hidden = decoder_cfg.get("hidden_features", max(C, 256))
        num_shared = decoder_cfg.get("num_shared_layers", 3)
        output_heads = decoder_cfg.get("output_heads", {"force": 1, "cd": 1})

        decoder = MultiHeadDecoder(
            in_features=C,
            hidden_features=hidden,
            num_shared_layers=num_shared,
            output_heads=output_heads,
        )
        return attn_pool, decoder

    # ─── helpers ─────────────────────────────────────────────────────────
    def _get_patch_positions(self) -> torch.Tensor:
        P = self.patch_size
        num_patches_D = self.D // P
        num_patches_H = self.H // P
        num_patches_W = self.W // P
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(num_patches_D, dtype=torch.float32),
                torch.arange(num_patches_H, dtype=torch.float32),
                torch.arange(num_patches_W, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 3)
        return positions

    @staticmethod
    def _compute_absolute_embeddings(positions: torch.Tensor, embed_dim: int) -> torch.Tensor:
        num_tokens, dims = positions.shape
        freq = 1 / 10000 ** (2 * torch.arange(0, embed_dim // 2, dtype=torch.float32, device=positions.device) / embed_dim)
        angles = positions[:, :, None] * freq[None, None, :]
        sum_sin = torch.sum(torch.sin(angles), dim=1)
        sum_cos = torch.sum(torch.cos(angles), dim=1)
        PE = torch.zeros(num_tokens, embed_dim, device=positions.device)
        PE[:, 0::2] = sum_sin
        PE[:, 1::2] = sum_cos
        return PE

    # ─── encode / process / decode ───────────────────────────────────────
    def encode(self, batch: Batch, token: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(batch=batch, latent_tokens=token)

    def process(self, rndata: torch.Tensor, condition: Optional[float] = None) -> torch.Tensor:
        B, N, C = rndata.shape
        D, H, W = self.D, self.H, self.W
        P = self.patch_size
        assert N == D * H * W
        assert D % P == 0 and H % P == 0 and W % P == 0

        nD, nH, nW = D // P, H // P, W // P
        num_patches = nD * nH * nW

        # patchify
        rndata = rndata.view(B, nD, P, nH, P, nW, P, C)
        rndata = rndata.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        rndata = rndata.view(B, num_patches, P * P * P * C)

        rndata = self.patch_linear(rndata)

        pos = self.positions.to(rndata.device)
        relative_positions = None
        if self.positional_embedding_name == "absolute":
            pos_emb = self._compute_absolute_embeddings(pos, P * P * P * self.node_latent_size)
            rndata = rndata + pos_emb
        elif self.positional_embedding_name == "rope":
            relative_positions = pos

        rndata = self.processor(rndata, condition=condition, relative_positions=relative_positions)

        # un-patchify
        rndata = rndata.view(B, nD, nH, nW, P, P, P, C)
        rndata = rndata.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        rndata = rndata.view(B, D * H * W, C)
        return rndata

    def decode(self, rndata: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.decoder(rndata)

    # ─── forward ─────────────────────────────────────────────────────────
    def forward(
        self,
        batch: Batch,
        tokens_pos: Optional[torch.Tensor] = None,
        tokens_batch_idx: Optional[torch.Tensor] = None,
        query_coord_pos: Optional[torch.Tensor] = None,
        query_coord_batch_idx: Optional[torch.Tensor] = None,
        condition: Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        num_graphs = batch.num_graphs
        device = batch.pos.device

        # ── latent tokens ────────────────────────────────────────────────
        if tokens_pos is None:
            latent_tokens_batched = self.latent_tokens.to(device).repeat(num_graphs, 1)
            batch_idx_latent = torch.arange(num_graphs, device=device).repeat_interleave(self.num_latent_tokens)
        elif tokens_batch_idx is None:
            latent_tokens_batched = tokens_pos.to(device).repeat(num_graphs, 1)
            batch_idx_latent = torch.arange(num_graphs, device=device).repeat_interleave(self.num_latent_tokens)
        else:
            latent_tokens_batched = tokens_pos.to(device)
            batch_idx_latent = tokens_batch_idx.to(device)

        # ── encode ───────────────────────────────────────────────────────
        rndata = self.encoder(
            batch=batch,
            latent_tokens_pos=latent_tokens_batched,
            latent_tokens_batch_idx=batch_idx_latent,
        )

        # ── process ──────────────────────────────────────────────────────
        rndata = self.process(rndata=rndata, condition=condition)

        # ── pool → decode ────────────────────────────────────────────────
        pooled = self.attn_pool(rndata)
        output = self.decoder(pooled)
        return output


# ═════════════════════════════════════════════════════════════════════════
#  Pooling & Decoder helpers
# ═════════════════════════════════════════════════════════════════════════


class MultiStatPooling(nn.Module):
    """Concatenates mean, max, std over tokens then projects back."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3 * embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        std_pool = x.std(dim=1).clamp(min=1e-6)
        return self.proj(torch.cat([mean_pool, max_pool, std_pool], dim=-1))


class MultiHeadDecoder(nn.Module):
    """Shared trunk → per-target heads."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 256,
        num_shared_layers: int = 3,
        output_heads: dict[str, int] | None = None,
    ):
        super().__init__()
        if output_heads is None:
            output_heads = {"force": 1, "cd": 1}
        self.head_names = list(output_heads.keys())

        layers: list[nn.Module] = [
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
        ]
        for _ in range(num_shared_layers - 1):
            layers.append(_ResidualBlock(hidden_features))
        self.shared_trunk = nn.Sequential(*layers)

        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_features, hidden_features // 2),
                    nn.GELU(),
                    nn.Linear(hidden_features // 2, out_dim),
                )
                for name, out_dim in output_heads.items()
            }
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared_trunk(x)
        return {name: head(shared) for name, head in self.heads.items()}


class _ResidualBlock(nn.Module):
    """Pre-norm residual: LN → Linear → GELU → Linear → add."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc2(self.act(self.fc1(h)))
        return x + h


# ═════════════════════════════════════════════════════════════════════════
#  Multi-head loss
# ═════════════════════════════════════════════════════════════════════════


def masked_multi_head_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_fn: Union[nn.Module, None] = None,
    weights: Union[dict[str, float], None] = None,
) -> torch.Tensor:
    """Loss over multiple heads, masking NaN targets per sample."""
    if loss_fn is None:
        loss_fn = nn.MSELoss(reduction="none")
    if weights are None:
        weights = {}

    device = next(iter(predictions.values())).device
    total = torch.tensor(0.0, device=device)
    n_active = 0

    for name, pred in predictions.items():
        gt = targets.get(name)
        if gt is None:
            continue
        gt = gt.to(device).view_as(pred)
        valid = ~torch.isnan(gt)
        if not valid.any():
            continue
        w = weights.get(name, 1.0)
        total = total + w * loss_fn(pred[valid], gt[valid]).mean()
        n_active += 1

        return total / max(n_active, 1)