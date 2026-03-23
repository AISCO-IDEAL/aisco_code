import os

import torch
import yaml
from torch import nn
from torch_geometric.data import Batch

from .layers.attn import Transformer
from .layers.magno import MAGNODecoder, MAGNOEncoder



class GAOT3D(nn.Module):
    """
    Geometry Aware Operator Transformer:
    Multiscale Attentional Graph Neural Operator + U Vision Transformer + Multiscale Attentional Graph Neural Operator
    """

    def __init__(
        self,
        config: dict,
        field_output_size: int = 0,
    ):
        nn.Module.__init__(self)
        self.config = config

        # ── unpack sub-configs ───────────────────────────────────────────
        magno_config = config["magno"]
        attn_config = config["transformer"]
        latent_tokens = config["latent_tokens"]  # e.g. [D, H, W]

        input_size = config["input_size"]
        output_size = config.get("output_size", field_output_size)
        self.field_output_size = config.get("field_output_size", field_output_size)

        self.input_size = input_size
        self.output_size = output_size
        self.node_latent_size = magno_config["lifting_channels"]
        self.patch_size = attn_config["patch_size"]
        self.D, self.H, self.W = latent_tokens
        self.depth, self.height, self.width = self.D, self.H, self.W
        self.num_latent_tokens = self.D * self.H * self.W
        self.coord_dim = magno_config["gno_coord_dim"]

        # ── norm domain → latent token grid ──────────────────────────────
        norm_domain = config.get("norm_domain", [[-1, -1, -1], [1, 1, 1]])
        x_min, y_min, z_min = norm_domain[0]
        x_max, y_max, z_max = norm_domain[1]
        meshgrid = torch.meshgrid(
            torch.linspace(x_min, x_max, self.depth),
            torch.linspace(y_min, y_max, self.height),
            torch.linspace(z_min, z_max, self.width),
            indexing="ij",
        )
        internal_latent_tokens = torch.stack(meshgrid, dim=-1).reshape(-1, self.coord_dim)
        self.latent_tokens: torch.Tensor
        self.register_buffer("latent_tokens", internal_latent_tokens)

        # ── sub-modules ──────────────────────────────────────────────────
        self.encoder = self._init_encoder(magno_config)
        self.processor = self._init_processor(self.config)
        self.attn_pool, self.decoder = self._init_decoder(config.get("decoder", {}))

        # ── optional flowfield decoder (MAGNODecoder) ────────────────────
        if self.field_output_size > 0:
            self.field_decoder = MAGNODecoder(
                in_channels=self.node_latent_size,
                out_channels=self.field_output_size,
                magno_config=magno_config,
            )

    # ─── encoder ─────────────────────────────────────────────────────────
    def _init_encoder(self, magno_cfg: dict) -> MAGNOEncoder:
        return MAGNOEncoder(
            in_channels=self.input_size,
            # out channel deactivated -> out_channels=self.node_latent_size,
            magno_config=magno_cfg,
        )

    # ─── processor (vision transformer) ─────────────────────────────────
    def _init_processor(self, config: dict) -> Transformer:
        transformer_cfg = config["transformer"]
        P = self.patch_size
        C = self.node_latent_size
        patch_dim = P * P * P * C

        self.patch_linear = nn.Linear(patch_dim, patch_dim)
        self.positional_embedding_name = transformer_cfg.get("positional_embedding", "rope")
        self.positions = self._get_patch_positions()

        # Inject grid dimensions into attn_config
        attn_cfg = dict(transformer_cfg.get("attn_config", {}))
        attn_cfg["D"] = self.depth
        attn_cfg["H"] = self.height
        attn_cfg["W"] = self.width

        # Build full transformer_config with runtime-computed sizes
        proc_cfg = dict(transformer_cfg)
        proc_cfg["input_size"] = patch_dim
        proc_cfg["output_size"] = patch_dim
        proc_cfg["attn_config"] = attn_cfg

        return Transformer(transformer_config=proc_cfg)

    # ─── decoder ─────────────────────────────────────────────────────────
    def _init_decoder(self, decoder_cfg: dict):
        attn_pool = MultiStatPooling(embed_dim=self.node_latent_size)

        hidden = decoder_cfg.get("hidden_features", max(self.node_latent_size, 256))
        num_shared = decoder_cfg.get("num_shared_layers", 3)
        output_heads = decoder_cfg.get("output_heads", {"force": 1, "cd": 1})

        decoder = MultiHeadDecoder(
            in_features=self.node_latent_size,
            hidden_features=hidden,
            num_shared_layers=num_shared,
            output_heads=output_heads,
        )
        return attn_pool, decoder

    # ─── helpers ─────────────────────────────────────────────────────────
    def _get_patch_positions(self) -> torch.Tensor:
        patch_edge = self.patch_size
        num_patches_depth = self.depth // patch_edge
        num_patches_height = self.height // patch_edge
        num_patches_width = self.width // patch_edge
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(num_patches_depth, dtype=torch.float32),
                torch.arange(num_patches_height, dtype=torch.float32),
                torch.arange(num_patches_width, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 3)
        return positions

    @staticmethod
    def _compute_absolute_embeddings(positions: torch.Tensor, embed_dim: int) -> torch.Tensor:
        num_tokens, _dims = positions.shape
        freq = 1 / 10000 ** (2 * torch.arange(0, embed_dim // 2, dtype=torch.float32, device=positions.device) / embed_dim)
        angles = positions[:, :, None] * freq[None, None, :]
        sum_sin = torch.sum(torch.sin(angles), dim=1)
        sum_cos = torch.sum(torch.cos(angles), dim=1)
        pos_encoding = torch.zeros(num_tokens, embed_dim, device=positions.device)
        pos_encoding[:, 0::2] = sum_sin
        pos_encoding[:, 1::2] = sum_cos
        return pos_encoding

    # ─── encode / process / decode ───────────────────────────────────────
    def encode(self, batch: Batch, token: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(batch=batch, latent_tokens=token)

    def process(self, rndata: torch.Tensor, condition: float | None = None) -> torch.Tensor:
        batch_size, num_tokens, channels = rndata.shape
        grid_depth, grid_height, grid_width = self.depth, self.height, self.width
        patch_edge = self.patch_size
        assert num_tokens == grid_depth * grid_height * grid_width
        assert grid_depth % patch_edge == 0 and grid_height % patch_edge == 0 and grid_width % patch_edge == 0

        patches_d, patches_h, patches_w = grid_depth // patch_edge, grid_height // patch_edge, grid_width // patch_edge
        num_patches = patches_d * patches_h * patches_w

        # patchify
        rndata = rndata.view(batch_size, patches_d, patch_edge, patches_h, patch_edge, patches_w, patch_edge, channels)
        rndata = rndata.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        rndata = rndata.view(batch_size, num_patches, patch_edge * patch_edge * patch_edge * channels)

        rndata = self.patch_linear(rndata)

        pos = self.positions.to(rndata.device)
        relative_positions = None
        if self.positional_embedding_name == "absolute":
            pos_emb = self._compute_absolute_embeddings(pos, patch_edge * patch_edge * patch_edge * self.node_latent_size)
            rndata = rndata + pos_emb
        elif self.positional_embedding_name == "rope":
            relative_positions = pos

        rndata = self.processor(rndata, condition=condition, relative_positions=relative_positions)

        # un-patchify
        rndata = rndata.view(batch_size, patches_d, patches_h, patches_w, patch_edge, patch_edge, patch_edge, channels)
        rndata = rndata.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        rndata = rndata.view(batch_size, grid_depth * grid_height * grid_width, channels)
        return rndata

    def decode(self, rndata: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.decoder(rndata)

    # ─── forward ─────────────────────────────────────────────────────────
    def forward(
        self,
        batch: Batch,
        tokens_pos: torch.Tensor | None = None,
        tokens_batch_idx: torch.Tensor | None = None,
        query_coord_pos: torch.Tensor | None = None,
        query_coord_batch_idx: torch.Tensor | None = None,
        condition: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for GAOT3D model using PyG Batch.

        Args:
            batch (Batch): Input PyG Batch (contains batch.pos, batch.x, batch.batch).
            tokens_pos (Tensor, optional): External latent token coordinates [TotalLatentNodes, D].
                                           If None, uses internal self.latent_tokens.
            tokens_batch_idx (Tensor, optional): Batch index for external tokens [TotalLatentNodes].
                                                 Required if tokens_pos is provided.
            query_coord_pos (Tensor, optional): External query coordinates for field decoder [TotalQueryNodes, D].
                                                 If None, uses batch.pos from input batch.
            query_coord_batch_idx (Tensor, optional): Batch index for external query coords [TotalQueryNodes].
                                                       Required if query_coord_pos is provided.
            condition (float, optional): Condition parameter for processor.

        Returns:
            dict: Always contains scalar heads {"force": [B,1], "cd": [B,1]}.
                  When field_output_size > 0, also contains
                  {"field": [TotalQueryNodes, field_output_size]}.
        """
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

        # ── determine decoder query coordinates ──────────────────────────
        if query_coord_pos is None:
            phys_pos_query_eff = batch.pos  # [TotalPhysicalNodes, D]
            batch_idx_phys_query_eff = batch.batch  # [TotalPhysicalNodes]
        else:
            assert query_coord_batch_idx is not None, "query_coord_batch_idx is required if query_coord_pos is provided"
            phys_pos_query_eff = query_coord_pos.to(device)
            batch_idx_phys_query_eff = query_coord_batch_idx.to(device)

        # ── encode ───────────────────────────────────────────────────────
        rndata = self.encoder(
            batch=batch,
            latent_tokens_pos=latent_tokens_batched,
            latent_tokens_batch_idx=batch_idx_latent,
        )

        # ── process ──────────────────────────────────────────────────────
        rndata_proc = self.process(rndata=rndata, condition=condition)

        # ── scalar path: pool → multi-head decoder ───────────────────────
        # Shape: [B, M, C_lifted] -> [B, C_lifted]
        pooled_features = self.attn_pool(rndata_proc)

        # Multi-head decode: returns {"force": [B,1], "cd": [B,1]}
        output = self.decoder(pooled_features)

        # ── field path (optional): MAGNODecoder ──────────────────────────
        if self.field_output_size > 0:
            # Flatten processor output: [B, M, C] -> [TotalLatentNodes, C]
            rndata_proc_flat = rndata_proc.reshape(-1, self.node_latent_size)

            # Decode to per-vertex field predictions
            field_pred = self.field_decoder(
                rndata_flat=rndata_proc_flat,
                phys_pos_query=phys_pos_query_eff,
                batch_idx_phys_query=batch_idx_phys_query_eff,
                latent_tokens_pos=latent_tokens_batched,
                latent_tokens_batch_idx=batch_idx_latent,
                batch=batch,
            )  # [TotalQueryNodes, field_output_size]
            output["field"] = field_pred

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
            layers.append(_ResidualBlock(hidden_features))  # noqa: PERF401
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
    loss_fn: nn.Module | None = None,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Loss over multiple heads, masking NaN targets per sample."""
    if loss_fn is None:
        loss_fn = nn.MSELoss(reduction="none")
    if weights is None:
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
