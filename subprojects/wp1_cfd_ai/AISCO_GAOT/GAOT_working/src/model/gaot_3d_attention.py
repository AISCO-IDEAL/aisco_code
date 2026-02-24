from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch

from .layers.attn import Transformer, TransformerConfig
from .layers.magno import MAGNOConfig, MAGNODecoder, MAGNOEncoder


class GAOT3D(nn.Module):
    """
    Geometry Aware Operator Transformer:
    Multiscale Attentional Graph Neural Operator + U Vision Transformer + Multiscale Attentional Graph Neural Operator
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        magno_config: MAGNOConfig = MAGNOConfig(),
        attn_config: TransformerConfig = TransformerConfig(),
        latent_tokens: tuple = (32, 32, 32),
        norm_domin: list = [(-1, -1, -1), (1, 1, 1)],
    ):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.node_latent_size = magno_config.lifting_channels
        self.patch_size = attn_config.patch_size
        self.D, self.H, self.W = latent_tokens
        self.num_latent_tokens = self.D * self.H * self.W
        self.coord_dim = magno_config.gno_coord_dim

        # --- Create and register latent_tokens ---
        x_min, y_min, z_min = norm_domin[0]
        x_max, y_max, z_max = norm_domin[1]
        meshgrid = torch.meshgrid(
            torch.linspace(x_min, x_max, self.D),
            torch.linspace(y_min, y_max, self.H),
            torch.linspace(z_min, z_max, self.W),
            indexing="ij",
        )

        internal_latent_tokens = torch.stack(meshgrid, dim=-1).reshape(-1, self.coord_dim)
        self.register_buffer("latent_tokens", internal_latent_tokens)
        # --- End latent_tokens creation ---
        # Initialize encoder, processor, and decoder
        self.encoder = self.init_encoder(input_size, self.node_latent_size, magno_config)
        self.processor = self.init_processor(self.node_latent_size, attn_config)
        self.decoder = self.init_decoder(self.node_latent_size, output_size)

    def init_encoder(self, input_size, node_latent_size, magno_config):
        return MAGNOEncoder(in_channels=input_size, out_channels=node_latent_size, gno_config=magno_config)

    def init_processor(self, node_latent_size, config):
        # Initialize the Vision Transformer processor
        self.patch_linear = nn.Linear(
            self.patch_size * self.patch_size * self.patch_size * self.node_latent_size,
            self.patch_size * self.patch_size * self.patch_size * self.node_latent_size,
        )

        self.positional_embedding_name = config.positional_embedding
        self.positions = self._get_patch_positions()

        config.attn_config.D = self.D
        config.attn_config.H = self.H
        config.attn_config.W = self.W

        return Transformer(
            input_size=self.node_latent_size * self.patch_size * self.patch_size * self.patch_size,
            output_size=self.node_latent_size * self.patch_size * self.patch_size * self.patch_size,
            config=config,
        )

    def init_decoder(self, node_latent_size: int, output_size: int) -> "MultiHeadDecoder":
        self.attn_pool = MultiStatPooling(embed_dim=node_latent_size)
        return MultiHeadDecoder(
            in_features=node_latent_size,
            hidden_features=max(node_latent_size, 256),
            num_shared_layers=3,
            output_heads={"force": 1, "cd": 1},
        )

    def _get_patch_positions(self):
        """
        Generate positional embeddings for the patches.
        """
        num_patches_D = self.D // self.patch_size
        num_patches_H = self.H // self.patch_size
        num_patches_W = self.W // self.patch_size
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

    def _compute_absolute_embeddings(self, positions, embed_dim):
        """
        Compute absolute positional embeddings based on geometric coordinates.

        Args:
            positions (torch.Tensor): Tensor of shape [num_tokens, dims] representing the coordinates of each token.
            embed_dim (int): The desired embedding dimension. Assumed to be even for simplicity.

        Returns:
            torch.Tensor: Positional embedding tensor of shape [num_tokens, embed_dim].
        """
        # Extract the number of tokens and dimensions from the input tensor
        num_tokens, dims = positions.shape

        # Compute the frequencies, one for each pair of embedding dimensions
        # omega_k = 1 / 10000^(2k / embed_dim) for k = 0, 1, ..., embed_dim//2 - 1
        freq = 1 / 10000 ** (2 * torch.arange(0, embed_dim // 2, dtype=torch.float32, device=positions.device) / embed_dim)

        # Expand dimensions for broadcasting: positions becomes [num_tokens, dims, 1]
        pos_expanded = positions[:, :, None]
        # freq becomes [1, 1, embed_dim//2]
        freq_expanded = freq[None, None, :]

        # Compute angles for all tokens, dimensions, and frequencies
        # angles[i, d, k] = positions[i, d] * freq[k]
        angles = pos_expanded * freq_expanded  # Shape: [num_tokens, dims, embed_dim//2]

        # Compute sine and cosine values for all angles
        sin_values = torch.sin(angles)  # Shape: [num_tokens, dims, embed_dim//2]
        cos_values = torch.cos(angles)  # Shape: [num_tokens, dims, embed_dim//2]

        # Sum the sine and cosine values across all dimensions
        sum_sin = torch.sum(sin_values, dim=1)  # Shape: [num_tokens, embed_dim//2]
        sum_cos = torch.sum(cos_values, dim=1)  # Shape: [num_tokens, embed_dim//2]

        # Initialize the positional embedding tensor
        PE = torch.zeros(num_tokens, embed_dim, device=positions.device)

        # Assign summed sine values to even indices and cosine values to odd indices
        PE[:, 0::2] = sum_sin  # Even positions
        PE[:, 1::2] = sum_cos  # Odd positions

        return PE

    def encode(self, batch: Batch, token: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        batch: Batch
            The input batch containing the data
        token: Optional[torch.Tensor]
            ND Tensor of shape [batch_size, n_token_nodes, n_dim]
        Returns
        -------
        torch.Tensor
            The regional node data of shape [..., n_regional_nodes, node_latent_size]
        """
        # Apply GNO encoder
        encoded = self.encoder(batch=batch, latent_tokens=token)
        return encoded

    def process(self, rndata: Optional[torch.Tensor] = None, condition: Optional[float] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        graph:Graph
            regional to regional graph, a homogeneous graph
        rndata:Optional[torch.Tensor]
            ND Tensor of shape [..., n_regional_nodes, node_latent_size]
        condition:Optional[float]
            The condition of the model

        Returns
        -------
        torch.Tensor
            The regional node data of shape [..., n_regional_nodes, node_latent_size]
        """
        batch_size = rndata.shape[0]
        n_regional_nodes = rndata.shape[1]
        C = rndata.shape[2]
        D, H, W = self.D, self.H, self.W
        assert n_regional_nodes == D * H * W, f"n_regional_nodes ({n_regional_nodes}) is not equal to H ({H}) * W ({W})"

        P = self.patch_size
        assert D % P == 0 and H % P == 0 and W % P == 0, "Dimensions must be divisible by patch size"
        num_patches_D = D // P
        num_patches_H = H // P
        num_patches_W = W // P
        num_patches = num_patches_D * num_patches_H * num_patches_W
        # Reshape to patches
        rndata = rndata.view(batch_size, D, H, W, C)
        rndata = rndata.view(batch_size, num_patches_D, P, num_patches_H, P, num_patches_W, P, C)
        rndata = rndata.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()  # [batch, nD, nH, nW, P, P, P, C]
        rndata = rndata.view(batch_size, num_patches, P * P * P * C)

        # Apply Vision Transformer
        rndata = self.patch_linear(rndata)
        pos = self.positions.to(rndata.device)  # shape [num_patches, 3]
        if self.positional_embedding_name == "absolute":
            pos_emb = self._compute_absolute_embeddings(pos, P * P * P * self.node_latent_size)
            rndata = rndata + pos_emb
            relative_positions = None

        elif self.positional_embedding_name == "rope":
            relative_positions = pos

        rndata = self.processor(rndata, condition=condition, relative_positions=relative_positions)

        # Reshape back to the original shape
        rndata = rndata.view(batch_size, num_patches_D, num_patches_H, num_patches_W, P, P, P, C)
        rndata = rndata.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        rndata = rndata.view(batch_size, D * H * W, C)

        return rndata

    def decode(self, rndata: torch.Tensor | None = None) -> torch.Tensor:
        return self.decoder(rndata_flat=rndata)

    def forward(
        self,
        batch: Batch,
        tokens_pos: Optional[torch.Tensor] = None,
        tokens_batch_idx: Optional[torch.Tensor] = None,
        query_coord_pos: Optional[torch.Tensor] = None,
        query_coord_batch_idx: Optional[torch.Tensor] = None,
        condition: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Forward pass for GAOT3D model using PyG Batch.

        Args:
            batch (Batch): Input PyG Batch (contains batch.pos, batch.x, batch.batch).
            tokens_pos (Tensor, optional): External latent token coordinates [TotalLatentNodes, D].
                                           If None, uses internal self.latent_tokens.
            tokens_batch_idx (Tensor, optional): Batch index for external tokens [TotalLatentNodes].
                                                 Required if tokens_pos is provided.
            query_coord_pos (Tensor, optional): External query coordinates for decoder [TotalQueryNodes, D].
                                                 If None, uses batch.pos from input batch.
            query_coord_batch_idx (Tensor, optional): Batch index for external query coords [TotalQueryNodes].
                                                       Required if query_coord_pos is provided.
            condition (float, optional): Condition parameter for processor.

        Returns:
            Tensor: Output tensor on physical/query nodes [TotalQueryNodes, OutputChannels].
        """
        num_graphs = batch.num_graphs
        device = batch.pos.device

        # --- Determine Latent Tokens ---
        if tokens_pos is None:
            assert tokens_batch_idx is None, "tokens_batch_idx should be None if tokens_pos is None"
            latent_tokens_batched = self.latent_tokens.to(device).repeat(num_graphs, 1)
            batch_idx_latent = torch.arange(num_graphs, device=device).repeat_interleave(self.num_latent_tokens)
        else:
            if tokens_batch_idx is None:
                latent_tokens_batched = tokens_pos.to(device).repeat(num_graphs, 1)
                batch_idx_latent = torch.arange(num_graphs, device=device).repeat_interleave(self.num_latent_tokens)
            else:
                assert tokens_pos.shape[0] == tokens_batch_idx.shape[0], "tokens_pos and tokens_batch_idx must have same length"
                assert tokens_batch_idx.max() == num_graphs - 1, "tokens_batch_idx does not match batch size"
                latent_tokens_batched = tokens_pos.to(device)
                batch_idx_latent = tokens_batch_idx.to(device)

        # -- End Determine Latent Tokens ---

        # --- Determine Decoder Query Coordinates ---
        if query_coord_pos is None:
            phys_pos_query_eff = batch.pos  # [TotalPhysicalNodes, D]
            batch_idx_phys_query_eff = batch.batch  # [TotalPhysicalNodes]
        else:
            assert query_coord_batch_idx is not None, "query_coord_batch_idx is required if query_coord_pos is provided"
            assert query_coord_pos.shape[0] == query_coord_batch_idx.shape[0], (
                "query_coord_pos and query_coord_batch_idx must have same length"
            )
            assert query_coord_batch_idx.max() == num_graphs - 1, "query_coord_batch_idx does not match batch size"
            phys_pos_query_eff = query_coord_pos.to(device)
            batch_idx_phys_query_eff = query_coord_batch_idx.to(device)

        # Encode: Map physical nodes to regional nodes using MAGNO Encoder
        rndata = self.encoder(
            batch=batch,  # Contains phys_pos, phys_feat, batch_idx_phys
            latent_tokens_pos=latent_tokens_batched,
            latent_tokens_batch_idx=batch_idx_latent,
        )  # Output shape: [B, M, C_lifted]

        # Process: Apply Vision Transformer on the regional nodes
        # Input shape: [B, M, C_lifted]
        rndata_proc = self.process(rndata=rndata, condition=condition)  # Output shape: [B, M, C_lifted]

        # Attention-weighted pooling over latent tokens.
        # Shape: [B, M, C_lifted] -> [B, C_lifted]
        pooled_features = self.attn_pool(rndata_proc)

        # Multi-head decode: returns {"force": [B,1], "cd": [B,1]}
        output = self.decoder(pooled_features)

        return output


class MultiStatPooling(nn.Module):
    """Multi-statistic pooling: concatenates mean, max, and std over the
    token dimension, then projects back down.

    This is far more expressive than single-query attention pooling because
    it captures first-order (mean), extreme (max), and spread (std)
    information simultaneously, giving the decoder a richer summary of
    the latent field.

    Output dimension: ``embed_dim`` (projected from 3 * embed_dim).
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3 * embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C]  (batch, tokens, channels)
        Returns:
            [B, C]  pooled representation
        """
        mean_pool = x.mean(dim=1)                    # [B, C]
        max_pool = x.max(dim=1).values               # [B, C]
        std_pool = x.std(dim=1).clamp(min=1e-6)      # [B, C]
        multi = torch.cat([mean_pool, max_pool, std_pool], dim=-1)  # [B, 3C]
        return self.proj(multi)  # [B, C]


class AttentionPooling(nn.Module):
    """Learned attention-weighted pooling over token dimension.

    Instead of plain mean pooling, this learns a query vector that attends
    over all tokens to produce a weighted sum, so the model can focus on
    the most aerodynamically relevant regions (e.g. separation zones).

    .. deprecated::
        Use :class:`MultiStatPooling` instead for better gradient flow.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.scale = embed_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C]  (batch, tokens, channels)
        Returns:
            [B, C]  pooled representation
        """
        # query: [1,1,C] broadcast to [B,1,C], keys: [B,N,C]
        attn_logits = (self.query * self.scale) @ x.transpose(-2, -1)  # [B, 1, N]
        attn_weights = F.softmax(attn_logits, dim=-1)
        pooled = (attn_weights @ x).squeeze(1)  # [B, C]
        return pooled


class MultiHeadDecoder(nn.Module):
    """Multi-head decoder with shared backbone and per-target heads.

    Supports partial labels: the model always predicts all outputs, but
    the training loss masks out heads whose ground truth is unavailable.

    Architecture:
        pooled features → shared MLP (with residuals + LayerNorm)
                        → head_force  → force prediction
                        → head_cd     → drag coefficient prediction
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 256,
        num_shared_layers: int = 3,
        output_heads: dict[str, int] | None = None,
    ):
        """
        Args:
            in_features: dimension of the pooled feature vector (C_lifted).
            hidden_features: width of hidden layers in the shared trunk.
            num_shared_layers: number of layers in the shared trunk.
            output_heads: mapping from head name to output dimension,
                          e.g. {"force": 1, "cd": 1} or {"force": 3, "cd": 1}.
                          Defaults to {"force": 1, "cd": 1}.
        """
        super().__init__()
        if output_heads is None:
            output_heads = {"force": 1, "cd": 1}
        self.head_names = list(output_heads.keys())

        # --- Shared Trunk (residual MLP) ---
        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.LayerNorm(hidden_features))
        layers.append(nn.GELU())
        for _ in range(num_shared_layers - 1):
            layers.append(_ResidualBlock(hidden_features))
        self.shared_trunk = nn.Sequential(*layers)

        # --- Per-target heads ---
        self.heads = nn.ModuleDict()
        for name, out_dim in output_heads.items():
            self.heads[name] = nn.Sequential(
                nn.Linear(hidden_features, hidden_features // 2),
                nn.GELU(),
                nn.Linear(hidden_features // 2, out_dim),
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C] pooled feature vector.
        Returns:
            dict mapping head name → prediction tensor [B, out_dim].
        """
        shared = self.shared_trunk(x)
        return {name: head(shared) for name, head in self.heads.items()}


class _ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → Linear → GELU → Linear → add."""

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


def masked_multi_head_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_fn: nn.Module = None,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute loss over multiple heads, masking out NaN targets per sample.

    In a mixed batch (e.g. Euler + DrivAer), every sample has both
    ``force`` and ``cd`` attributes, but unavailable labels are stored
    as NaN.  This function skips NaN entries so only valid labels
    contribute to the gradient.

    Args:
        predictions: dict from head name → predicted tensor [B, D].
        targets: dict from head name → target tensor [B, D].
                 Samples with NaN are skipped for that head.
        loss_fn: element-wise loss (default: MSELoss with reduction='none').
        weights: optional per-head scalar weights.

    Returns:
        Scalar loss (mean of active losses). Returns 0 if nothing active.

    Example usage in training loop::

        preds = model(batch)   # {"force": [B,1], "cd": [B,1]}
        targets = {
            "force": batch.force,   # has NaN where unavailable
            "cd":    batch.cd,      # has NaN where unavailable
        }
        loss = masked_multi_head_loss(preds, targets)
        loss.backward()
    """
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
        valid_mask = ~torch.isnan(gt)
        if not valid_mask.any():
            continue
        per_elem_loss = loss_fn(pred[valid_mask], gt[valid_mask])
        w = weights.get(name, 1.0)
        total = total + w * per_elem_loss.mean()
        n_active += 1

    return total / max(n_active, 1)
