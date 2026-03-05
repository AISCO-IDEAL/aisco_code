import torch
import torch.nn as nn  # noqa: PLR0402
import torch_geometric as pyg
from torch_geometric.nn import knn as pyg_knn
from torch_geometric.nn import radius as pyg_radius
from torch_geometric.utils import coalesce, dropout_edge

from .geoembed import GeometricEmbedding
from .integral_transform import IntegralTransform
from .mlp import ChannelMLP, LinearChannelMLP


############
# Utils Functions
############
def parse_neighbor_strategy(neighbor_strategy: str | list[str]) -> tuple[str, str]:
    """
    Parse neighbor_strategy into encoder and decoder strategies.
    """
    if isinstance(neighbor_strategy, str):
        return neighbor_strategy, neighbor_strategy
    elif isinstance(neighbor_strategy, list) and len(neighbor_strategy) == 2:
        return neighbor_strategy[0], neighbor_strategy[1]
    else:
        raise ValueError(f"neighbor_strategy must be str or list of length 2, got {neighbor_strategy}")


def parse_geoembed_strategy(use_geoembed: bool | list[bool]) -> tuple[bool, bool]:
    """
    Parse use_geoembed into encoder and decoder settings.
    """
    if isinstance(use_geoembed, bool):
        return use_geoembed, use_geoembed
    elif isinstance(use_geoembed, list) and len(use_geoembed) == 2:
        return use_geoembed[0], use_geoembed[1]
    else:
        raise ValueError(f"use_geoembed must be bool or list of length 2, got {use_geoembed}")


def get_neighbor_strategy(
    neighbor_strategy: str,
    phys_pos: torch.Tensor,
    batch_idx_phys: torch.Tensor,
    latent_tokens_pos: torch.Tensor,
    batch_idx_latent: torch.Tensor,
    radius: float,
    k_neighbors: int = 1,
    is_decoder: bool = False,
):
    if is_decoder:
        return _get_decoder_strategy(
            neighbor_strategy, phys_pos, batch_idx_phys, latent_tokens_pos, batch_idx_latent, radius, k_neighbors
        )
    else:
        return _get_encoder_strategy(
            neighbor_strategy, phys_pos, batch_idx_phys, latent_tokens_pos, batch_idx_latent, radius, k_neighbors
        )


def _get_encoder_strategy(
    neighbor_strategy: str,
    phys_pos: torch.Tensor,
    batch_idx_phys: torch.Tensor,
    latent_tokens_pos: torch.Tensor,
    batch_idx_latent: torch.Tensor,
    radius: float,
    k_neighbors: int,
):
    device = phys_pos.device
    edge_index_knn = None
    edge_index_radius = None

    if neighbor_strategy in ["knn", "bidirectional"]:
        edge_index_knn = pyg_knn(x=latent_tokens_pos, y=phys_pos, k=k_neighbors, batch_x=batch_idx_latent, batch_y=batch_idx_phys)

    if neighbor_strategy in ["radius", "bidirectional"]:
        edge_index_radius_raw = pyg_radius(
            x=phys_pos,
            y=latent_tokens_pos,
            r=radius,
            batch_x=batch_idx_phys,
            batch_y=batch_idx_latent,
        )
        edge_index_radius = edge_index_radius_raw.flip(0)

    if neighbor_strategy == "knn":
        return edge_index_knn if edge_index_knn is not None else torch.empty((2, 0), dtype=torch.long, device=device)
    elif neighbor_strategy == "radius":
        return edge_index_radius if edge_index_radius is not None else torch.empty((2, 0), dtype=torch.long, device=device)
    elif neighbor_strategy == "bidirectional":
        edges = []
        if edge_index_knn is not None:
            edges.append(edge_index_knn)
        if edge_index_radius is not None:
            edges.append(edge_index_radius)
        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        elif len(edges) == 1:
            return edges[0]
        else:
            combined = torch.cat(edges, dim=1)
            return coalesce(combined)
    else:
        raise ValueError(f"Unknown encoder strategy: {neighbor_strategy}")


def _get_decoder_strategy(
    neighbor_strategy: str,
    phys_pos: torch.Tensor,
    batch_idx_phys: torch.Tensor,
    latent_tokens_pos: torch.Tensor,
    batch_idx_latent: torch.Tensor,
    radius: float,
    k_neighbors: int,
):
    device = phys_pos.device
    edge_index_knn = None
    edge_index_radius = None

    if neighbor_strategy in ["knn", "bidirectional"]:
        edge_index_knn_raw = pyg_knn(
            x=latent_tokens_pos, y=phys_pos, k=k_neighbors, batch_x=batch_idx_latent, batch_y=batch_idx_phys
        )
        edge_index_knn = edge_index_knn_raw.flip(0)

    if neighbor_strategy in ["radius", "bidirectional"]:
        edge_index_radius_raw = pyg_radius(
            x=latent_tokens_pos,
            y=phys_pos,
            r=radius,
            batch_x=batch_idx_latent,
            batch_y=batch_idx_phys,
        )
        edge_index_radius = edge_index_radius_raw.flip(0)

    if neighbor_strategy == "reverse":
        encoder_edges = _get_encoder_strategy(
            "bidirectional",
            phys_pos,
            batch_idx_phys,
            latent_tokens_pos,
            batch_idx_latent,
            radius,
            k_neighbors,
        )
        return encoder_edges.flip(0)

    if neighbor_strategy == "knn":
        return edge_index_knn if edge_index_knn is not None else torch.empty((2, 0), dtype=torch.long, device=device)
    elif neighbor_strategy == "radius":
        return edge_index_radius if edge_index_radius is not None else torch.empty((2, 0), dtype=torch.long, device=device)
    elif neighbor_strategy == "bidirectional":
        edges = []
        if edge_index_knn is not None:
            edges.append(edge_index_knn)
        if edge_index_radius is not None:
            edges.append(edge_index_radius)
        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        elif len(edges) == 1:
            return edges[0]
        else:
            combined = torch.cat(edges, dim=1)
            return coalesce(combined)
    else:
        raise ValueError(f"Unknown decoder strategy: {neighbor_strategy}")


def apply_neighbor_sampling(
    edge_index: torch.Tensor,
    num_query_nodes: int,
    device: torch.device,
    sampling_strategy: str | None = None,
    max_neighbors: int | None = None,
    sample_ratio: float | None = None,
    training: bool = True,
) -> torch.Tensor:
    if sampling_strategy is None:
        return edge_index

    num_total_original_edges = edge_index.shape[1]

    if num_query_nodes == 0 or num_total_original_edges == 0:
        return edge_index

    if sampling_strategy == "max_neighbors":
        if max_neighbors is None:
            raise ValueError("max_neighbors must be provided when using 'max_neighbors' sampling strategy")

        dest_nodes = edge_index[1]
        counts = torch.bincount(dest_nodes, minlength=num_query_nodes)
        needs_sampling_mask = counts > max_neighbors

        if not torch.any(needs_sampling_mask):
            return edge_index

        keep_mask = torch.ones(num_total_original_edges, dtype=torch.bool, device=device)
        queries_to_sample_idx = torch.where(needs_sampling_mask)[0]

        for i in queries_to_sample_idx:
            node_edge_mask = dest_nodes == i
            node_edge_indices = torch.where(node_edge_mask)[0]
            num_node_edges = len(node_edge_indices)

            perm = torch.randperm(num_node_edges, device=device)[:max_neighbors]
            edges_to_keep_for_node = node_edge_indices[perm]
            node_keep_mask = torch.zeros_like(node_edge_mask)
            node_keep_mask[edges_to_keep_for_node] = True
            keep_mask[node_edge_mask] = node_keep_mask[node_edge_mask]

        sampled_edge_index = edge_index[:, keep_mask]
        return sampled_edge_index

    elif sampling_strategy == "ratio":
        if sample_ratio is None:
            raise ValueError("sample_ratio must be provided when using 'ratio' sampling strategy")

        if sample_ratio >= 1.0:
            return edge_index
        p_drop = 1.0 - sample_ratio
        sampled_edge_index, _ = dropout_edge(edge_index, p=p_drop, force_undirected=False, training=training)
        return sampled_edge_index

    else:
        raise ValueError(f"Invalid sampling strategy: {sampling_strategy}")


############
# MAGNOEncoder
############
class MAGNOEncoder(nn.Module):
    def __init__(self, in_channels, magno_config: dict):
        super().__init__()
        self.magno_config = magno_config

        self.gno_radius = magno_config["gno_radius"]
        self.scales = magno_config.get("scales", [1.0])
        self.lifting_channels = magno_config["lifting_channels"]
        self.coord_dim = magno_config["gno_coord_dim"]
        self.feature_attr_name = magno_config.get("encoder_feature_attr", "x")
        self.precompute_edges = magno_config.get("precompute_edges", False)
        self.mlp_type = magno_config.get("mlp_type", "linear")

        # --- Store Neighbor Finding Strategy ---
        self.encoder_strategy, self.decoder_strategy = parse_neighbor_strategy(magno_config.get("neighbor_strategy", "radius"))
        self.k_neighbors = magno_config.get("k_neighbors", 1)

        # --- Store Sampling Strategy ---
        self.sampling_strategy = magno_config.get("sampling_strategy")
        self.max_neighbors = magno_config.get("max_neighbors")
        self.sample_ratio = magno_config.get("sample_ratio")
        if self.sampling_strategy == "max_neighbors":
            print("Warning: 'max_neighbors' sampling strategy with PyG edge_index is less efficient. Consider using 'ratio'.")

        # --- Init GNO Layer ---
        self.use_gno = magno_config.get("use_gno", True)
        self.lifting: LinearChannelMLP | None = None
        if self.use_gno:
            in_kernel_in_dim = self.coord_dim * 2
            if magno_config.get("in_gno_transform_type", "linear") in ["nonlinear", "nonlinear_kernelonly"]:
                in_kernel_in_dim += in_channels

            in_gno_channel_mlp_hidden_layers = list(magno_config.get("in_gno_channel_mlp_hidden_layers", [128, 128]))
            in_gno_channel_mlp_hidden_layers.insert(0, in_kernel_in_dim)
            in_gno_channel_mlp_hidden_layers.append(self.lifting_channels)

            self.gno = IntegralTransform(
                channel_mlp_layers=in_gno_channel_mlp_hidden_layers,
                transform_type=magno_config.get("in_gno_transform_type", "linear"),
                use_attn=magno_config.get("use_attn"),
                coord_dim=self.coord_dim,
                attention_type=magno_config.get("attention_type", "cosine"),
            )

            ## --- Init Lifting MLP ---
            if self.mlp_type == "linear":
                self.lifting = LinearChannelMLP(layers=[in_channels, self.lifting_channels])
            else:
                self.lifting = LinearChannelMLP(layers=[in_channels, self.lifting_channels])
        else:
            self.gno = IntegralTransform(
                channel_mlp_layers=[],
                transform_type="linear",
                use_attn=False,
                coord_dim=self.coord_dim,
                attention_type="cosine",
            )
            if in_channels > 0:
                self.lifting = LinearChannelMLP(layers=[in_channels, self.lifting_channels])
                print(
                    "Warning: MAGNOEncoder has input_channels > 0 but use_gno=False. Input features (batch.x) will be ignored by the encoder path."
                )
            else:
                self.lifting = None

        # --- Init GeoEmbed (optional) ---
        use_geoembed_encoder, _ = parse_geoembed_strategy(magno_config.get("use_geoembed", False))
        self.use_geoembed = use_geoembed_encoder
        self.recvery: LinearChannelMLP | ChannelMLP | None
        if self.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=self.coord_dim,
                output_dim=self.lifting_channels,
                method=magno_config.get("embedding_method", "statistical"),
                pooling=magno_config.get("pooling", "max"),
            )
            self.recovery: LinearChannelMLP | ChannelMLP
            if self.mlp_type == "linear":
                self.recovery = LinearChannelMLP(layers=[2 * self.lifting_channels, self.lifting_channels])
            else:
                self.recovery = ChannelMLP(in_channels=2 * self.lifting_channels, out_channels=self.lifting_channels, n_layers=1)

        # --- Init Scale Weighting (optional) ---
        self.use_scale_weights = magno_config.get("use_scale_weights", False)
        if self.use_scale_weights:
            self.num_scales = len(self.scales)
            self.scale_weighting = nn.Sequential(nn.Linear(self.coord_dim, 16), nn.ReLU(), nn.Linear(16, self.num_scales))
            self.scale_weight_activation = nn.Softmax(dim=-1)

    def forward(  # noqa: PLR0912, PLR0915
        self, batch: "pyg.data.Batch", latent_tokens_pos: torch.Tensor, latent_tokens_batch_idx: torch.Tensor
    ) -> torch.Tensor:
        phys_pos = batch.pos
        batch_idx_phys = batch.batch
        device = phys_pos.device
        num_graphs = batch.num_graphs
        num_latent_tokens_per_graph = latent_tokens_pos.shape[0] // num_graphs

        if isinstance(self.feature_attr_name, list):
            phys_feats = []
            for attr_name in self.feature_attr_name:
                feat = getattr(batch, attr_name, None)
                if feat is None:
                    if self.use_gno:
                        raise AttributeError(
                            f"MAGNOEncoder requires feature attribute '{attr_name}' but it was not found in the batch."
                        )
                else:
                    phys_feats.append(feat)
            phys_feat = torch.cat(phys_feats, dim=-1) if phys_feats else None
        else:
            phys_feat = getattr(batch, self.feature_attr_name, None)
            if phys_feat is None and self.use_gno:
                raise AttributeError(
                    f"MAGNOEncoder requires feature attribute '{self.feature_attr_name}' but it was not found in the batch."
                )

        # --- Multi-Scale GNO encoding ---
        encoded_scales = []
        for scale_idx, scale in enumerate(self.scales):
            scaled_radius = self.gno_radius * scale

            if self.precompute_edges:
                edge_index_attr = f"encoder_edge_index_s{scale_idx}"
                if not hasattr(batch, edge_index_attr):
                    raise AttributeError(f"Batch object missing pre-computed '{edge_index_attr}'")
                edge_index = getattr(batch, edge_index_attr).to(device)
                neighbor_counts = None
            else:
                edge_index = get_neighbor_strategy(
                    neighbor_strategy=self.encoder_strategy,
                    phys_pos=phys_pos,
                    batch_idx_phys=batch_idx_phys,
                    latent_tokens_pos=latent_tokens_pos,
                    batch_idx_latent=latent_tokens_batch_idx,
                    radius=scaled_radius,
                    k_neighbors=self.k_neighbors,
                    is_decoder=False,
                )
                neighbor_counts = None

            # --- Apply Neighbor Sampling ---
            edge_index = apply_neighbor_sampling(
                edge_index=edge_index,
                num_query_nodes=latent_tokens_pos.shape[0],
                device=device,
                sampling_strategy=self.sampling_strategy,
                max_neighbors=self.max_neighbors,
                sample_ratio=self.sample_ratio,
                training=self.training,
            )

            # --- Conditional GNO Path ---
            if self.use_gno:
                if self.mlp_type == "linear":
                    phys_feat_lifted = self.lifting(phys_feat) if phys_feat is not None and self.lifting is not None else None
                else:
                    phys_feat_lifted = (
                        self.lifting(phys_feat.transpose(0, 1)).transpose(0, 1)
                        if phys_feat is not None and self.lifting is not None
                        else None
                    )

                encoded_gno = self.gno(
                    y_pos=phys_pos,
                    x_pos=latent_tokens_pos,
                    edge_index=edge_index,
                    f_y=phys_feat_lifted,
                    batch_y=batch_idx_phys,
                    batch_x=latent_tokens_batch_idx,
                )
            else:
                encoded_gno = None

            # --- Conditional GeoEmbed Path ---
            if self.use_geoembed:
                geo_embedding = self.geoembed(
                    source_pos=phys_pos,
                    query_pos=latent_tokens_pos,
                    edge_index=edge_index,
                    batch_source=batch_idx_phys,
                    batch_query=latent_tokens_batch_idx,
                    neighbors_counts=neighbor_counts,
                )
            else:
                geo_embedding = None

            # --- Combine GNO and GeoEmbed ---
            if self.use_gno and self.use_geoembed:
                combined = torch.cat([encoded_gno, geo_embedding], dim=-1)
                if self.mlp_type == "linear":
                    encoded_unpatched = self.recovery(combined)
                else:
                    encoded_unpatched = self.recovery(combined.permute(1, 0)).permute(1, 0)
            elif self.use_gno:
                encoded_unpatched = encoded_gno
            elif self.use_geoembed:
                encoded_unpatched = geo_embedding
            else:
                raise ValueError("GNO and GeoEmbed are both disabled. No encoding will be performed.")

            encoded_scales.append(encoded_unpatched)

        # --- Aggregate Scales ---
        if len(encoded_scales) == 1:
            encoded_data = encoded_scales[0]
        else:
            encoded_stack = torch.stack(encoded_scales, dim=0)
            if self.use_scale_weights:
                scale_w = self.scale_weighting(latent_tokens_pos)
                scale_w = self.scale_weight_activation(scale_w)
                weights_reshaped = scale_w.permute(1, 0).unsqueeze(-1)
                encoded_data = (encoded_stack * weights_reshaped).sum(dim=0)
            else:
                encoded_data = encoded_stack.sum(dim=0)

        encoded_data = encoded_data.view(num_graphs, num_latent_tokens_per_graph, self.lifting_channels)

        return encoded_data


############
# MAGNODecoder
############
class MAGNODecoder(nn.Module):
    def __init__(self, in_channels, out_channels, magno_config: dict):
        super().__init__()
        self.magno_config = magno_config

        self.gno_radius = magno_config["gno_radius"]
        self.scales = magno_config.get("scales", [1.0])
        self.coord_dim = magno_config["gno_coord_dim"]
        self.in_channels = in_channels
        self.out_channels = out_channels
        _, use_geoembed_decoder = parse_geoembed_strategy(magno_config.get("use_geoembed", False))
        self.use_geoembed = use_geoembed_decoder
        self.use_scale_weights = magno_config.get("use_scale_weights", False)
        self.precompute_edges = magno_config.get("precompute_edges", False)
        self.mlp_type = magno_config.get("mlp_type", "linear")

        # --- Store Neighbor Strategy ---
        _, self.decoder_strategy = parse_neighbor_strategy(magno_config.get("neighbor_strategy", "radius"))
        self.k_neighbors = magno_config.get("k_neighbors", 1)

        # --- Store Sampling Strategy ---
        self.sampling_strategy = magno_config.get("sampling_strategy")
        self.max_neighbors = magno_config.get("max_neighbors")
        self.sample_ratio = magno_config.get("sample_ratio")
        if self.sampling_strategy == "max_neighbors":
            print("Warning: 'max_neighbors' sampling strategy with PyG edge_index is less efficient. Consider using 'ratio'.")

        # --- Calculate MLP input dimension ---
        out_kernel_in_dim = self.coord_dim * 2
        if magno_config.get("out_gno_transform_type", "linear") in ["nonlinear", "nonlinear_kernelonly"]:
            out_kernel_in_dim += self.in_channels

        # --- Init GNO Layer ---
        out_gno_channel_mlp_hidden_layers = list(magno_config.get("out_gno_channel_mlp_hidden_layers", [64, 64]))
        out_gno_channel_mlp_hidden_layers.insert(0, out_kernel_in_dim)
        out_gno_channel_mlp_hidden_layers.append(self.in_channels)

        self.gno = IntegralTransform(
            channel_mlp_layers=out_gno_channel_mlp_hidden_layers,
            transform_type=magno_config.get("out_gno_transform_type", "linear"),
            use_attn=magno_config.get("use_attn"),
            coord_dim=self.coord_dim,
            attention_type=magno_config.get("attention_type", "cosine"),
        )

        # --- Init Projection ---
        if self.mlp_type == "linear":
            self.projection: LinearChannelMLP | ChannelMLP = LinearChannelMLP(
                layers=[in_channels, magno_config.get("projection_channels", 256), out_channels]
            )

        # --- Init GeoEmbed (Optional) ---
        if self.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=self.coord_dim,
                output_dim=self.in_channels,
                method=magno_config.get("embedding_method", "statistical"),
                pooling=magno_config.get("pooling", "max"),
            )
            self.recovery: LinearChannelMLP | ChannelMLP
            if self.mlp_type == "linear":
                self.recovery = LinearChannelMLP(layers=[2 * in_channels, in_channels])
            else:
                self.recovery = ChannelMLP(in_channels=2 * in_channels, out_channels=in_channels, n_layers=1)

        # --- Init Scale Weighting (Optional) ---
        if self.use_scale_weights:
            self.num_scales = len(self.scales)
            self.scale_weighting = nn.Sequential(nn.Linear(self.coord_dim, 16), nn.ReLU(), nn.Linear(16, self.num_scales))
            self.scale_weight_activation = nn.Softmax(dim=-1)

    def forward(  # noqa: PLR0912
        self,
        rndata_flat: torch.Tensor,
        phys_pos_query: torch.Tensor,
        batch_idx_phys_query: torch.Tensor,
        latent_tokens_pos: torch.Tensor,
        latent_tokens_batch_idx: torch.Tensor,
        batch: "pyg.data.Batch" = None,
    ) -> torch.Tensor:
        device = rndata_flat.device

        # --- Multi-Scale GNO decoding ---
        decoded_scales = []
        for scale_idx, scale in enumerate(self.scales):
            scaled_radius = self.gno_radius * scale

            if self.precompute_edges:
                edge_index_attr = f"decoder_edge_index_s{scale_idx}"
                if not hasattr(batch, edge_index_attr):
                    raise AttributeError(f"Batch object missing pre-computed '{edge_index_attr}'")
                edge_index = getattr(batch, edge_index_attr).to(device)
                neighbor_counts = None
            else:
                edge_index = get_neighbor_strategy(
                    neighbor_strategy=self.decoder_strategy,
                    phys_pos=phys_pos_query,
                    batch_idx_phys=batch_idx_phys_query,
                    latent_tokens_pos=latent_tokens_pos,
                    batch_idx_latent=latent_tokens_batch_idx,
                    radius=scaled_radius,
                    k_neighbors=self.k_neighbors,
                    is_decoder=True,
                )
                neighbor_counts = None

            # --- Apply Neighbor Sampling ---
            edge_index = apply_neighbor_sampling(
                edge_index=edge_index,
                num_query_nodes=phys_pos_query.shape[0],
                device=device,
                sampling_strategy=self.sampling_strategy,
                max_neighbors=self.max_neighbors,
                sample_ratio=self.sample_ratio,
                training=self.training,
            )

            # GNO Layer Call
            decoded_unpatched = self.gno(
                y_pos=latent_tokens_pos,
                x_pos=phys_pos_query,
                edge_index=edge_index,
                f_y=rndata_flat,
                batch_y=latent_tokens_batch_idx,
                batch_x=batch_idx_phys_query,
            )

            # --- GeoEmbed (Optional) ---
            if self.use_geoembed:
                geoembedding = self.geoembed(
                    source_pos=latent_tokens_pos,
                    query_pos=phys_pos_query,
                    edge_index=edge_index,
                    batch_source=latent_tokens_batch_idx,
                    batch_query=batch_idx_phys_query,
                    neighbors_counts=neighbor_counts,
                )
                combined = torch.cat([decoded_unpatched, geoembedding], dim=-1)
                if self.mlp_type == "linear":
                    decoded_unpatched = self.recovery(combined)
                else:
                    decoded_unpatched = self.recovery(combined.permute(1, 0)).permute(1, 0)

            decoded_scales.append(decoded_unpatched)

        # --- Aggregate Scales ---
        if len(decoded_scales) == 1:
            decoded_data = decoded_scales[0]
        else:
            decoded_stack = torch.stack(decoded_scales, dim=0)
            if self.use_scale_weights:
                scale_w = self.scale_weighting(phys_pos_query)
                scale_w = self.scale_weight_activation(scale_w)
                weights_reshaped = scale_w.permute(1, 0).unsqueeze(-1)
                decoded_data = (decoded_stack * weights_reshaped).sum(dim=0)
            else:
                decoded_data = decoded_stack.sum(dim=0)

        # --- Final Projection ---
        if self.mlp_type == "linear":
            decoded_data = self.projection(decoded_data)
        else:
            decoded_data = decoded_data.permute(1, 0)
            decoded_data = self.projection(decoded_data).permute(1, 0)
        return decoded_data
