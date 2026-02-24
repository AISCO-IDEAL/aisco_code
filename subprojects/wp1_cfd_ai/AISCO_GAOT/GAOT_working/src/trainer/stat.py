import logging
import os
import statistics
import time

import numpy as np
import torch
import torch_geometric as pyg
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm

from src.dataloader.collate_functions import create_collate_function
from src.dataloader.pyg_datasets import EnrichedData, VTKMeshDataset
from src.dataloader.pyg_transforms import NormalizeFeatures, RescalePosition, RescalePositionNew
from src.model.gaot_3d import GAOT3D, masked_multi_head_loss
from src.utils.scale import rescale, rescale_new

from .base import TrainerBase
from .utils.metric import compute_drivaernet_metric
from .utils.plot import plot_3d_comparison_matplotlib, plot_3d_comparison_pyvista

EPSILON = 1e-10


def move_to_device(data, device):
    """Recursively move all tensors in a nested structure to the specified device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    else:
        return data


def _magno_get(model_config, key, default=None):
    """Helper to access magno config keys whether magno is a dict or dataclass."""
    magno = model_config.args.magno
    if isinstance(magno, dict):
        return magno.get(key, default)
    else:
        return getattr(magno, key, default)


def _magno_set(model_config, key, value):
    """Helper to set magno config keys whether magno is a dict or dataclass."""
    magno = model_config.args.magno
    if isinstance(magno, dict):
        magno[key] = value
    else:
        setattr(magno, key, value)


def _build_gaot_config(model_config, input_size: int, output_size: int) -> dict:
    """
    Build the flat config dict that GAOT3D expects from the structured
    ModelConfig dataclass.  Runtime-detected input_size / output_size
    override whatever is stored in the YAML.
    """
    args = model_config.args

    # args may be a dataclass (ModelArgsConfig) or already a dict
    if hasattr(args, "__dataclass_fields__"):
        from dataclasses import asdict

        cfg = asdict(args)
    elif isinstance(args, dict):
        cfg = dict(args)
    else:
        # OmegaConf DictConfig or similar
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(args, resolve=True)

    # Inject runtime-detected sizes
    cfg["input_size"] = input_size
    cfg["output_size"] = output_size

    return cfg


class StaticTrainer3D(TrainerBase):
    """
    Trainer for static problems, i.e. problems that do not depend on time.
    """

    def __init__(self, args):
        super().__init__(args)
        if self.dataset_config.training_strategy == "neural_field":
            if self.dataset_config.update_pt_files_with_edges:
                raise ValueError("neural_field training strategy requires update_pt_files_with_edges=False")
            if _magno_get(self.model_config, "precompute_edges", True):
                logging.getLogger(__name__).warning(
                    "WARNING: neural_field training strategy requires precompute_edges=False, setting it to False automatically."
                )
                _magno_set(self.model_config, "precompute_edges", False)

    def _calculate_or_load_stats(self, dataset_config, order_file_path, data_root):
        """Calculates or loads normalization statistics."""
        stats_file = os.path.join(data_root, f"{dataset_config.name}_norm_stats.pt")

        if os.path.exists(stats_file) and not getattr(dataset_config, "force_recompute_stats", False):
            logging.getLogger(__name__).info(f"Loading pre-calculated normalization stats from {stats_file}")
            stats = torch.load(stats_file, weights_only=False)
            self.u_mean = stats["mean"].to(self.dtype)
            self.u_std = stats["std"].to(self.dtype)
            logging.getLogger(__name__).info(f"Loaded x - Mean: {self.u_mean}, Std: {self.u_std}")

            if "c_mean" in stats and "c_std" in stats:
                self.c_mean = stats["c_mean"].to(self.dtype)
                self.c_std = stats["c_std"].to(self.dtype)
                logging.getLogger(__name__).info(f"Loaded c - Mean: {self.c_mean}, Std: {self.c_std}")
            else:
                self.c_mean = None
                self.c_std = None
            self.per_head_stats = stats.get("per_head_stats", {})
        else:
            logging.getLogger(__name__).info("Calculating normalization statistics from training set...")
            temp_train_dataset = VTKMeshDataset(
                root=dataset_config.base_path,
                order_file=order_file_path,
                dataset_config=dataset_config,
                split="train",
                transform=RescalePosition(),
            )

            temp_loader = PyGDataLoader(temp_train_dataset, batch_size=dataset_config.batch_size, shuffle=False, num_workers=4)

            all_x = []
            all_c = []
            all_force = []
            all_cd = []
            has_c_field = False

            for batch in tqdm(temp_loader, desc="Calculating Stats", disable=(getattr(self.setup_config, "rank", 0) != 0)):
                # Multi-head: collect per-head stats (force and cd separately)
                if hasattr(batch, "force") and batch.force is not None:
                    valid = batch.force[~torch.isnan(batch.force)]
                    if valid.numel() > 0:
                        all_force.append(valid.cpu())
                if hasattr(batch, "cd") and batch.cd is not None:
                    valid = batch.cd[~torch.isnan(batch.cd)]
                    if valid.numel() > 0:
                        all_cd.append(valid.cpu())
                # Fallback: if data still uses old .y format
                if hasattr(batch, "y") and batch.y is not None:
                    all_x.append(batch.y.cpu())
                if hasattr(batch, "c") and batch.c is not None:
                    all_c.append(batch.c.cpu())
                    has_c_field = True

            # Compute per-head stats for multi-head model
            self.per_head_stats = {}
            if all_force:
                force_tensor = torch.cat(all_force, dim=0)
                self.per_head_stats["force"] = {
                    "mean": torch.mean(force_tensor, dtype=self.dtype),
                    "std": torch.std(force_tensor).to(self.dtype),
                }
                logging.getLogger(__name__).info(
                    f"Force stats - Mean: {self.per_head_stats['force']['mean']}, Std: {self.per_head_stats['force']['std']}"
                )
            if all_cd:
                cd_tensor = torch.cat(all_cd, dim=0)
                self.per_head_stats["cd"] = {
                    "mean": torch.mean(cd_tensor, dtype=self.dtype),
                    "std": torch.std(cd_tensor).to(self.dtype),
                }
                logging.getLogger(__name__).info(
                    f"Cd stats - Mean: {self.per_head_stats['cd']['mean']}, Std: {self.per_head_stats['cd']['std']}"
                )

            # Build unified u_mean/u_std from whatever data is available
            if all_x:
                # Old-format data with .y
                combined = all_x
            elif all_force or all_cd:
                # Multi-head: combine all valid values for a rough unified stat
                combined = all_force + all_cd
            else:
                raise ValueError("No data found in training set to calculate statistics.")

            full_x_tensor = torch.cat(combined, dim=0)
            self.u_mean = torch.mean(full_x_tensor, dim=0, dtype=self.dtype)
            self.u_std = torch.std(full_x_tensor, dim=0).to(self.dtype)

            if has_c_field and all_c:
                full_c_tensor = torch.cat(all_c, dim=0)
                self.c_mean = torch.mean(full_c_tensor, dim=0, dtype=self.dtype)
                self.c_std = torch.std(full_c_tensor, dim=0).to(self.dtype)
                logging.getLogger(__name__).info(f"Calculated c - Mean: {self.c_mean}, Std: {self.c_std}")
            else:
                self.c_mean = None
                self.c_std = None

            logging.getLogger(__name__).info(f"Calculated x - Mean: {self.u_mean}, Std: {self.u_std}")
            logging.getLogger(__name__).info(f"Saving normalization stats to {stats_file}")
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)

            stats_to_save = {"mean": self.u_mean, "std": self.u_std}
            if self.c_mean is not None:
                stats_to_save["c_mean"] = self.c_mean
                stats_to_save["c_std"] = self.c_std
            if self.per_head_stats:
                stats_to_save["per_head_stats"] = self.per_head_stats
            torch.save(stats_to_save, stats_file)
        if self.u_mean is None or self.u_std is None:
            raise RuntimeError("Normalization mean/std could not be calculated or loaded.")
        self.u_mean = self.u_mean.to(self.dtype)
        self.u_std = self.u_std.to(self.dtype)

    def _update_pt_files_with_edges(self, dataset_config, order_file_path, magno_cfg: dict):
        """
        Iterates through processed .pt files, computes edges (and optionally counts),
        and saves them back to the files. Runs ONLY on rank 0 in DDP.
        """
        if self.setup_config.rank != 0:
            return

        processed_dir = os.path.join(dataset_config.base_path, dataset_config.processed_folder)
        if not os.path.isdir(processed_dir):
            raise FileNotFoundError(f"Processed directory for update not found: {processed_dir}")

        with open(order_file_path) as f:
            all_filenames_base = [line.strip() for line in f if line.strip()]
        total_samples = len(all_filenames_base)
        train_size = dataset_config.train_size
        val_size = dataset_config.val_size
        test_size = dataset_config.test_size
        indices = np.arange(total_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[-test_size:]
        relevant_indices = np.concatenate([train_indices, val_indices, test_indices])
        filenames_to_process = [f"{all_filenames_base[i]}.pt" for i in relevant_indices]

        logging.getLogger(__name__).info(
            f"Rank 0: Checking/Updating {len(filenames_to_process)} '.pt' files in {processed_dir} with edge information..."
        )

        latent_tokens_cpu = self.latent_tokens.cpu()
        num_latent_tokens = latent_tokens_cpu.shape[0]

        from src.model.layers.magno import get_neighbor_strategy, parse_neighbor_strategy

        # Extract magno config values (dict access)
        gno_radius = magno_cfg.get("gno_radius", 0.15)
        scales = magno_cfg.get("scales", [1.0])
        neighbor_strategy = magno_cfg.get("neighbor_strategy", "radius")
        k_neighbors = magno_cfg.get("k_neighbors", 1)

        for filename in tqdm(filenames_to_process, desc="Updating .pt files"):
            fpath = os.path.join(processed_dir, filename)
            fpath_tmp = fpath + ".tmp"
            try:
                data_old = torch.load(fpath, map_location="cpu", weights_only=False)
                # Multi-head: .y may not exist; use .force or .cd as x placeholder
                x_attr = (
                    data_old.y if hasattr(data_old, "y") and data_old.y is not None else torch.zeros(data_old.pos.shape[0], 1)
                )
                data = EnrichedData(pos=data_old.pos, x=x_attr)
                for attr, value in data_old:
                    if attr not in ["pos", "y", "force", "cd"]:
                        setattr(data, attr, value)
                data.num_latent_nodes = num_latent_tokens
                if dataset_config.use_rescale_new:
                    phys_pos = rescale_new(data.pos.to(torch.float), (-1, 1), self.metadata.domain_x)
                else:
                    phys_pos = rescale(data.pos.to(torch.float), (-1, 1))  # Rescale to [-1, 1], very important!
                num_physical_nodes = phys_pos.shape[0]
                batch_idx_phys = torch.zeros(num_physical_nodes, dtype=torch.long)
                batch_idx_latent = torch.zeros(num_latent_tokens, dtype=torch.long)

                for scale_idx, scale in enumerate(scales):
                    scaled_radius = gno_radius * scale
                    # --- Parse neighbor strategy ---
                    encoder_strategy, decoder_strategy = parse_neighbor_strategy(neighbor_strategy)

                    # --- Encoder Edges ---
                    enc_edge_index = get_neighbor_strategy(
                        neighbor_strategy=encoder_strategy,
                        phys_pos=phys_pos,
                        batch_idx_phys=batch_idx_phys,
                        latent_tokens_pos=latent_tokens_cpu,
                        batch_idx_latent=batch_idx_latent,
                        radius=scaled_radius,
                        k_neighbors=k_neighbors,
                        is_decoder=False,
                    ).to(torch.int32)
                    setattr(data, f"encoder_edge_index_s{scale_idx}", enc_edge_index)
                    if enc_edge_index.numel() > 0:
                        enc_counts = torch.bincount(enc_edge_index[1], minlength=num_latent_tokens).to(torch.int32)
                    else:
                        enc_counts = torch.zeros(num_latent_tokens, dtype=torch.int32)
                    setattr(data, f"encoder_query_counts_s{scale_idx}", enc_counts)
                    # --- Decoder Edges ---
                    dec_edge_index = get_neighbor_strategy(
                        neighbor_strategy=decoder_strategy,
                        phys_pos=phys_pos,
                        batch_idx_phys=batch_idx_phys,
                        latent_tokens_pos=latent_tokens_cpu,
                        batch_idx_latent=batch_idx_latent,
                        radius=scaled_radius,
                        k_neighbors=k_neighbors,
                        is_decoder=True,
                    ).to(torch.int32)
                    setattr(data, f"decoder_edge_index_s{scale_idx}", dec_edge_index)
                    if dec_edge_index.numel() > 0:
                        dec_counts = torch.bincount(dec_edge_index[1], minlength=num_physical_nodes).to(torch.int32)
                    else:
                        dec_counts = torch.zeros(num_physical_nodes, dtype=torch.int32)
                    setattr(data, f"decoder_query_counts_s{scale_idx}", dec_counts)
                # --- Save updated data ---
                torch.save(data, fpath_tmp)
                os.replace(fpath_tmp, fpath)
            except FileNotFoundError:
                logging.getLogger(__name__).warning(f"Warning: File not found during update: {fpath}")
            except Exception as e:
                logging.getLogger(__name__).error(f"Error processing file {fpath}: {e}")
                if os.path.exists(fpath_tmp):
                    os.remove(fpath_tmp)
        logging.getLogger(__name__).info("Rank 0: Finished checking/updating '.pt' files.")

    def init_dataset(self, dataset_config):
        logging.getLogger(__name__).info("Initializing PyG dataset ...")
        self.latent_token_size = self.model_config.args.latent_tokens
        data_root = dataset_config.base_path
        order_file_path = os.path.join(data_root, f"order_{dataset_config.processed_folder}.txt")
        processed_data_path = os.path.join(data_root, dataset_config.processed_folder)

        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"Processed data directory does not exist: {processed_data_path}")
        if not os.path.exists(order_file_path):
            raise FileNotFoundError(f"Order file does not exist: {order_file_path}")

        # --- Latent Tokens ---
        phy_domain = self.metadata.domain_x
        x_min, y_min, z_min = phy_domain[0]
        x_max, y_max, z_max = phy_domain[1]
        meshgrid = torch.meshgrid(
            torch.linspace(x_min, x_max, self.latent_token_size[0]),
            torch.linspace(y_min, y_max, self.latent_token_size[1]),
            torch.linspace(z_min, z_max, self.latent_token_size[2]),
            indexing="ij",
        )
        latent_queries = torch.stack(meshgrid, dim=-1).reshape(-1, 3)
        if dataset_config.use_rescale_new:
            self.latent_tokens = rescale_new(latent_queries, (-1, 1), phy_domain)
        else:
            self.latent_tokens = rescale(latent_queries, (-1, 1))

        logging.getLogger(__name__).info(f"Rank {self.setup_config.rank}: Dataset initialization finished.")

        # --- Pre-computation /Update Step ---
        if dataset_config.update_pt_files_with_edges:
            magno_cfg = self.model_config.args.magno
            if self.setup_config.rank == 0:
                self._update_pt_files_with_edges(dataset_config, order_file_path, magno_cfg)
            if self.setup_config.distributed:
                logging.getLogger(__name__).info(f"Rank {self.setup_config.rank}: Waiting for edge pre-computation barrier...")
                torch.distributed.barrier()
                logging.getLogger(__name__).info(f"Rank {self.setup_config.rank}: Barrier passed.")

            _magno_set(self.model_config, "precompute_edges", True)
            logging.getLogger(__name__).info(f"Rank {self.setup_config.rank}: Set model config to use precomputed edges.")
        # --- end Pre-computation /Update Step ---

        # --- Calculate or Load Normalization Stats ---
        self._calculate_or_load_stats(dataset_config, order_file_path, data_root)

        # --- Define Transforms ---
        if dataset_config.use_rescale_new:
            rescale_transform = RescalePositionNew(lims=(-1.0, 1.0), phy_domain=phy_domain)
        else:
            rescale_transform = RescalePosition(lims=(-1.0, 1.0))

        # Apply active_variables selection to normalization stats if specified
        if dataset_config.active_variables is not None:
            mean_for_norm = self.u_mean[dataset_config.active_variables]
            std_for_norm = self.u_std[dataset_config.active_variables]
        else:
            mean_for_norm = self.u_mean
            std_for_norm = self.u_std

        # Prepare c normalization stats if they exist
        c_mean_for_norm = None
        c_std_for_norm = None
        if self.c_mean is not None and self.c_std is not None:
            c_mean_for_norm = self.c_mean
            c_std_for_norm = self.c_std

        normalize_transform = NormalizeFeatures(
            mean=mean_for_norm,
            std=std_for_norm,
            c_mean=c_mean_for_norm,
            c_std=c_std_for_norm,
            per_head_stats=self.per_head_stats if hasattr(self, "per_head_stats") else None,
        )
        composed_transform = Compose([rescale_transform, normalize_transform])

        # --- Create collate function for online graph building in DataLoader ---
        magno_cfg = self.model_config.args.magno
        collate_fn = create_collate_function(
            coord_dim=magno_cfg["gno_coord_dim"] if isinstance(magno_cfg, dict) else magno_cfg.gno_coord_dim,
            magno_radius=magno_cfg["gno_radius"] if isinstance(magno_cfg, dict) else magno_cfg.gno_radius,
            magno_scales=magno_cfg["scales"] if isinstance(magno_cfg, dict) else magno_cfg.scales,
            latent_tokens=self.latent_tokens,
            neighbor_search_method=magno_cfg["neighbor_strategy"] if isinstance(magno_cfg, dict) else magno_cfg.neighbor_strategy,
            k_neighbors=magno_cfg["k_neighbors"] if isinstance(magno_cfg, dict) else magno_cfg.k_neighbors,
            asynchronous_graph_building=magno_cfg.get("asynchronous_graph_building", False)
            if isinstance(magno_cfg, dict)
            else getattr(magno_cfg, "asynchronous_graph_building", False),
        )
        # Ensure the model expects precomputed edges coming from collate
        async_graph = (
            magno_cfg.get("asynchronous_graph_building", False)
            if isinstance(magno_cfg, dict)
            else getattr(magno_cfg, "asynchronous_graph_building", False)
        )
        if async_graph:
            _magno_set(self.model_config, "precompute_edges", True)
            logging.getLogger(__name__).info("Enabled precompute_edges for MAGNO to consume collate-built graphs.")

        if self.setup_config.train:
            logging.getLogger(__name__).info(f"Rank {self.setup_config.rank}: Loading train dataset...")
            train_ds = VTKMeshDataset(
                root=data_root,
                order_file=order_file_path,
                dataset_config=dataset_config,
                split="train",
                transform=composed_transform,
            )
            logging.getLogger(__name__).info(f"Rank {self.setup_config.rank}: Loading validation dataset...")
            val_ds = VTKMeshDataset(
                root=data_root,
                order_file=order_file_path,
                dataset_config=dataset_config,
                split="val",
                transform=composed_transform,
            )
        logging.getLogger(__name__).info(f"Rank {self.setup_config.rank}: Loading test dataset...")
        test_ds = VTKMeshDataset(
            root=data_root, order_file=order_file_path, dataset_config=dataset_config, split="test", transform=composed_transform
        )

        encoder_feature_attr = _magno_get(self.model_config, "encoder_feature_attr", "x")
        if isinstance(encoder_feature_attr, list):
            self.num_input_channels = 0
            for attr_name in encoder_feature_attr:
                if hasattr(test_ds[0], attr_name):
                    self.num_input_channels += getattr(test_ds[0], attr_name).shape[1]
                else:
                    logging.getLogger(__name__).warning(f"Attribute '{attr_name}' not found in test dataset, using pos instead.")
                    self.num_input_channels += test_ds[0].pos.shape[1]
        elif hasattr(test_ds[0], encoder_feature_attr):
            self.num_input_channels = getattr(test_ds[0], encoder_feature_attr).shape[1]
        else:
            logging.getLogger(__name__).warning(
                f"Attribute '{encoder_feature_attr}' not found in test dataset, using pos instead."
            )
            self.num_input_channels = test_ds[0].pos.shape[1]
        # Multi-head: output channels determined by model heads, not data.y
        if hasattr(test_ds[0], "y") and test_ds[0].y is not None:
            self.num_output_channels = test_ds[0].y.shape[1] if test_ds[0].y.dim() > 1 else 1
        else:
            # Multi-head decoder: count total output dims across heads
            self.num_output_channels = 2  # force(1) + cd(1)
        # --- Create DataLoaders ---
        if self.setup_config.train:
            ## --- Create DistributedSampler for training ---
            train_sampler = None
            if self.setup_config.distributed:
                train_sampler = DistributedSampler(
                    train_ds,
                    num_replicas=self.setup_config.world_size,
                    rank=self.setup_config.rank,
                    shuffle=dataset_config.shuffle,  # Sampler handles shuffling
                    drop_last=True,  # Often good practice for DDP
                )
                logging.getLogger(__name__).info(f"Rank {self.setup_config.rank}: Created DistributedSampler for training.")

            self.train_loader = DataLoader(
                train_ds,
                batch_size=dataset_config.batch_size,
                shuffle=False if train_sampler is not None else dataset_config.shuffle,
                num_workers=dataset_config.num_workers,
                sampler=train_sampler,
                pin_memory=False,
                drop_last=train_sampler is not None,
                collate_fn=collate_fn,
            )

            val_sampler = None
            if self.setup_config.distributed:
                val_sampler = DistributedSampler(
                    val_ds, num_replicas=self.setup_config.world_size, rank=self.setup_config.rank, shuffle=False, drop_last=False
                )
                logging.getLogger(__name__).info(f"Rank {self.setup_config.rank}: Created DistributedSampler for validation.")
            self.val_loader = DataLoader(
                val_ds,
                batch_size=dataset_config.batch_size,
                shuffle=False,
                num_workers=dataset_config.num_workers,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )

        test_sampler = None
        self.test_loader = DataLoader(
            test_ds,
            batch_size=dataset_config.batch_size,
            shuffle=False,
            num_workers=dataset_config.num_workers,
            pin_memory=False,
            sampler=test_sampler,
            collate_fn=collate_fn,
        )

    def init_model(self, model_config):
        config_dict = _build_gaot_config(
            model_config,
            input_size=self.num_input_channels,
            output_size=self.num_output_channels,
        )

        self.model = GAOT3D(config=config_dict)
        self.model.to(self.device)

        if self.setup_config.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.setup_config.local_rank],
                output_device=self.setup_config.local_rank,
                find_unused_parameters=True,
            )

    def _sample_nodes_neural_field(self, batch: "pyg.data.Batch", num_input_nodes: int, num_query_nodes: int):
        """
        Sample nodes for neural field training strategy.
        """
        sampled_data_list = []
        query_pos_list = []
        query_batch_idx_list = []
        target_for_loss_list = []

        use_same_sampling = num_input_nodes == num_query_nodes

        for i in range(batch.num_graphs):
            start_idx = batch.ptr[i]
            end_idx = batch.ptr[i + 1]
            num_full_nodes_i = end_idx - start_idx

            full_pos_i = batch.pos[start_idx:end_idx]
            full_y_i = batch.y[start_idx:end_idx]

            full_c_i = None
            if hasattr(batch, "c") and batch.c is not None:
                full_c_i = batch.c[start_idx:end_idx]

            n_input_sample = min(num_input_nodes, num_full_nodes_i)
            if n_input_sample <= 0:
                continue

            input_perm = torch.randperm(num_full_nodes_i)[:n_input_sample]

            if use_same_sampling:
                query_perm = input_perm
                n_query_sample = n_input_sample
            else:
                n_query_sample = min(num_query_nodes, num_full_nodes_i)
                query_perm = torch.randperm(num_full_nodes_i)[:n_query_sample]

            sampled_pos_i = full_pos_i[input_perm]
            sampled_y_i = full_y_i[input_perm]

            data_i = Data(pos=sampled_pos_i, y=sampled_y_i)
            if full_c_i is not None:
                data_i.c = full_c_i[input_perm]

            essential_attrs = ["filename", "num_latent_nodes"]
            for attr in essential_attrs:
                if hasattr(batch, attr) and getattr(batch, attr) is not None:
                    if isinstance(getattr(batch, attr), list):
                        setattr(data_i, attr, getattr(batch, attr)[i])
                    else:
                        attr_value = getattr(batch, attr)
                        if attr_value.dim() > 0 and len(attr_value) == batch.num_graphs:
                            setattr(data_i, attr, attr_value[i])

            sampled_data_list.append(data_i)

            query_pos_list.append(full_pos_i[query_perm])
            query_batch_idx_list.append(torch.full((n_query_sample,), i, dtype=torch.long))
            target_for_loss_list.append(full_y_i[query_perm])

        sampled_batch = Batch.from_data_list(sampled_data_list)

        query_pos = torch.cat(query_pos_list, dim=0)
        query_batch_idx = torch.cat(query_batch_idx_list, dim=0)
        target_for_loss = torch.cat(target_for_loss_list, dim=0)

        return sampled_batch, query_pos, query_batch_idx, target_for_loss

    def train_step(self, batch: "pyg.data.Batch") -> torch.Tensor:
        strategy = self.dataset_config.training_strategy
        latent_tokens_dev = self.latent_tokens.to(self.device)

        if strategy == "neural_field":
            # --- Neural Field Strategy ---
            num_input_nodes = self.dataset_config.neural_field_input_nodes
            num_query_nodes = self.dataset_config.neural_field_query_nodes_train

            sampled_batch, query_pos, query_batch_idx, target_for_loss = self._sample_nodes_neural_field(
                batch, num_input_nodes, num_query_nodes
            )

            sampled_batch = sampled_batch.to(self.device)
            query_pos = query_pos.to(self.device)
            query_batch_idx = query_batch_idx.to(self.device)
            target_for_loss = target_for_loss.to(self.device)

            pred = self.model(
                batch=sampled_batch,
                tokens_pos=latent_tokens_dev,
                query_coord_pos=query_pos,
                query_coord_batch_idx=query_batch_idx,
            )

            target = target_for_loss
        else:
            batch = batch.to(self.device)
            pred = self.model(batch=batch, tokens_pos=latent_tokens_dev)
            target = batch.y

        # Multi-head output: pred is a dict {"force": [B,1], "cd": [B,1]}
        if isinstance(pred, dict):
            targets = {
                "force": getattr(batch, "force", None),
                "cd": getattr(batch, "cd", None),
            }
            return masked_multi_head_loss(pred, targets)
        else:
            return self.loss_fn(pred, target)

    def validate(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                latent_tokens_dev = self.latent_tokens.to(self.device)
                pred = self.model(batch=batch, tokens_pos=latent_tokens_dev)

                # Multi-head output: pred is a dict {"force": [B,1], "cd": [B,1]}
                if isinstance(pred, dict):
                    targets = {
                        "force": getattr(batch, "force", None),
                        "cd": getattr(batch, "cd", None),
                    }
                    loss = masked_multi_head_loss(pred, targets)
                else:
                    target = batch.y
                    loss = self.loss_fn(pred, target)

                total_loss += loss.item()
        # Global reduction across ranks (average per-dataset, not per-batch)
        local_sum = torch.tensor([total_loss], dtype=torch.float64, device=self.device)
        if getattr(self.setup_config, "distributed", False):
            torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
        denom = len(loader.dataset)
        return local_sum.item() / denom

    def test(self):
        self.model.eval()
        metric_suite = self.dataset_config.metric_suite

        # Determine if model uses multi-head output
        _raw_model = self.model.module if hasattr(self.model, "module") else self.model
        _is_multi_head = hasattr(_raw_model, "decoder") and hasattr(_raw_model.decoder, "head_names")
        if _is_multi_head:
            all_batch_targets_denorm = {"force": [], "cd": []}
            all_batch_preds_denorm = {"force": [], "cd": []}
        else:
            all_batch_targets_denorm = []
            all_batch_preds_denorm = []
        plot_coords, plot_gtr, plot_prd = None, None, None

        # Inference speed measurement variables
        inference_times = []
        data_loading_times = []
        total_samples = 0

        logging.getLogger(__name__).info(f"Starting testing with metric suite: '{metric_suite}'")

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                data_start_time = time.time()
                batch = batch.to(self.device)
                latent_tokens_dev = self.latent_tokens.to(self.device)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                data_end_time = time.time()
                data_loading_time = data_end_time - data_start_time
                data_loading_times.append(data_loading_time)

                inference_start_time = time.time()
                pred_norm = self.model(batch, latent_tokens_dev)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                inference_end_time = time.time()
                inference_time = inference_end_time - inference_start_time
                inference_times.append(inference_time)

                batch_size = batch.num_graphs if hasattr(batch, "num_graphs") else 1
                total_samples += batch_size

                # Handle multi-head predictions
                if isinstance(pred_norm, dict):
                    pred_denorm = {}
                    target_denorm_dict = {}
                    for head_name, head_pred in pred_norm.items():
                        head_target = getattr(batch, head_name, None)
                        if head_target is None:
                            continue
                        head_target = head_target.to(self.device)
                        if hasattr(self, "per_head_stats") and head_name in self.per_head_stats:
                            h_mean = self.per_head_stats[head_name]["mean"].to(self.device)
                            h_std = self.per_head_stats[head_name]["std"].to(self.device)
                            pred_denorm[head_name] = head_pred * h_std + h_mean
                            target_denorm_dict[head_name] = head_target * h_std + h_mean
                        else:
                            pred_denorm[head_name] = head_pred
                            target_denorm_dict[head_name] = head_target

                    if i == 0 and self.setup_config.rank == 0:
                        plot_coords = None
                        parts = []
                        if "force" in pred_denorm:
                            parts.append(f"Force={pred_denorm['force'][0].item():.4f}")
                        if "cd" in pred_denorm:
                            parts.append(f"Cd={pred_denorm['cd'][0].item():.4f}")
                        logging.getLogger(__name__).info(f"First test sample predictions (denorm): {', '.join(parts)}")

                    for head_name in pred_denorm:
                        head_target_dn = target_denorm_dict[head_name]
                        head_pred_dn = pred_denorm[head_name]
                        valid = ~torch.isnan(head_target_dn)
                        if valid.any():
                            all_batch_targets_denorm[head_name].append(head_target_dn[valid].cpu())
                            all_batch_preds_denorm[head_name].append(head_pred_dn[valid].detach().cpu())
                else:
                    target_norm = batch.y

                    u_std_dev = self.u_std[self.dataset_config.active_variables].to(self.device)
                    u_mean_dev = self.u_mean[self.dataset_config.active_variables].to(self.device)
                    pred_de_norm = pred_norm * u_std_dev + u_mean_dev
                    target_de_norm = target_norm * u_std_dev + u_mean_dev

                    all_batch_targets_denorm.append(target_de_norm.cpu())
                    all_batch_preds_denorm.append(pred_de_norm.cpu())

                    if i == 0 and self.setup_config.rank == 0:
                        plot_coords = None
                        plot_gtr = np.array([target_de_norm[0].item()])
                        plot_prd = np.array([pred_de_norm[0].item()])

                        logging.getLogger(__name__).info(
                            f"Extracted first test sample: True Drag = {plot_gtr[0]:.4f}, Predicted Drag = {plot_prd[0]:.4f}"
                        )

                if (i + 1) % 10 == 0 and self.setup_config.rank == 0:
                    current_avg_inference_time = statistics.mean(inference_times[-10:])
                    throughput = 1.0 / current_avg_inference_time
                    logging.getLogger(__name__).info(
                        f"Processed {i + 1} batches, "
                        f"Current average inference time: {current_avg_inference_time * 1000:.2f}ms, "
                        f"Inference throughput: {throughput:.2f} samples/s"
                    )

        if self.setup_config.rank == 0:
            if isinstance(all_batch_preds_denorm, dict):
                for head_name in all_batch_preds_denorm:
                    if not all_batch_preds_denorm[head_name]:
                        continue
                    head_preds = torch.cat(all_batch_preds_denorm[head_name], dim=0)
                    head_targets = torch.cat(all_batch_targets_denorm[head_name], dim=0)
                    diff = head_preds - head_targets
                    mse = torch.mean(diff**2).item()
                    mae = torch.mean(torch.abs(diff)).item()
                    logging.getLogger(__name__).info(
                        f"--- Head '{head_name}': MSE={mse:.6f}, MAE={mae:.6f}, n_samples={head_preds.shape[0]} ---"
                    )

                    results_array = torch.cat([head_targets.view(-1, 1), head_preds.view(-1, 1)], dim=1).cpu().numpy()
                    csv_path = self.path_config.result_path.replace(".png", f"_{head_name}_predictions.csv")
                    np.savetxt(
                        csv_path, results_array, delimiter=",", header=f"True_{head_name},Predicted_{head_name}", comments=""
                    )
                    logging.getLogger(__name__).info(f"Saved {head_name} predictions to {csv_path}")

                full_preds = torch.tensor([0.0])
                full_targets = torch.tensor([0.0])
                for head_name in all_batch_preds_denorm:
                    if all_batch_preds_denorm[head_name]:
                        full_preds = torch.cat(all_batch_preds_denorm[head_name], dim=0)
                        full_targets = torch.cat(all_batch_targets_denorm[head_name], dim=0)
                        break
            else:
                full_preds = torch.cat(all_batch_preds_denorm, dim=0)
                full_targets = torch.cat(all_batch_targets_denorm, dim=0)
            logging.getLogger(__name__).info(
                f"Concatenated results: preds shape {full_preds.shape}, targets shape {full_targets.shape}"
            )

            # --- Inference Speed Statistics Report ---
            if inference_times:
                total_inference_time = sum(inference_times)
                avg_inference_time = statistics.mean(inference_times)
                median_inference_time = statistics.median(inference_times)
                min_inference_time = min(inference_times)
                max_inference_time = max(inference_times)
                std_inference_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0

                total_data_loading_time = sum(data_loading_times)
                avg_data_loading_time = statistics.mean(data_loading_times)

                total_test_time = total_inference_time + total_data_loading_time
                overall_throughput = total_samples / total_test_time
                inference_throughput = total_samples / total_inference_time

                logging.getLogger(__name__).info("=" * 60)
                logging.getLogger(__name__).info("Inference Speed Performance Report")
                logging.getLogger(__name__).info("=" * 60)
                logging.getLogger(__name__).info(f"Total number of test samples: {total_samples}")
                logging.getLogger(__name__).info(f"Total number of batches: {len(inference_times)}")
                logging.getLogger(__name__).info("")
                logging.getLogger(__name__).info("Inference Time Statistics:")
                logging.getLogger(__name__).info(f"  Total inference time: {total_inference_time:.4f}s")
                logging.getLogger(__name__).info(f"  Average inference time: {avg_inference_time * 1000:.2f}ms")
                logging.getLogger(__name__).info(f"  Median inference time: {median_inference_time * 1000:.2f}ms")
                logging.getLogger(__name__).info(f"  Minimum inference time: {min_inference_time * 1000:.2f}ms")
                logging.getLogger(__name__).info(f"  Maximum inference time: {max_inference_time * 1000:.2f}ms")
                logging.getLogger(__name__).info(f"  Inference time std dev: {std_inference_time * 1000:.2f}ms")
                logging.getLogger(__name__).info("")
                logging.getLogger(__name__).info("Data Loading Time Statistics:")
                logging.getLogger(__name__).info(f"  Total data loading time: {total_data_loading_time:.4f}s")
                logging.getLogger(__name__).info(f"  Average data loading time: {avg_data_loading_time * 1000:.2f}ms")
                logging.getLogger(__name__).info("")
                logging.getLogger(__name__).info("Throughput Statistics:")
                logging.getLogger(__name__).info(f"  Inference throughput only: {inference_throughput:.2f} samples/s")
                logging.getLogger(__name__).info(f"  Overall throughput: {overall_throughput:.2f} samples/s")
                logging.getLogger(__name__).info(
                    f"  Batches processed per second: {len(inference_times) / total_test_time:.2f} batches/s"
                )
                logging.getLogger(__name__).info("")
                logging.getLogger(__name__).info("Performance Ratio:")
                data_loading_ratio = (total_data_loading_time / total_test_time) * 100
                inference_ratio = (total_inference_time / total_test_time) * 100
                logging.getLogger(__name__).info(f"  Data loading time ratio: {data_loading_ratio:.1f}%")
                logging.getLogger(__name__).info(f"  Inference time ratio: {inference_ratio:.1f}%")
                logging.getLogger(__name__).info("=" * 60)

            # --- Calculate Metrics ---
            if metric_suite == "poseidon":
                logging.getLogger(__name__).warning(
                    "Warning: 'poseidon' metric suite requires adaptation for variable nodes per sample PyG structure. Skipping calculation."
                )
                final_metric = float("nan")
                self.config.datarow["relative error (direct)"] = final_metric

            if metric_suite == "drivaernet":
                if self.dataset_config.active_variables is not None:
                    self.metadata.global_mean = [self.metadata.global_mean[i] for i in self.dataset_config.active_variables]
                    self.metadata.global_std = [self.metadata.global_std[i] for i in self.dataset_config.active_variables]
                agg_metrics = compute_drivaernet_metric(
                    gtr_ls=all_batch_targets_denorm, prd_ls=all_batch_preds_denorm, metadata=self.metadata
                )

                logging.getLogger(__name__).info("--- Final Metrics (Drivaernet Suite - Full Dataset) ---")
                logging.getLogger(__name__).info(f"MSE (x10^-2):       {agg_metrics['MSE'] * 100:.4f}")
                logging.getLogger(__name__).info(f"MAE (x10^-1):       {agg_metrics['MAE'] * 10:.4f}")
                logging.getLogger(__name__).info(f"RMSE:               {agg_metrics['RMSE']:.4f}")
                logging.getLogger(__name__).info(f"Max_Error:          {agg_metrics['Max_Error']:.4f}")
                logging.getLogger(__name__).info(f"Rel L2 Error (%):   {agg_metrics['Rel_L2'] * 100:.4f}")
                logging.getLogger(__name__).info(f"Rel L1 Error (%):   {agg_metrics['Rel_L1'] * 100:.4f}")

            elif metric_suite == "general":
                diff = full_preds - full_targets
                mse = torch.mean(diff**2).item()
                mae = torch.mean(torch.abs(diff)).item()
                max_ae = torch.max(torch.abs(diff)).item()

                norm_diff_l2 = torch.linalg.norm(diff.float())
                norm_gtr_l2 = torch.linalg.norm(full_targets.float())
                rel_l2 = (norm_diff_l2 / (norm_gtr_l2 + EPSILON)).item() * 100.0

                norm_diff_l1 = torch.linalg.norm(diff.float(), ord=1)
                norm_gtr_l1 = torch.linalg.norm(full_targets.float(), ord=1)
                rel_l1 = (norm_diff_l1 / (norm_gtr_l1 + EPSILON)).item() * 100.0

                final_metrics_dict = {
                    "MSE": mse,
                    "MAE": mae,
                    "Max AE": max_ae,
                    "Rel L2 Error (%)": rel_l2,
                    "Rel L1 Error (%)": rel_l1,
                }

                self.config.datarow["MSE (x10^-2)"] = final_metrics_dict["MSE"] * 100
                self.config.datarow["MAE (x10^-1)"] = final_metrics_dict["MAE"] * 10
                self.config.datarow["Max AE"] = final_metrics_dict["Max AE"]
                self.config.datarow["Rel L2 Error (%)"] = final_metrics_dict["Rel L2 Error (%)"]
                self.config.datarow["Rel L1 Error (%)"] = final_metrics_dict["Rel L1 Error (%)"]

                logging.getLogger(__name__).info("--- Final Metrics (General Suite - Full Dataset) ---")
                logging.getLogger(__name__).info(f"MSE (x10^-2):       {final_metrics_dict['MSE'] * 100:.4f}")
                logging.getLogger(__name__).info(f"MAE (x10^-1):       {final_metrics_dict['MAE'] * 10:.4f}")
                logging.getLogger(__name__).info(f"Max AE:             {final_metrics_dict['Max AE']:.4f}")
                logging.getLogger(__name__).info(f"Rel L2 Error (%):   {final_metrics_dict['Rel L2 Error (%)']:.4f}")
                logging.getLogger(__name__).info(f"Rel L1 Error (%):   {final_metrics_dict['Rel L1 Error (%)']:.4f}")

                results_array = torch.cat([full_targets.view(-1, 1), full_preds.view(-1, 1)], dim=1).cpu().numpy()
                csv_save_path = self.path_config.result_path.replace(".png", "_predictions.csv")
                np.savetxt(csv_save_path, results_array, delimiter=",", header="True_Drag,Predicted_Drag", comments="")
                logging.getLogger(__name__).info(f"Saved all test predictions to {csv_save_path}")

            logging.getLogger(__name__).info("Skipping 3D plotting for scalar drag prediction")
