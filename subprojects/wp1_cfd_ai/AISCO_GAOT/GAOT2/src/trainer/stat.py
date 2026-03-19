import logging
import math
import os
import random
import statistics
import time
from dataclasses import asdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f
import torch_geometric as pyg
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm

from src.dataloader.collate_functions import create_collate_function
from src.dataloader.pyg_datasets import EnrichedData, VTKMeshDataset
from src.dataloader.pyg_transforms import NormalizeFeatures, RescalePosition, RescalePositionNew
from src.model.gaot_3d import GAOT3D, masked_multi_head_loss
from src.model.layers.magno import get_neighbor_strategy, parse_neighbor_strategy
from src.utils.scale import rescale, rescale_new

from .base import TrainerBase
from .utils.metric import compute_drivaernet_metric

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
    magno = model_config["args"]["magno"]
    if isinstance(magno, dict):
        return magno.get(key, default)
    else:
        return getattr(magno, key, default)


def _magno_set(model_config, key, value):
    """Helper to set magno config keys whether magno is a dict or dataclass."""
    magno = model_config["args"]["magno"]
    if isinstance(magno, dict):
        magno[key] = value
    else:
        setattr(magno, key, value)


def _build_gaot_config(model_config, input_size: int, output_size: int) -> dict[str, Any]:
    """
    Build the flat config dict that GAOT3D expects from the structured
    ModelConfig dataclass.  Runtime-detected input_size / output_size
    override whatever is stored in the YAML.
    """
    args = model_config["args"]

    # args may be a dataclass (ModelArgsConfig) or already a dict
    cfg: dict[str, Any]
    if hasattr(args, "__dataclass_fields__"):
        cfg = asdict(args)
    elif isinstance(args, dict):
        cfg = dict(args)
    else:
        # OmegaConf DictConfig or similar
        cfg_container = OmegaConf.to_container(args, resolve=True)
        cfg = {str(k): v for k, v in cfg_container.items()} if isinstance(cfg_container, dict) else {}

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
        if self.dataset_config["training_strategy"] == "neural_field":
            if self.dataset_config["update_pt_files_with_edges"]:
                raise ValueError("neural_field training strategy requires update_pt_files_with_edges=False")
            if _magno_get(self.model_config, "precompute_edges", True):
                logging.getLogger(__name__).warning(
                    "WARNING: neural_field training strategy requires precompute_edges=False, setting it to False automatically."
                )
                _magno_set(self.model_config, "precompute_edges", False)

        # --- Uncertainty weighting (Kendall et al., 2018) ---
        # Learns log(sigma) per task so the combined loss self-scales.
        # Enabled via dataset.uncertainty_weighting: true in config.
        self._uncertainty_weighting = self.dataset_config["uncertainty_weighting"]
        if self._uncertainty_weighting:
            self.log_sigma_field = torch.nn.Parameter(torch.tensor([math.log(75.0)], device=self.device))
            self.log_sigma_scalar = torch.nn.Parameter(torch.tensor([math.log(0.5)], device=self.device))
            # Register as an extra param group in the existing AdamW optimizer
            # (no weight-decay on task-noise parameters)
            self.optimizer.optimizer.add_param_group(
                {
                    "params": [self.log_sigma_field, self.log_sigma_scalar],
                    "lr": self.optimizer_config["args"]["lr"],
                    "weight_decay": 0.0,
                }
            )
            logging.getLogger(__name__).info(
                "Uncertainty weighting enabled: log_sigma_field and log_sigma_scalar will be learned."
            )

    def _calculate_or_load_stats(self, dataset_config, order_file_path, data_root):
        """Calculates or loads normalization statistics.

        In DDP mode, only rank 0 computes and saves stats; other ranks
        wait at a barrier and then load from the saved file.
        """
        stats_file = os.path.join(data_root, dataset_config["name"] + "_norm_stats.pt")
        rank = self.setup_config["rank"]
        distributed = self.setup_config["distributed"]

        need_compute = not os.path.exists(stats_file) or dataset_config["force_recompute_stats"]

        if need_compute:
            if rank == 0:
                self._compute_and_save_stats(dataset_config, order_file_path, stats_file)
            if distributed:
                torch.distributed.barrier()
            # All ranks load from saved file
            self._load_stats(stats_file)
        else:
            self._load_stats(stats_file)

    def _load_stats(self, stats_file):
        """Load pre-computed normalization stats from disk."""
        logging.getLogger(__name__).info(f"Loading normalization stats from {stats_file}")
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

        if self.u_mean is None or self.u_std is None:
            raise RuntimeError("Normalization mean/std could not be calculated or loaded.")
        self.u_mean = self.u_mean.to(self.dtype)
        self.u_std = self.u_std.to(self.dtype)

    def _compute_and_save_stats(self, dataset_config, order_file_path, stats_file):  # noqa: PLR0912, PLR0915
        """Compute normalization stats from training data and save to disk (rank 0 only)."""
        logging.getLogger(__name__).info("Calculating normalization statistics from training set...")
        temp_train_dataset = VTKMeshDataset(
            root=dataset_config["base_path"],
            order_file=order_file_path,
            dataset_config=dataset_config,
            split="train",
            transform=RescalePosition(),
        )

        temp_loader = PyGDataLoader(temp_train_dataset, batch_size=dataset_config["batch_size"], shuffle=False, num_workers=4)

        all_x = []
        all_c = []
        all_force = []
        all_cd = []
        has_c_field = False

        for batch in tqdm(temp_loader, desc="Calculating Stats"):
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
        per_head_stats = {}
        if all_force:
            force_tensor = torch.cat(all_force, dim=0)
            per_head_stats["force"] = {
                "mean": torch.mean(force_tensor, dtype=self.dtype),
                "std": torch.std(force_tensor).to(self.dtype),
            }
            logging.getLogger(__name__).info(
                f"Force stats - Mean: {per_head_stats['force']['mean']}, Std: {per_head_stats['force']['std']}"
            )
        if all_cd:
            cd_tensor = torch.cat(all_cd, dim=0)
            per_head_stats["cd"] = {
                "mean": torch.mean(cd_tensor, dtype=self.dtype),
                "std": torch.std(cd_tensor).to(self.dtype),
            }
            logging.getLogger(__name__).info(
                f"Cd stats - Mean: {per_head_stats['cd']['mean']}, Std: {per_head_stats['cd']['std']}"
            )

        # Build unified u_mean/u_std from whatever data is available
        if all_x:
            combined = all_x
        elif all_force or all_cd:
            combined = all_force + all_cd
        else:
            raise ValueError("No data found in training set to calculate statistics.")

        full_x_tensor = torch.cat(combined, dim=0)
        u_mean = torch.mean(full_x_tensor, dim=0, dtype=self.dtype)
        u_std = torch.std(full_x_tensor, dim=0).to(self.dtype)

        c_mean = None
        c_std = None
        if has_c_field and all_c:
            full_c_tensor = torch.cat(all_c, dim=0)
            c_mean = torch.mean(full_c_tensor, dim=0, dtype=self.dtype)
            c_std = torch.std(full_c_tensor, dim=0).to(self.dtype)
            logging.getLogger(__name__).info(f"Calculated c - Mean: {c_mean}, Std: {c_std}")

        logging.getLogger(__name__).info(f"Calculated x - Mean: {u_mean}, Std: {u_std}")
        logging.getLogger(__name__).info(f"Saving normalization stats to {stats_file}")
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)

        stats_to_save: dict[str, Any] = {"mean": u_mean, "std": u_std}
        if c_mean is not None:
            assert c_std is not None
            stats_to_save["c_mean"] = c_mean
            stats_to_save["c_std"] = c_std
        if per_head_stats:
            stats_to_save["per_head_stats"] = per_head_stats
        torch.save(stats_to_save, stats_file)

    def _update_pt_files_with_edges(self, dataset_config, order_file_path, magno_cfg: dict):  # noqa: PLR0912, PLR0915
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
            except Exception:
                logging.getLogger(__name__).exception(f"Error processing file {fpath}")
                if os.path.exists(fpath_tmp):
                    os.remove(fpath_tmp)
        logging.getLogger(__name__).info("Rank 0: Finished checking/updating '.pt' files.")

    def init_dataset(self, dataset_config):  # noqa: PLR0912, PLR0915
        logging.getLogger(__name__).info("Initializing PyG dataset ...")
        self.latent_token_size = self.model_config["args"]["latent_tokens"]
        data_root = dataset_config["base_path"]

        # Allow custom order file name via config (e.g. "order_force_only.txt")
        order_file_name = dataset_config["order_file_name"]
        if order_file_name:
            order_file_path = os.path.join(data_root, order_file_name)
        else:
            order_file_path = os.path.join(data_root, "order_" + dataset_config["processed_folder"] + ".txt")

        # Store active_heads for filtering (e.g. ["force"] or ["force","cd"])
        self.active_heads = dataset_config["active_heads"]
        if self.active_heads:
            logging.getLogger(__name__).info(f"Active heads (training only on): {self.active_heads}")

        processed_data_path = os.path.join(data_root, dataset_config["processed_folder"])

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
        if dataset_config["use_rescale_new"]:
            self.latent_tokens = rescale_new(latent_queries, (-1, 1), phy_domain)
        else:
            self.latent_tokens = rescale(latent_queries, (-1, 1))

        logging.getLogger(__name__).info("Rank" + str(self.setup_config["rank"]) + ": Dataset initialization finished.")

        # --- Pre-computation /Update Step ---
        if dataset_config["update_pt_files_with_edges"]:
            magno_cfg = self.model_config.args.magno
            if self.setup_config["rank"] == 0:
                self._update_pt_files_with_edges(dataset_config, order_file_path, magno_cfg)
            if self.setup_config["distributed"]:
                logging.getLogger(__name__).info(
                    "Rank" + str(self.setup_config.rank) + ": Waiting for edge pre-computation barrier..."
                )
                torch.distributed.barrier()
                logging.getLogger(__name__).info("Rank" + str(self.setup_config.rank) + ": Barrier passed.")

            _magno_set(self.model_config, "precompute_edges", True)
            logging.getLogger(__name__).info(f"Rank {self.setup_config.rank}: Set model config to use precomputed edges.")
        # --- end Pre-computation /Update Step ---

        # --- Calculate or Load Normalization Stats ---
        self._calculate_or_load_stats(dataset_config, order_file_path, data_root)

        # --- Define Transforms ---
        if dataset_config["use_rescale_new"]:
            rescale_transform = RescalePositionNew(lims=(-1.0, 1.0), phy_domain=phy_domain)
        else:
            rescale_transform = RescalePosition(lims=(-1.0, 1.0))

        # Apply active_variables selection to normalization stats if specified
        if dataset_config["active_variables"] is not None:
            mean_for_norm = self.u_mean[dataset_config["active_variables"]]
            std_for_norm = self.u_std[dataset_config["active_variables"]]
        else:
            mean_for_norm = self.u_mean
            std_for_norm = self.u_std

        # Prepare c normalization stats if they exist
        c_mean_for_norm = None
        c_std_for_norm = None
        if self.c_mean is not None and self.c_std is not None:
            c_mean_for_norm = self.c_mean
            c_std_for_norm = self.c_std

        # If active_heads is set, filter per_head_stats to only those heads
        active_per_head_stats = self.per_head_stats if hasattr(self, "per_head_stats") else None
        if self.active_heads and active_per_head_stats:
            active_per_head_stats = {k: v for k, v in active_per_head_stats.items() if k in self.active_heads}

        normalize_transform = NormalizeFeatures(
            mean=mean_for_norm,
            std=std_for_norm,
            c_mean=c_mean_for_norm,
            c_std=c_std_for_norm,
            per_head_stats=active_per_head_stats,
        )
        composed_transform = Compose([rescale_transform, normalize_transform])

        # --- Create collate function for online graph building in DataLoader ---
        magno_cfg = self.model_config["args"]["magno"]
        collate_fn = create_collate_function(
            coord_dim=magno_cfg["gno_coord_dim"] if isinstance(magno_cfg, dict) else magno_cfg.gno_coord_dim,
            magno_radius=magno_cfg["gno_radius"] if isinstance(magno_cfg, dict) else magno_cfg.gno_radius,
            magno_scales=magno_cfg["scales"] if isinstance(magno_cfg, dict) else magno_cfg.scales,
            latent_tokens=self.latent_tokens,
            neighbor_search_method=magno_cfg["neighbor_strategy"] if isinstance(magno_cfg, dict) else magno_cfg.neighbor_strategy,
            k_neighbors=magno_cfg["k_neighbors"] if isinstance(magno_cfg, dict) else magno_cfg.k_neighbors,
            asynchronous_graph_building=magno_cfg.get("asynchronous_graph_building", False)
            if isinstance(magno_cfg, dict)
            else magno_cfg["asynchronous_graph_building"],
        )
        # Store collate_fn so train_step can rebuild edges for subsampled batches
        self.collate_fn = collate_fn
        # Ensure the model expects precomputed edges coming from collate
        async_graph = (
            magno_cfg.get("asynchronous_graph_building", False)
            if isinstance(magno_cfg, dict)
            else magno_cfg["asynchronous_graph_building"]
        )
        if async_graph:
            _magno_set(self.model_config, "precompute_edges", True)
            logging.getLogger(__name__).info("Enabled precompute_edges for MAGNO to consume collate-built graphs.")

        if self.setup_config["train"]:
            train_ds = VTKMeshDataset(
                root=data_root,
                order_file=order_file_path,
                dataset_config=dataset_config,
                split="train",
                transform=composed_transform,
            )
            val_ds = VTKMeshDataset(
                root=data_root,
                order_file=order_file_path,
                dataset_config=dataset_config,
                split="val",
                transform=composed_transform,
            )
        test_ds = VTKMeshDataset(
            root=data_root, order_file=order_file_path, dataset_config=dataset_config, split="test", transform=composed_transform
        )

        encoder_feature_attr = list(_magno_get(self.model_config, "encoder_feature_attr", "x"))
        print(test_ds[0], encoder_feature_attr)
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
        if self.setup_config["train"]:
            ## --- Create DistributedSampler for training ---
            train_sampler = None
            if self.setup_config["distributed"]:
                train_sampler = DistributedSampler(
                    train_ds,
                    num_replicas=self.setup_config["world_size"],
                    rank=self.setup_config["rank"],
                    shuffle=dataset_config.shuffle,  # Sampler handles shuffling
                    drop_last=True,  # Often good practice for DDP
                )
                logging.getLogger(__name__).info(f"Rank {self.setup_config.rank}: Created DistributedSampler for training.")

            self.train_loader = DataLoader(
                train_ds,
                batch_size=dataset_config["batch_size"],
                shuffle=False if train_sampler is not None else dataset_config["shuffle"],
                num_workers=dataset_config["num_workers"],
                sampler=train_sampler,
                pin_memory=False,
                drop_last=train_sampler is not None,
                collate_fn=collate_fn,
            )

            val_sampler = None
            if self.setup_config["distributed"]:
                val_sampler = DistributedSampler(
                    val_ds,
                    num_replicas=self.setup_config["world_size"],
                    rank=self.setup_config["rank"],
                    shuffle=False,
                    drop_last=False,
                )
            self.val_loader = DataLoader(
                val_ds,
                batch_size=dataset_config["batch_size"],
                shuffle=False,
                num_workers=dataset_config["num_workers"],
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )

        test_sampler = None
        self.test_loader = DataLoader(
            test_ds,
            batch_size=dataset_config["batch_size"],
            shuffle=False,
            num_workers=dataset_config["num_workers"],
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

        if self.setup_config["distributed"]:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.setup_config["local_rank"]],
                output_device=self.setup_config["local_rank"],
                find_unused_parameters=True,
            )

    def _sample_nodes_neural_field(self, batch: "pyg.data.Batch", num_input_nodes: int, num_query_nodes: int):  # noqa: PLR0912
        """
        Sample nodes for neural field training strategy.
        """
        sampled_data_list = []
        query_pos_list = []
        query_batch_idx_list = []
        field_target_list = []

        use_same_sampling = num_input_nodes == num_query_nodes

        # Check if we have per-vertex field data (data.x) to sample targets from
        has_field_data = hasattr(batch, "x") and batch.x is not None

        for i in range(batch.num_graphs):
            start_idx = batch.ptr[i]
            end_idx = batch.ptr[i + 1]
            num_full_nodes_i = end_idx - start_idx

            full_pos_i = batch.pos[start_idx:end_idx]

            full_c_i = None
            if hasattr(batch, "c") and batch.c is not None:
                full_c_i = batch.c[start_idx:end_idx]

            full_x_i = None
            if has_field_data:
                full_x_i = batch.x[start_idx:end_idx]

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

            data_i = Data(pos=sampled_pos_i)
            if full_c_i is not None:
                data_i.c = full_c_i[input_perm]
            if full_x_i is not None:
                data_i.x = full_x_i[input_perm]

            # Copy scalar/sample-level attributes to sampled data
            for attr in ["force", "cd", "filename", "num_latent_nodes", "v_inf", "geom_extent", "geom_aspect", "conditioning"]:
                if hasattr(batch, attr) and getattr(batch, attr) is not None:
                    attr_value = getattr(batch, attr)
                    if isinstance(attr_value, list) or (attr_value.dim() > 0 and len(attr_value) == batch.num_graphs):
                        selected = attr_value[i]
                        if attr == "conditioning" and isinstance(selected, torch.Tensor) and selected.dim() == 1:
                            selected = selected.unsqueeze(0)
                        setattr(data_i, attr, selected)

            sampled_data_list.append(data_i)

            query_pos_list.append(full_pos_i[query_perm])
            query_batch_idx_list.append(torch.full((n_query_sample,), i, dtype=torch.long))
            if full_x_i is not None:
                field_target_list.append(full_x_i[query_perm])

        query_pos = torch.cat(query_pos_list, dim=0)
        query_batch_idx = torch.cat(query_batch_idx_list, dim=0)
        field_target = torch.cat(field_target_list, dim=0) if field_target_list else None

        return sampled_data_list, query_pos, query_batch_idx, field_target

    def _pick_viz_samples(self):
        """Randomly select one sample from train and one from val for per-epoch visualisation."""
        # Use a fresh RNG seeded from OS entropy so the global training seed doesn't fix these picks
        rng = random.Random(int.from_bytes(os.urandom(4), "big"))
        train_idx = rng.randint(0, len(self.train_loader.dataset) - 1)
        val_idx = rng.randint(0, len(self.val_loader.dataset) - 1)
        train_sample = self.train_loader.dataset[train_idx]
        val_sample = self.val_loader.dataset[val_idx]
        logging.getLogger(__name__).info(
            f"Rank {self.setup_config.rank}: Viz samples — "
            f"train[{train_idx}] ({getattr(train_sample, 'filename', 'N/A')}), "
            f"val[{val_idx}] ({getattr(val_sample, 'filename', 'N/A')})"
        )
        return train_sample, val_sample

    def _run_viz_for_sample(self, sample, split_label: str, epoch: int):
        """Run inference on a single sample and log field + scalar images to W&B."""
        loader = PyGDataLoader([sample], batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(self.device)
        latent_tokens_dev = self.latent_tokens.to(self.device)
        with torch.no_grad():
            pred = self.model(batch=batch, tokens_pos=latent_tokens_dev)

        coords = batch.pos.cpu().numpy()
        fname = getattr(sample, "filename", split_label)

        # --- Field visualisation ---
        if isinstance(pred, dict) and "field" in pred:
            field_pred = pred["field"].cpu().numpy().squeeze()
            field_gt = batch.x.cpu().numpy().squeeze() if batch.x is not None else None

            if field_gt is not None:
                u_mean = self.u_mean.cpu().numpy().item() if self.u_mean.numel() == 1 else 0.0
                u_std = self.u_std.cpu().numpy().item() if self.u_std.numel() == 1 else 1.0
                field_pred_dn = field_pred * u_std + u_mean
                field_gt_dn = field_gt * u_std + u_mean
                field_diff = np.abs(field_gt_dn - field_pred_dn)

                common_min = min(np.min(field_gt_dn), np.min(field_pred_dn))
                common_max = max(np.max(field_gt_dn), np.max(field_pred_dn))

                fig = plt.figure(figsize=(20, 6))
                for panel_idx, (data, title, cmap_name, vmin, vmax) in enumerate(
                    [
                        (field_gt_dn, "Ground Truth", "jet", common_min, common_max),
                        (field_pred_dn, "Prediction", "jet", common_min, common_max),
                        (field_diff, "Absolute Difference", "Reds", None, None),
                    ]
                ):
                    ax = fig.add_subplot(1, 3, panel_idx + 1, projection="3d")
                    sc = ax.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        coords[:, 2],
                        c=data,
                        cmap=cmap_name,
                        s=2,
                        vmin=vmin,
                        vmax=vmax,
                        depthshade=False,
                    )
                    ax.set_title(title)
                    fig.colorbar(sc, ax=ax, shrink=0.5)
                    ax.view_init(elev=20, azim=-120)

                fig.suptitle(f"Pressure Field [{split_label}] {fname} — Epoch {epoch + 1}", fontsize=13)
                plt.tight_layout()
                self.tracker.log_image(
                    f"viz/{split_label}/pressure_field", fig, step=epoch + 1, caption=f"{split_label} {fname} epoch {epoch + 1}"
                )
                plt.close(fig)

        # --- Scalar visualisation ---
        if isinstance(pred, dict):
            scalar_parts = []
            for head in ["force", "cd"]:
                if head in pred and hasattr(batch, head):
                    p = pred[head].cpu().item()
                    gt_val = getattr(batch, head)
                    if gt_val is not None and not torch.isnan(gt_val).all():
                        g = gt_val.cpu().item()
                        if hasattr(self, "per_head_stats") and head in self.per_head_stats:
                            h_mean = self.per_head_stats[head]["mean"].item()
                            h_std = self.per_head_stats[head]["std"].item()
                            p = p * h_std + h_mean
                            g = g * h_std + h_mean
                        scalar_parts.append((head, g, p))

            if scalar_parts:
                fig, axes = plt.subplots(1, len(scalar_parts), figsize=(5 * len(scalar_parts), 4))
                if len(scalar_parts) == 1:
                    axes = [axes]
                for ax, (name, gt_v, pr_v) in zip(axes, scalar_parts, strict=False):
                    ax.bar(["GT", "Pred"], [gt_v, pr_v], color=["steelblue", "coral"])
                    ax.set_title(f"{name}  (GT={gt_v:.4f}, Pred={pr_v:.4f})")
                    ax.set_ylabel(name)
                fig.suptitle(f"Scalars [{split_label}] {fname} — Epoch {epoch + 1}", fontsize=13)
                plt.tight_layout()
                self.tracker.log_image(
                    f"viz/{split_label}/scalar_predictions",
                    fig,
                    step=epoch + 1,
                    caption=f"{split_label} {fname} epoch {epoch + 1}",
                )
                plt.close(fig)

    def visualize_sample(self, epoch: int):
        """Run inference on fixed train+val samples and log images to W&B each epoch."""
        if self.setup_config.rank != 0:
            return
        if not hasattr(self, "_viz_train_sample"):
            self._viz_train_sample, self._viz_val_sample = self._pick_viz_samples()

        self.model.eval()
        for sample, label in [(self._viz_train_sample, "train"), (self._viz_val_sample, "val")]:
            try:
                self._run_viz_for_sample(sample, label, epoch)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Viz failed for {label} sample: {e}")

    def train_step(self, batch: "pyg.data.Batch"):  # noqa: PLR0912, PLR0915
        strategy = self.dataset_config["training_strategy"]
        latent_tokens_dev = self.latent_tokens.to(self.device)

        field_target = None  # per-vertex target from neural_field sampling

        if strategy == "neural_field":
            # --- Neural Field Strategy ---
            num_input_nodes = self.dataset_config["neural_field_input_nodes"]
            num_query_nodes = self.dataset_config["neural_field_query_nodes_train"]

            sampled_data_list, query_pos, query_batch_idx, field_target = self._sample_nodes_neural_field(
                batch, num_input_nodes, num_query_nodes
            )

            # Build edges for the subsampled data via collate, then move to device
            sampled_batch = self.collate_fn(sampled_data_list)
            sampled_batch = sampled_batch.to(self.device)
            query_pos = query_pos.to(self.device)
            query_batch_idx = query_batch_idx.to(self.device)
            if field_target is not None:
                field_target = field_target.to(self.device)

            pred = self.model(
                batch=sampled_batch,
                tokens_pos=latent_tokens_dev,
                query_coord_pos=query_pos,
                query_coord_batch_idx=query_batch_idx,
            )

            # For neural_field, the batch for scalar targets is the sampled_batch
            batch = sampled_batch
        else:
            batch = batch.to(self.device)
            pred = self.model(batch=batch, tokens_pos=latent_tokens_dev)

        # pred is always a dict: {"force": [B,1], "cd": [B,1], optionally "field": [Q, D]}
        if isinstance(pred, dict):
            # --- Scalar head loss ---
            targets = {"force": batch.force, "cd": batch.cd}
            if self.active_heads:
                targets = {k: v for k, v in targets.items() if k in self.active_heads}
            scalar_loss = masked_multi_head_loss(pred, targets)

            # --- Per-head MSE for logging ---
            head_losses = {}
            loss_fn_none = torch.nn.MSELoss(reduction="none")
            for name in targets:
                if name not in pred:
                    continue
                gt = targets[name]
                if gt is None:
                    continue
                gt = gt.to(self.device).view_as(pred[name])
                valid = ~torch.isnan(gt)
                if valid.any():
                    head_losses[name] = loss_fn_none(pred[name][valid], gt[valid]).mean().item()

            # --- Field loss (if model produces field predictions) ---
            if "field" in pred:
                # Field target: use sampled field_target (neural_field) or batch.x
                if field_target is not None:
                    ft = field_target
                elif hasattr(batch, "x") and batch.x is not None:
                    ft = batch.x.to(self.device)
                else:
                    ft = None

                if ft is not None:
                    field_pred = pred["field"]
                    # Mask out NaN entries (samples without field data)
                    valid = ~torch.isnan(ft)
                    if valid.any():
                        field_loss = f.mse_loss(field_pred[valid], ft[valid])
                    else:
                        field_loss = torch.tensor(0.0, device=self.device)

                    head_losses["field"] = field_loss.item()

                    if self._uncertainty_weighting:
                        # Kendall et al. (2018) multi-task uncertainty loss:
                        # L = (exp(-2s_f)/2)*L_f + s_f + (exp(-2s_s)/2)*L_scalar + s_s
                        # where s_x = log(sigma_x)  (numerically stable via exp(-2s))
                        s_f = self.log_sigma_field
                        s_s = self.log_sigma_scalar
                        combined = (
                            0.5 * torch.exp(-2.0 * s_f) * field_loss + s_f + 0.5 * torch.exp(-2.0 * s_s) * scalar_loss + s_s
                        )
                        head_losses["sigma_field"] = torch.exp(s_f).item()
                        head_losses["sigma_scalar"] = torch.exp(s_s).item()
                        return combined.squeeze(), head_losses
                    else:
                        field_weight = self.dataset_config["field_loss_weight"]
                        return scalar_loss + field_weight * field_loss, head_losses

            return scalar_loss, head_losses
        else:
            target = batch.y
            return self.loss_fn(pred, target), {}

    def _add_field_loss(self, batch, pred, loss, head_losses):
        """Compute field MSE and add it to loss; updates head_losses in-place."""
        if "field" not in pred or not hasattr(batch, "x") or batch.x is None:
            return loss
        ft = batch.x.to(self.device)
        valid = ~torch.isnan(ft)
        if valid.any():
            field_loss = f.mse_loss(pred["field"][valid], ft[valid])
            head_losses["field"] = head_losses.get("field", 0.0) + field_loss.item()
            field_weight = self.dataset_config["field_loss_weight"]  # 1.0
            loss = loss + field_weight * field_loss
        return loss

    def validate(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        total_rel_l2 = 0.0
        head_losses: dict[str, float] = {}
        num_batches = len(loader)

        with torch.no_grad():
            for batch_raw in loader:
                batch = batch_raw.to(self.device)
                latent_tokens_dev = self.latent_tokens.to(self.device)
                pred = self.model(batch=batch, tokens_pos=latent_tokens_dev)

                if isinstance(pred, dict):
                    # --- Scalar loss ---
                    targets = {"force": batch.force, "cd": batch.cd}
                    if self.active_heads:
                        targets = {k: v for k, v in targets.items() if k in self.active_heads}

                    loss = masked_multi_head_loss(pred, targets)

                    # --- Per-head scalar MSE ---
                    loss_fn_none = torch.nn.MSELoss(reduction="none")
                    for head_name, gt_raw in targets.items():
                        if head_name not in pred or gt_raw is None:
                            continue
                        gt_head = gt_raw.to(self.device).view_as(pred[head_name])
                        valid = ~torch.isnan(gt_head)
                        if valid.any():
                            head_mse = loss_fn_none(pred[head_name][valid], gt_head[valid]).mean().item()
                            head_losses[head_name] = head_losses.get(head_name, 0.0) + head_mse

                    # For Relative L2, we use the primary head (e.g., 'force')
                    primary_key = self.active_heads[0] if self.active_heads else next(iter(pred.keys()))
                    p = pred[primary_key]
                    t = targets[primary_key]

                    # --- Field loss ---
                    loss = self._add_field_loss(batch, pred, loss, head_losses)
                else:
                    target = batch.y
                    loss = self.loss_fn(pred, target)
                    p, t = pred, target

                total_loss += loss.item()

                # 2. Calculate Relative L2 Error with NaN Masking
                # We must mask NaNs before taking the norm, otherwise the result is NaN
                valid_mask = ~torch.isnan(t)
                if valid_mask.any():
                    p_valid = p[valid_mask]
                    t_valid = t[valid_mask]

                    diff_norm = torch.linalg.norm(p_valid - t_valid)
                    target_norm = torch.linalg.norm(t_valid)

                    # Add small epsilon to prevent division by zero
                    rel_l2 = diff_norm / (target_norm + 1e-10)
                    total_rel_l2 += rel_l2.item()
                else:
                    # If no valid targets in this batch, we don't add to total_rel_l2
                    pass

        # Global reduction across ranks for DDP
        local_metrics = torch.tensor([total_loss, total_rel_l2], dtype=torch.float64, device=self.device)
        world_size = 1
        if self.setup_config["distributed"]:  # False
            torch.distributed.all_reduce(local_metrics, op=torch.distributed.ReduceOp.SUM)
            world_size = torch.distributed.get_world_size()

        denom = num_batches * world_size

        result = {"loss": local_metrics[0].item() / denom, "rel_l2": local_metrics[1].item() / denom}
        for h, v in head_losses.items():
            result[f"head_{h}"] = v / num_batches
        return result

    def test(self):  # noqa: PLR0912, PLR0915
        self.model.eval()
        metric_suite = self.dataset_config["metric_suite"]

        # Determine if model uses multi-head output
        _raw_model = self.model.module if hasattr(self.model, "module") else self.model
        _is_multi_head = hasattr(_raw_model, "decoder") and hasattr(_raw_model.decoder, "head_names")
        all_batch_targets_denorm: dict[str, list[Any]] | list[Any]
        all_batch_preds_denorm: dict[str, list[Any]] | list[Any]
        if _is_multi_head:
            all_batch_targets_denorm = {"force": [], "cd": []}
            all_batch_preds_denorm = {"force": [], "cd": []}
        else:
            all_batch_targets_denorm = []
            all_batch_preds_denorm = []
        _plot_coords, plot_gtr, plot_prd = None, None, None

        # Inference speed measurement variables
        inference_times = []
        data_loading_times = []
        total_samples = 0

        logging.getLogger(__name__).info(f"Starting testing with metric suite: '{metric_suite}'")

        with torch.no_grad():
            for i, batch_raw in enumerate(self.test_loader):
                data_start_time = time.time()
                batch = batch_raw.to(self.device)
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
                    # Collect per-head predictions — use force if available, else cd
                    # For now, concatenate all heads for metric reporting
                    pred_norm.get("force")
                    pred_norm.get("cd")

                    # Denormalize multi-head predictions using per-head stats.
                    # Both predictions AND targets are in normalised space
                    # (NormalizeFeatures transform applied at data loading).
                    pred_denorm = {}
                    target_denorm_dict = {}
                    for head_name, head_pred in pred_norm.items():
                        # Skip field output — handled separately below
                        if head_name == "field":
                            continue
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

                    if i == 0 and self.setup_config["rank"] == 0:
                        parts = []
                        if "force" in pred_denorm:
                            parts.append(f"Force={pred_denorm['force'][0].item():.4f}")
                        if "cd" in pred_denorm:
                            parts.append(f"Cd={pred_denorm['cd'][0].item():.4f}")
                        logging.getLogger(__name__).info(f"First test sample predictions (denorm): {', '.join(parts)}")

                    for head_name, head_pred_dn in pred_denorm.items():
                        head_target_dn = target_denorm_dict[head_name]
                        valid = ~torch.isnan(head_target_dn)
                        if valid.any():
                            all_batch_targets_denorm[head_name].append(head_target_dn[valid].cpu())
                            all_batch_preds_denorm[head_name].append(head_pred_dn[valid].detach().cpu())
                else:
                    target_norm = batch.y

                    u_std_dev = self.u_std[self.dataset_config["active_variables"]].to(self.device)
                    u_mean_dev = self.u_mean[self.dataset_config["active_variables"]].to(self.device)
                    pred_de_norm = pred_norm * u_std_dev + u_mean_dev
                    target_de_norm = target_norm * u_std_dev + u_mean_dev

                    assert isinstance(all_batch_targets_denorm, list)
                    assert isinstance(all_batch_preds_denorm, list)
                    all_batch_targets_denorm.append(target_de_norm.detach().cpu())
                    all_batch_preds_denorm.append(pred_de_norm.detach().cpu())

                    if i == 0 and self.setup_config["rank"] == 0:
                        plot_gtr = np.array([target_de_norm[0].item()])
                        plot_prd = np.array([pred_de_norm[0].item()])

                        logging.getLogger(__name__).info(
                            f"Extracted first test sample: True Drag = {plot_gtr[0]:.4f}, Predicted Drag = {plot_prd[0]:.4f}"
                        )

                if (i + 1) % 10 == 0 and self.setup_config["rank"] == 0:
                    current_avg_inference_time = statistics.mean(inference_times[-10:])
                    throughput = 1.0 / current_avg_inference_time
                    logging.getLogger(__name__).info(
                        f"Processed {i + 1} batches, "
                        f"Current average inference time: {current_avg_inference_time * 1000:.2f}ms, "
                        f"Inference throughput: {throughput:.2f} samples/s"
                    )

        if self.setup_config["rank"] == 0:
            if isinstance(all_batch_preds_denorm, dict):
                assert isinstance(all_batch_targets_denorm, dict)
                for head_name in all_batch_preds_denorm:
                    if not all_batch_preds_denorm[head_name]:
                        continue
                    head_preds = torch.cat(all_batch_preds_denorm[head_name], dim=0)
                    head_targets = torch.cat(all_batch_targets_denorm[head_name], dim=0)
                    diff = head_preds - head_targets
                    mse = torch.mean(diff**2).item()
                    mae = torch.mean(torch.abs(diff)).item()

                    # Calculate MRE for better W&B tracking
                    rel_errors = torch.abs(diff) / (torch.abs(head_targets) + EPSILON)
                    mre_percent = torch.mean(rel_errors).item() * 100.0

                    logging.getLogger(__name__).info(
                        f"--- Head '{head_name}': MSE={mse:.6f}, MAE={mae:.6f}, n_samples={head_preds.shape[0]} ---"
                    )

                    # Log Multi-Head metrics to tracker
                    self.tracker.log_metrics(
                        {f"test/{head_name}_mse": mse, f"test/{head_name}_mae": mae, f"test/{head_name}_mre_percent": mre_percent}
                    )

                    results_array = torch.cat([head_targets.view(-1, 1), head_preds.view(-1, 1)], dim=1).cpu().numpy()
                    csv_path = self.path_config["result_path"].replace(".png", f"_{head_name}_predictions.csv")
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
                assert isinstance(all_batch_preds_denorm, list)
                assert isinstance(all_batch_targets_denorm, list)
                full_preds = torch.cat(all_batch_preds_denorm, dim=0)
                full_targets = torch.cat(all_batch_targets_denorm, dim=0)

                diff = full_preds - full_targets
                mse = torch.mean(diff**2).item()
                mre_percent = (torch.abs(diff) / (torch.abs(full_targets) + EPSILON)).mean().item() * 100.0

                # Log Single-Head metrics to tracker
                self.tracker.log_metrics({"test/mse": mse, "test/mre_percent": mre_percent})

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
                data_loading_ratio = (total_data_loading_time / total_test_time) * 100
                inference_ratio = (total_inference_time / total_test_time) * 100

                # Log Performance metrics to tracker
                self.tracker.log_metrics(
                    {
                        "perf/inference_throughput_sps": inference_throughput,
                        "perf/overall_throughput_sps": overall_throughput,
                        "perf/avg_latency_ms": avg_inference_time * 1000,
                        "perf/data_loading_ratio_percent": data_loading_ratio,
                    }
                )

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
                if self.dataset_config["active_variables"] is not None:
                    self.metadata.global_mean = [self.metadata.global_mean[i] for i in self.dataset_config["active_variables"]]
                    self.metadata.global_std = [self.metadata.global_std[i] for i in self.dataset_config["active_variables"]]
                assert isinstance(all_batch_targets_denorm, list), "drivaernet metric requires non-multi-head (list) targets"
                assert isinstance(all_batch_preds_denorm, list), "drivaernet metric requires non-multi-head (list) preds"
                agg_metrics = compute_drivaernet_metric(
                    gtr_ls=all_batch_targets_denorm,
                    prd_ls=all_batch_preds_denorm,
                    metadata=self.metadata,
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
                csv_save_path = self.path_config["result_path"].replace(".png", "_predictions.csv")
                np.savetxt(csv_save_path, results_array, delimiter=",", header="True_Drag,Predicted_Drag", comments="")
                logging.getLogger(__name__).info(f"Saved all test predictions to {csv_save_path}")

            logging.getLogger(__name__).info("Skipping 3D plotting for scalar drag prediction")
