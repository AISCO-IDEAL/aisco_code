"""DataHandler is called with config and trainer context, then prepares datasets and loaders."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data
from torch_geometric.transforms import Compose

from src.dataloader.collate_functions import create_collate_function
from src.dataloader.pyg_datasets import VTKMeshDataset
from src.dataloader.pyg_transforms import NormalizeFeatures, RescalePosition, RescalePositionNew
from src.utils.scale import rescale, rescale_new

from .edge_handler import EdgeHandler
from .statistics_handler import StatisticsHandler


class DataHandler:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def prepare_data(self, trainer) -> None:
        logging.getLogger(__name__).info("Initializing PyG dataset ...")
        trainer.latent_token_size = trainer.model_config["args"]["latent_tokens"]
        self._prepare_training_strategy(trainer)
        self._prepare_order_and_heads(trainer)
        self._prepare_latent_tokens(trainer)
        self._precompute_edges_when_requested(trainer)
        StatisticsHandler(self.config).calculate_or_load_stats(trainer, trainer.order_file_path, trainer.data_root)
        self._prepare_viz_settings(trainer)
        composed_transform = self._build_transform(trainer)
        collate_fn = self._build_collate(trainer)
        test_ds = self._build_datasets_and_loaders(trainer, composed_transform, collate_fn)
        self._detect_channel_counts(trainer, test_ds)

    def sample_nodes_neural_field(self, trainer, batch, num_input_nodes: int, num_query_nodes: int):
        sampled_data_list = []
        query_pos_list = []
        query_batch_idx_list = []
        field_target_list = []
        use_same_sampling = num_input_nodes == num_query_nodes
        has_field_data = hasattr(batch, "x") and batch.x is not None
        for index in range(batch.num_graphs):
            start_idx = batch.ptr[index]
            end_idx = batch.ptr[index + 1]
            num_full_nodes_i = end_idx - start_idx
            full_pos_i = batch.pos[start_idx:end_idx]
            full_c_i = batch.c[start_idx:end_idx] if hasattr(batch, "c") and batch.c is not None else None
            full_x_i = batch.x[start_idx:end_idx] if has_field_data else None
            n_input_sample = min(num_input_nodes, num_full_nodes_i)
            if n_input_sample <= 0:
                continue
            sample_seed = int(trainer.setup_config["seed"]) + int(getattr(trainer, "_neural_field_sample_calls", 0))
            trainer._neural_field_sample_calls = getattr(trainer, "_neural_field_sample_calls", 0) + 1
            sample_generator = torch.Generator(device=full_pos_i.device)
            sample_generator.manual_seed(sample_seed)
            input_perm = torch.randperm(num_full_nodes_i, device=full_pos_i.device, generator=sample_generator)[:n_input_sample]
            if use_same_sampling:
                query_perm = input_perm
                n_query_sample = n_input_sample
            else:
                n_query_sample = min(num_query_nodes, num_full_nodes_i)
                query_perm = torch.randperm(num_full_nodes_i, device=full_pos_i.device, generator=sample_generator)[
                    :n_query_sample
                ]
            data_i = Data(pos=full_pos_i[input_perm])
            if full_c_i is not None:
                data_i.c = full_c_i[input_perm]
            if full_x_i is not None:
                data_i.x = full_x_i[input_perm]
            for attr in ["force", "cd", "filename", "num_latent_nodes"]:
                if hasattr(batch, attr) and getattr(batch, attr) is not None:
                    attr_value = getattr(batch, attr)
                    if isinstance(attr_value, list) or (attr_value.dim() > 0 and len(attr_value) == batch.num_graphs):
                        setattr(data_i, attr, attr_value[index])
            sampled_data_list.append(data_i)
            query_pos_list.append(full_pos_i[query_perm])
            query_batch_idx_list.append(torch.full((n_query_sample,), index, dtype=torch.long))
            if full_x_i is not None:
                field_target_list.append(full_x_i[query_perm])
        field_target = torch.cat(field_target_list, dim=0) if field_target_list else None
        return sampled_data_list, torch.cat(query_pos_list, dim=0), torch.cat(query_batch_idx_list, dim=0), field_target

    def _prepare_training_strategy(self, trainer) -> None:
        if trainer.dataset_config["training_strategy"] != "neural_field":
            return
        if trainer.dataset_config["update_pt_files_with_edges"]:
            raise ValueError("neural_field training strategy requires update_pt_files_with_edges=False")
        if trainer.model_config["args"]["magno"].get("precompute_edges", True):
            logging.getLogger(__name__).warning(
                "WARNING: neural_field training strategy requires precompute_edges=False, setting it to False automatically."
            )
            trainer.model_config["args"]["magno"]["precompute_edges"] = False

    def _prepare_order_and_heads(self, trainer) -> None:
        trainer.data_root = trainer.dataset_config["base_path"]
        order_file_name = trainer.dataset_config["order_file_name"]
        trainer.order_file_path = (
            os.path.join(trainer.data_root, order_file_name)
            if order_file_name
            else os.path.join(trainer.data_root, "order_" + trainer.dataset_config["processed_folder"] + ".txt")
        )
        trainer.active_heads = trainer.dataset_config["active_heads"]
        if trainer.active_heads:
            logging.getLogger(__name__).info(f"Active heads (training only on): {trainer.active_heads}")
        processed_data_path = os.path.join(trainer.data_root, trainer.dataset_config["processed_folder"])
        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"Processed data directory does not exist: {processed_data_path}")
        if not os.path.exists(trainer.order_file_path):
            raise FileNotFoundError(f"Order file does not exist: {trainer.order_file_path}")

    def _prepare_latent_tokens(self, trainer) -> None:
        phy_domain = trainer.metadata.domain_x
        x_min, y_min, z_min = phy_domain[0]
        x_max, y_max, z_max = phy_domain[1]
        meshgrid = torch.meshgrid(
            torch.linspace(x_min, x_max, trainer.latent_token_size[0]),
            torch.linspace(y_min, y_max, trainer.latent_token_size[1]),
            torch.linspace(z_min, z_max, trainer.latent_token_size[2]),
            indexing="ij",
        )
        latent_queries = torch.stack(meshgrid, dim=-1).reshape(-1, 3)
        trainer.latent_tokens = (
            rescale_new(latent_queries, (-1, 1), phy_domain)
            if trainer.dataset_config["use_rescale_new"]
            else rescale(latent_queries, (-1, 1))
        )

    def _precompute_edges_when_requested(self, trainer) -> None:
        if not trainer.dataset_config["update_pt_files_with_edges"]:
            return
        EdgeHandler(self.config).update_pt_files_with_edges(
            trainer, trainer.order_file_path, trainer.model_config["args"]["magno"]
        )
        if trainer.setup_config["distributed"]:
            torch.distributed.barrier()
        trainer.model_config["args"]["magno"]["precompute_edges"] = True

    def _build_transform(self, trainer):
        phy_domain = trainer.metadata.domain_x
        rescale_transform = (
            RescalePositionNew(lims=(-1.0, 1.0), phy_domain=phy_domain)
            if trainer.dataset_config["use_rescale_new"]
            else RescalePosition(lims=(-1.0, 1.0))
        )
        if trainer.dataset_config["active_variables"] is not None:
            mean_for_norm = trainer.u_mean[trainer.dataset_config["active_variables"]]
            std_for_norm = trainer.u_std[trainer.dataset_config["active_variables"]]
        else:
            mean_for_norm = trainer.u_mean
            std_for_norm = trainer.u_std
        active_per_head_stats = trainer.per_head_stats if hasattr(trainer, "per_head_stats") else None
        if trainer.active_heads and active_per_head_stats:
            active_per_head_stats = {k: v for k, v in active_per_head_stats.items() if k in trainer.active_heads}
        normalize_transform = NormalizeFeatures(
            mean=mean_for_norm,
            std=std_for_norm,
            c_mean=None,
            c_std=None,
            per_head_stats=active_per_head_stats,
        )
        return Compose([rescale_transform, normalize_transform])

    def _build_collate(self, trainer):
        magno_cfg = trainer.model_config["args"]["magno"]
        collate_fn = create_collate_function(
            coord_dim=magno_cfg["gno_coord_dim"],
            magno_radius=magno_cfg["gno_radius"],
            magno_scales=magno_cfg["scales"],
            latent_tokens=trainer.latent_tokens,
            neighbor_search_method=magno_cfg["neighbor_strategy"],
            k_neighbors=magno_cfg["k_neighbors"],
            asynchronous_graph_building=magno_cfg.get("asynchronous_graph_building", False),
        )
        trainer.collate_fn = collate_fn
        if magno_cfg.get("asynchronous_graph_building", False):
            trainer.model_config["args"]["magno"]["precompute_edges"] = True
        return collate_fn

    def _build_datasets_and_loaders(self, trainer, composed_transform, collate_fn):
        if trainer.setup_config["train"]:
            train_ds = VTKMeshDataset(
                trainer.data_root, trainer.order_file_path, trainer.dataset_config, "train", composed_transform
            )
            val_ds = VTKMeshDataset(trainer.data_root, trainer.order_file_path, trainer.dataset_config, "val", composed_transform)
            train_sampler = self._sampler(trainer, train_ds, trainer.dataset_config["shuffle"], True)
            trainer.train_loader = self._loader(trainer, train_ds, collate_fn, train_sampler, trainer.dataset_config["shuffle"])
            val_sampler = self._sampler(trainer, val_ds, False, False)
            trainer.val_loader = self._loader(trainer, val_ds, collate_fn, val_sampler, False)
        test_ds = VTKMeshDataset(trainer.data_root, trainer.order_file_path, trainer.dataset_config, "test", composed_transform)
        trainer.test_loader = self._loader(trainer, test_ds, collate_fn, None, False)
        return test_ds

    def _detect_channel_counts(self, trainer, test_ds) -> None:
        encoder_feature_attr = list(trainer.model_config["args"]["magno"].get("encoder_feature_attr", "x"))
        trainer.num_input_channels = 0
        for attr_name in encoder_feature_attr:
            if hasattr(test_ds[0], attr_name):
                trainer.num_input_channels += getattr(test_ds[0], attr_name).shape[1]
            else:
                trainer.num_input_channels += test_ds[0].pos.shape[1]
        trainer.num_output_channels = test_ds[0].y.shape[1] if hasattr(test_ds[0], "y") and test_ds[0].y is not None else 2

    def _sampler(self, trainer, dataset, shuffle: bool, drop_last: bool):
        if not trainer.setup_config["distributed"]:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=trainer.setup_config["world_size"],
            rank=trainer.setup_config["rank"],
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def _loader(self, trainer, dataset, collate_fn, sampler, shuffle: bool):
        seed_worker, loader_generator = self._dataloader_seed_tools(trainer)
        return DataLoader(
            dataset,
            batch_size=trainer.dataset_config["batch_size"],
            shuffle=False if sampler is not None else shuffle,
            num_workers=trainer.dataset_config["num_workers"],
            sampler=sampler,
            pin_memory=False,
            drop_last=sampler is not None,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
            generator=loader_generator,
        )

    def _dataloader_seed_tools(self, trainer):
        base_seed = int(trainer.setup_config["seed"]) + int(trainer.setup_config.get("rank", 0))
        def seed_worker(worker_id: int) -> None:
            worker_seed = (base_seed + worker_id) % (2**32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        generator = torch.Generator()
        generator.manual_seed(base_seed)
        return seed_worker, generator

    def _prepare_viz_settings(self, trainer) -> None:
        trainer._viz_fixed_axis_limits = {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "z": (-3.0, 3.0)}
        axis_cfg = trainer.dataset_config.get("viz_axis_limits")
        if axis_cfg is not None:
            trainer._viz_fixed_axis_limits = {
                axis: (float(axis_cfg[axis][0]), float(axis_cfg[axis][1])) for axis in ("x", "y", "z")
            }
        trainer._viz_fixed_clim_common = (
            float(trainer.dataset_config["viz_field_vmin"]),
            float(trainer.dataset_config["viz_field_vmax"]),
        )
        trainer._viz_fixed_clim_diff = (0.0, float(trainer.dataset_config["viz_abs_diff_vmax"]))
