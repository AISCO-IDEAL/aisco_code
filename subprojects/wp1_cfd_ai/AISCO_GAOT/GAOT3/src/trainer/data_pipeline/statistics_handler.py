"""StatisticsHandler is called with config and trainer context to prepare normalization stats."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from src.dataloader.pyg_datasets import VTKMeshDataset
from src.dataloader.pyg_transforms import RescalePosition


class StatisticsHandler:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def calculate_or_load_stats(self, trainer, order_file_path: str, data_root: str) -> None:
        stats_file = os.path.join(data_root, trainer.dataset_config["name"] + "_norm_stats.pt")
        need_compute = not os.path.exists(stats_file) or trainer.dataset_config["force_recompute_stats"]
        if need_compute:
            if trainer.setup_config["rank"] == 0:
                self._compute_and_save_stats(trainer, order_file_path, stats_file)
            if trainer.setup_config["distributed"]:
                torch.distributed.barrier()
        self._load_stats(trainer, stats_file)

    def _load_stats(self, trainer, stats_file: str) -> None:
        logging.getLogger(__name__).info(f"Loading normalization stats from {stats_file}")
        stats = torch.load(stats_file, weights_only=False)
        trainer.u_mean = stats["mean"].to(trainer.dtype)
        trainer.u_std = stats["std"].to(trainer.dtype)
        trainer.c_mean = None
        trainer.c_std = None
        trainer.per_head_stats = stats.get("per_head_stats", {})
        if trainer.u_mean is None or trainer.u_std is None:
            raise RuntimeError("Normalization mean/std could not be calculated or loaded.")

    def _compute_and_save_stats(self, trainer, order_file_path: str, stats_file: str) -> None:
        logging.getLogger(__name__).info("Calculating normalization statistics from training set...")
        temp_train_dataset = VTKMeshDataset(
            root=trainer.dataset_config["base_path"],
            order_file=order_file_path,
            dataset_config=trainer.dataset_config,
            split="train",
            transform=RescalePosition(),
        )
        seed_worker, loader_generator = trainer.data_handler._dataloader_seed_tools(trainer)
        temp_loader = PyGDataLoader(
            temp_train_dataset,
            batch_size=trainer.dataset_config["batch_size"],
            shuffle=False,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=loader_generator,
        )
        all_x = []
        all_force = []
        all_cd = []
        for batch in tqdm(temp_loader, desc="Calculating Stats"):
            if hasattr(batch, "force") and batch.force is not None:
                valid_force = batch.force[~torch.isnan(batch.force)]
                if valid_force.numel() > 0:
                    all_force.append(valid_force.cpu())
            if hasattr(batch, "cd") and batch.cd is not None:
                valid_cd = batch.cd[~torch.isnan(batch.cd)]
                if valid_cd.numel() > 0:
                    all_cd.append(valid_cd.cpu())
            if hasattr(batch, "x") and batch.x is not None:
                all_x.append(batch.x.cpu())
            elif hasattr(batch, "y") and batch.y is not None:
                all_x.append(batch.y.cpu())
        per_head_stats = self._per_head_stats(trainer, all_force, all_cd)
        combined: list[torch.Tensor]
        if all_x:
            combined = all_x
        elif all_force or all_cd:
            combined = all_force + all_cd
        else:
            raise ValueError("No data found in training set to calculate statistics.")
        full_x_tensor = torch.cat(combined, dim=0)
        stats_to_save: dict[str, Any] = {
            "mean": torch.mean(full_x_tensor, dim=0, dtype=trainer.dtype),
            "std": torch.std(full_x_tensor, dim=0).to(trainer.dtype),
        }
        if per_head_stats:
            stats_to_save["per_head_stats"] = per_head_stats
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        torch.save(stats_to_save, stats_file)

    def _per_head_stats(self, trainer, all_force, all_cd) -> dict:
        per_head_stats = {}
        if all_force:
            force_tensor = torch.cat(all_force, dim=0)
            per_head_stats["force"] = {
                "mean": torch.mean(force_tensor, dtype=trainer.dtype),
                "std": torch.std(force_tensor).to(trainer.dtype),
            }
        if all_cd:
            cd_tensor = torch.cat(all_cd, dim=0)
            per_head_stats["cd"] = {
                "mean": torch.mean(cd_tensor, dtype=trainer.dtype),
                "std": torch.std(cd_tensor).to(trainer.dtype),
            }
        return per_head_stats
