"""EdgeHandler is called with config and trainer context to precompute processed edge tensors."""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from src.dataloader.pyg_datasets import EnrichedData
from src.model.layers.magno import get_neighbor_strategy, parse_neighbor_strategy
from src.utils.scale import rescale, rescale_new


class EdgeHandler:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def update_pt_files_with_edges(self, trainer, order_file_path: str, magno_config: dict) -> None:
        if trainer.setup_config["rank"] != 0:
            return
        processed_dir = os.path.join(trainer.dataset_config["base_path"], trainer.dataset_config["processed_folder"])
        if not os.path.isdir(processed_dir):
            raise FileNotFoundError(f"Processed directory for update not found: {processed_dir}")
        filenames_to_process = self._filenames(trainer, order_file_path)
        latent_tokens_cpu = trainer.latent_tokens.cpu()
        for filename in tqdm(filenames_to_process, desc="Updating .pt files"):
            self._update_file(trainer, filename, processed_dir, magno_config, latent_tokens_cpu)
        logging.getLogger(__name__).info("Rank 0: Finished checking/updating '.pt' files.")

    def _filenames(self, trainer, order_file_path: str) -> list[str]:
        with open(order_file_path) as f:
            all_filenames_base = [line.strip() for line in f if line.strip()]
        total_samples = len(all_filenames_base)
        indices = np.arange(total_samples)
        train_indices = indices[: trainer.dataset_config["train_size"]]
        val_indices = indices[trainer.dataset_config["train_size"] : trainer.dataset_config["train_size"] + trainer.dataset_config["val_size"]]
        test_indices = indices[-trainer.dataset_config["test_size"] :]
        relevant_indices = np.concatenate([train_indices, val_indices, test_indices])
        return [f"{all_filenames_base[index]}.pt" for index in relevant_indices]

    def _update_file(self, trainer, filename: str, processed_dir: str, magno_config: dict, latent_tokens_cpu) -> None:
        fpath = os.path.join(processed_dir, filename)
        fpath_tmp = fpath + ".tmp"
        try:
            data_old = torch.load(fpath, map_location="cpu", weights_only=False)
            x_attr = data_old.y if hasattr(data_old, "y") and data_old.y is not None else torch.zeros(data_old.pos.shape[0], 1)
            data = EnrichedData(pos=data_old.pos, x=x_attr)
            for attr, value in data_old:
                if attr not in ["pos", "y", "force", "cd"]:
                    setattr(data, attr, value)
            self._set_edges(trainer, data, magno_config, latent_tokens_cpu)
            torch.save(data, fpath_tmp)
            os.replace(fpath_tmp, fpath)
        except FileNotFoundError:
            logging.getLogger(__name__).warning(f"Warning: File not found during update: {fpath}")
        except Exception:
            logging.getLogger(__name__).exception(f"Error processing file {fpath}")
            if os.path.exists(fpath_tmp):
                os.remove(fpath_tmp)

    def _set_edges(self, trainer, data, magno_config: dict, latent_tokens_cpu) -> None:
        data.num_latent_nodes = latent_tokens_cpu.shape[0]
        if trainer.dataset_config["use_rescale_new"]:
            phys_pos = rescale_new(data.pos.to(torch.float), (-1, 1), trainer.metadata.domain_x)
        else:
            phys_pos = rescale(data.pos.to(torch.float), (-1, 1))
        batch_idx_phys = torch.zeros(phys_pos.shape[0], dtype=torch.long)
        batch_idx_latent = torch.zeros(latent_tokens_cpu.shape[0], dtype=torch.long)
        encoder_strategy, decoder_strategy = parse_neighbor_strategy(magno_config.get("neighbor_strategy", "radius"))
        for scale_idx, scale in enumerate(magno_config.get("scales", [1.0])):
            radius = magno_config.get("gno_radius", 0.15) * scale
            enc_edge_index = get_neighbor_strategy(
                encoder_strategy, phys_pos, batch_idx_phys, latent_tokens_cpu, batch_idx_latent, radius, magno_config.get("k_neighbors", 1), False
            ).to(torch.int32)
            dec_edge_index = get_neighbor_strategy(
                decoder_strategy, phys_pos, batch_idx_phys, latent_tokens_cpu, batch_idx_latent, radius, magno_config.get("k_neighbors", 1), True
            ).to(torch.int32)
            setattr(data, f"encoder_edge_index_s{scale_idx}", enc_edge_index)
            setattr(data, f"decoder_edge_index_s{scale_idx}", dec_edge_index)
            setattr(data, f"encoder_query_counts_s{scale_idx}", self._counts(enc_edge_index, latent_tokens_cpu.shape[0], 1))
            setattr(data, f"decoder_query_counts_s{scale_idx}", self._counts(dec_edge_index, phys_pos.shape[0], 1))

    def _counts(self, edge_index, length: int, dim: int):
        if edge_index.numel() > 0:
            return torch.bincount(edge_index[dim], minlength=length).to(torch.int32)
        return torch.zeros(length, dtype=torch.int32)
