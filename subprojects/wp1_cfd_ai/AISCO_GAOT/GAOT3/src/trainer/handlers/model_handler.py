"""ModelHandler is called with config and trainer context to build the GAOT3D model."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import torch
from omegaconf import OmegaConf

from src.model.gaot_3d import GAOT3D


class ModelHandler:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def build_model(self, trainer) -> None:
        config_dict = self._build_gaot_config(trainer.model_config, trainer.num_input_channels, trainer.num_output_channels)
        if trainer.active_heads is not None:
            decoder_cfg = config_dict.get("decoder", {})
            orig_heads = decoder_cfg.get("output_heads", {"force": 1, "cd": 1})
            filtered = {key: value for key, value in orig_heads.items() if key in trainer.active_heads}
            decoder_cfg["output_heads"] = filtered
            config_dict["decoder"] = decoder_cfg
            logging.getLogger(__name__).info(
                f"Filtered output_heads by active_heads={trainer.active_heads}: {orig_heads} -> {filtered}"
            )
        trainer.model = GAOT3D(config=config_dict)
        trainer.model.to(trainer.device)
        if trainer.setup_config["distributed"]:
            trainer.model = torch.nn.parallel.DistributedDataParallel(
                trainer.model,
                device_ids=[trainer.setup_config["local_rank"]],
                output_device=trainer.setup_config["local_rank"],
                find_unused_parameters=True,
            )

    def _build_gaot_config(self, model_config, input_size: int, output_size: int) -> dict[str, Any]:
        args = model_config["args"]
        if hasattr(args, "__dataclass_fields__"):
            cfg = asdict(args)
        elif isinstance(args, dict):
            cfg = dict(args)
        else:
            cfg_container = OmegaConf.to_container(args, resolve=True)
            cfg = {str(key): value for key, value in cfg_container.items()} if isinstance(cfg_container, dict) else {}
        cfg["input_size"] = input_size
        cfg["output_size"] = output_size
        return cfg
