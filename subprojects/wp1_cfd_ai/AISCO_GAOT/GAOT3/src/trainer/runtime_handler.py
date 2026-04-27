"""RuntimeHandler is called with config and prepares device, dtype, seeds, and DDP."""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

import torch
import torch.distributed as dist

from ..dataloader.metadata import DATASET_METADATA
from .utils.setup import manual_seed


class RuntimeHandler:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def prepare_runtime(self, trainer) -> None:
        trainer.config = self.config
        trainer.setup_config = self.config["setup"]
        trainer.model_config = self.config["model"]
        trainer.dataset_config = self.config["dataset"]
        trainer.optimizer_config = self.config["optimizer"]
        trainer.path_config = self.config["path"]
        trainer.metadata = DATASET_METADATA[trainer.dataset_config["metaname"]]
        self._prepare_distributed(trainer)
        manual_seed(trainer.setup_config["seed"] + trainer.setup_config["rank"])
        self._prepare_dtype(trainer)

    def serializable_config(self) -> dict[str, Any]:
        def to_dict(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return dataclasses.asdict(obj)
            if isinstance(obj, dict):
                return {key: to_dict(value) for key, value in obj.items()}
            return obj

        return {
            "setup": to_dict(self.config["setup"]),
            "model": to_dict(self.config["model"]),
            "optimizer": {"name": self.config["optimizer"]["name"], "args": to_dict(self.config["optimizer"]["args"])},
            "dataset": to_dict(self.config["dataset"]),
            "path": to_dict(self.config["path"]),
        }

    def _prepare_distributed(self, trainer) -> None:
        if trainer.setup_config["distributed"]:
            self._init_distributed_mode(trainer)
            torch.cuda.set_device(trainer.setup_config["local_rank"])
            trainer.device = torch.device("cuda", trainer.setup_config["local_rank"])
            return
        trainer.setup_config["rank"] = 0
        trainer.setup_config["world_size"] = 1
        trainer.setup_config["local_rank"] = 0
        trainer.device = torch.device(trainer.setup_config["device"])

    def _init_distributed_mode(self, trainer) -> None:
        logger = logging.getLogger(__name__)
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            logger.info("Not using distributed mode")
            trainer.setup_config["distributed"] = False
            trainer.setup_config["rank"] = 0
            trainer.setup_config["world_size"] = 1
            trainer.setup_config["local_rank"] = 0
            return
        trainer.setup_config["rank"] = int(os.environ["RANK"])
        trainer.setup_config["world_size"] = int(os.environ["WORLD_SIZE"])
        trainer.setup_config["local_rank"] = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(
            backend=str(trainer.setup_config["backend"] + trainer.setup_config["nccl"]),
            init_method="env://",
            world_size=trainer.setup_config["world_size"],
            rank=trainer.setup_config["rank"],
        )
        dist.barrier()

    def _prepare_dtype(self, trainer) -> None:
        if trainer.setup_config["dtype"] in ["float", "torch.float32", "torch.FloatTensor"]:
            trainer.dtype = torch.float32
            trainer.use_bf16 = False
        elif trainer.setup_config["dtype"] in ["double", "torch.float64", "torch.DoubleTensor"]:
            trainer.dtype = torch.float64
            trainer.use_bf16 = False
        elif trainer.setup_config["dtype"] in ["bfloat16", "torch.bfloat16", "bf16"]:
            trainer.dtype = torch.float32
            trainer.use_bf16 = True
        else:
            raise ValueError("Invalid dtype:", trainer.setup_config["dtype"])
        trainer.autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if trainer.use_bf16
            else torch.amp.autocast(device_type="cuda", enabled=False)
        )
