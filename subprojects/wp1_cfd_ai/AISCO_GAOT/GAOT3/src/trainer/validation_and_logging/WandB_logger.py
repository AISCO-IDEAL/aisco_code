"""WandBLogger is called with the run config dict and owns W&B logging and artifacts."""

from __future__ import annotations

import logging
import os
from typing import Any

import wandb


class WandBLogger:
    def __init__(self, config: dict) -> None:
        self.config: dict = config
        self.wandb_run = None
        self.logger = logging.getLogger(__name__)
        self.enabled = self.config["setup"]["wandb"] and self.config["setup"]["rank"] == 0
        if self.enabled:
            self._init_wandb()

    def watch_model(self, model) -> None:
        if not self.enabled or not self.wandb_run:
            return
        if not self.config["setup"].get("wandb_watch_model", False):
            return
        try:
            raw_model = model.module if hasattr(model, "module") else model
            wandb.watch(raw_model, log="all", log_freq=100)
        except Exception as exc:
            self.logger.warning(f"wandb.watch failed: {exc}")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or not self.wandb_run:
            return
        try:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
        except Exception as exc:
            self.logger.warning(f"Failed to log metrics to W&B: {exc}")

    def log_image(self, key: str, image, step: int | None = None, caption: str | None = None) -> None:
        if not self.enabled or not self.wandb_run:
            return
        try:
            payload = {key: wandb.Image(image, caption=caption)}
            if step is not None:
                wandb.log(payload, step=step)
            else:
                wandb.log(payload)
        except Exception as exc:
            self.logger.warning(f"Failed to log image to W&B: {exc}")

    def close(self) -> None:
        if not self.enabled or not self.wandb_run:
            return
        try:
            wandb.finish()
        except Exception as exc:
            self.logger.warning(f"Failed to finalize W&B run: {exc}")

    def _init_wandb(self) -> None:
        self.logger.info(f"Rank {self.config['setup']['rank']}: Initializing wandb...")
        if self.config["setup"].get("wandb_mode"):
            os.environ["WANDB_MODE"] = str(self.config["setup"]["wandb_mode"])
        init_kwargs = {
            "project": self.config["setup"]["wandb_project"],
            "entity": self.config["setup"]["wandb_entity"],
            "name": self.config["setup"]["wandb_run_name"],
            "group": self.config["setup"]["wandb_group"],
            "notes": self.config["setup"]["wandb_notes"],
            "tags": self.config["setup"]["wandb_tags"],
            "config": self.config,
            "resume": "allow",
        }
        init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
        try:
            self.wandb_run = wandb.init(**init_kwargs)
        except Exception as exc:
            self.logger.warning(f"Failed to initialize Weights & Biases: {exc}")
            self.wandb_run = None
