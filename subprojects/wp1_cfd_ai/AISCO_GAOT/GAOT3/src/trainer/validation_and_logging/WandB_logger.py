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

    def save_checkpoint_artifact(
        self, epoch: int, train_loss: float, val_loss: float, is_best: bool = False, config_dict: dict | None = None
    ) -> None:
        if not self.enabled or not self.wandb_run:
            return
        artifact_name = self.config["setup"].get("wandb_run_name", "model_ckpt")
        artifact = wandb.Artifact(name=artifact_name, type="model", description=f"Model checkpoint at epoch {epoch}")
        if self.config["path"].get("ckpt_path") and os.path.exists(self.config["path"]["ckpt_path"]):
            artifact.add_file(self.config["path"]["ckpt_path"], name=os.path.basename(self.config["path"]["ckpt_path"]))
        artifact.metadata = {
            "epoch": epoch,
            "train_loss": float(train_loss) if train_loss is not None else None,
            "val_loss": float(val_loss) if val_loss is not None else None,
            "is_best": is_best,
            "model_config": config_dict.get("model", {}) if config_dict else {},
            "dataset_config": config_dict.get("dataset", {}) if config_dict else {},
        }
        aliases = ["latest"]
        if is_best:
            aliases.append("best")
        try:
            wandb.log_artifact(artifact, aliases=aliases)
            self.logger.info(f"Saved checkpoint artifact: {artifact_name}:{aliases}")
        except Exception as exc:
            self.logger.warning(f"Failed to log artifact: {exc}")

    def download_artifact(self, artifact_name: str) -> tuple[str | None, dict | None]:
        if not self.enabled:
            self.logger.warning("W&B is not enabled; cannot download artifact.")
            return None, None
        try:
            artifact = wandb.use_artifact(artifact_name)
            artifact_dir = os.path.join(".artifacts", artifact.name)
            if not os.path.exists(artifact_dir):
                self.logger.info(f"Downloading artifact from wandb: {artifact_name}")
                artifact_dir = artifact.download(root=artifact_dir)
            ckpt_files = [name for name in os.listdir(artifact_dir) if name.endswith(".pt")]
            if not ckpt_files:
                self.logger.error("No .pt file found in the artifact.")
                return None, artifact.metadata
            if len(ckpt_files) > 1:
                self.logger.warning(f"Multiple .pt files found, using first one: {ckpt_files[0]}")
            return os.path.join(artifact_dir, ckpt_files[0]), artifact.metadata
        except Exception as exc:
            self.logger.warning(f"Failed to download artifact: {exc}")
            return None, None

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
