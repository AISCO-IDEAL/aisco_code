import logging
import os
from typing import Any

import wandb


class WandbTracker:
    """Handles all Weights & Biases logging and artifact management."""

    def __init__(self, setup_config, path_config, full_config_dict):
        self.setup_config = setup_config
        self.path_config = path_config
        self.wandb_run = None
        self.logger = logging.getLogger(__name__)

        # Only initialize if enabled in config AND process is Rank 0
        self.enabled = getattr(self.setup_config, "wandb", False) and getattr(self.setup_config, "rank", 0) == 0

        if self.enabled:
            self._init_wandb(full_config_dict)

    def _init_wandb(self, config_dict: dict[str, Any]) -> None:
        self.logger.info(f"Rank {self.setup_config.rank}: Initializing wandb...")

        # Respect mode: 'online' | 'offline' | 'disabled'
        if hasattr(self.setup_config, "wandb_mode") and self.setup_config.wandb_mode:
            os.environ["WANDB_MODE"] = str(self.setup_config.wandb_mode)

        init_kwargs = {
            "project": getattr(self.setup_config, "wandb_project", None),
            "entity": getattr(self.setup_config, "wandb_entity", None),
            "name": getattr(self.setup_config, "wandb_run_name", None),
            "group": getattr(self.setup_config, "wandb_group", None),
            "notes": getattr(self.setup_config, "wandb_notes", None),
            "tags": getattr(self.setup_config, "wandb_tags", None),
            "config": config_dict,
            # Using finish_previous instead of deprecated reinit
            "resume": "allow",
        }
        # Remove None values to avoid API complaints
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        try:
            self.wandb_run = wandb.init(**init_kwargs)  # type: ignore[arg-type]
        except Exception as e:
            self.logger.warning(f"Failed to initialize Weights & Biases: {e}")
            self.wandb_run = None

    def watch_model(self, model) -> None:
        if not self.enabled or not self.wandb_run:
            return
        if not getattr(self.setup_config, "wandb_watch_model", False):
            return
        try:
            # Handle DDP wrapped models safely
            raw_model = model.module if hasattr(model, "module") else model
            wandb.watch(raw_model, log="all", log_freq=100)
        except Exception as e:
            self.logger.warning(f"wandb.watch failed: {e}")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or not self.wandb_run:
            return
        try:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
        except Exception as e:
            self.logger.warning(f"Failed to log metrics to W&B: {e}")

    def save_checkpoint_artifact(
        self, epoch: int, train_loss: float, val_loss: float, is_best: bool = False, config_dict: dict | None = None
    ) -> None:
        if not self.enabled or not self.wandb_run:
            return

        artifact_name = getattr(self.setup_config, "wandb_run_name", "model_ckpt")
        artifact = wandb.Artifact(name=artifact_name, type="model", description=f"Model checkpoint at epoch {epoch}")

        ckpt_path = getattr(self.path_config, "ckpt_path", None)
        if ckpt_path and os.path.exists(ckpt_path):
            artifact.add_file(ckpt_path, name=os.path.basename(ckpt_path))

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
        except Exception as e:
            self.logger.warning(f"Failed to log artifact: {e}")

    def download_artifact(self, artifact_name: str) -> tuple[str | None, dict | None]:
        """Downloads a wandb artifact and returns the path and metadata."""
        if not self.enabled:
            self.logger.warning("W&B is not enabled; cannot download artifact.")
            return None, None

        try:
            artifact = wandb.use_artifact(artifact_name)
            artifact_dir = os.path.join(".artifacts", artifact.name)
            if not os.path.exists(artifact_dir):
                self.logger.info(f"Downloading artifact from wandb: {artifact_name}")
                artifact_dir = artifact.download(root=artifact_dir)

            ckpt_files = [f for f in os.listdir(artifact_dir) if f.endswith(".pt")]
            if not ckpt_files:
                self.logger.error("No .pt file found in the artifact.")
                return None, artifact.metadata

            if len(ckpt_files) > 1:
                self.logger.warning(f"Multiple .pt files found, using first one: {ckpt_files[0]}")

            ckpt_path = os.path.join(artifact_dir, ckpt_files[0])
        except Exception as e:
            self.logger.warning(f"Failed to download artifact: {e}")
            return None, None
        else:
            return ckpt_path, artifact.metadata

    def close(self) -> None:
        if not self.enabled or not self.wandb_run:
            return
        try:
            wandb.finish()
        except Exception as e:
            self.logger.warning(f"Failed to finalize W&B run: {e}")
