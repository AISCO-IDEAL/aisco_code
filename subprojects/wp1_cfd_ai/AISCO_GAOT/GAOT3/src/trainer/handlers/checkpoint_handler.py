"""CheckpointHandler is called with a config dict and a trainer to load and save checkpoints."""

from __future__ import annotations

import logging
import os
import shutil

import yaml

from ..utils.setup import load_ckpt, save_ckpt


class CheckpointHandler:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def load_for_training(self, trainer) -> None:
        if self.config["setup"].get("resume_from_ckpt") is not None:
            trainer.load_ckpt(path=self.config["setup"]["resume_from_ckpt"])
        elif self.config["setup"]["ckpt"]:
            trainer.load_ckpt()

    def load_checkpoint(self, trainer, path: str | None = None):
        if path is None:
            path = self.config["path"]["ckpt_path"]
        load_ckpt(path, model=trainer.model)
        return trainer

    def save_checkpoint(self, trainer, path: str | None = None):
        if path is None:
            path = self.config["path"]["ckpt_path"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_ckpt(path, model=trainer.model)
        return trainer

    def load_for_testing(self, trainer) -> None:
        if self.config["setup"].get("resume_from_ckpt") is not None:
            trainer.load_ckpt(path=self.config["setup"]["resume_from_ckpt"])
        else:
            trainer.load_ckpt()

    def save_model_config(self, trainer) -> None:
        ckpt_path = self.config["path"]["ckpt_path"]
        config_save_path = ckpt_path.replace(".pt", "_config.yaml")
        raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
        with open(config_save_path, "w") as f:
            yaml.dump(raw_model.config, f, default_flow_style=False, sort_keys=False)
        logging.getLogger(__name__).info(f"Model config saved to {config_save_path}")

    def copy_run_config(self) -> None:
        if "config_save_path" not in self.config["path"]:
            return
        config_file_path = self.config.get("_config_file_path")
        full_config_save_path = self.config["path"]["config_save_path"]
        if config_file_path and os.path.exists(config_file_path):
            os.makedirs(os.path.dirname(full_config_save_path), exist_ok=True)
            shutil.copy(config_file_path, full_config_save_path)
            logging.getLogger(__name__).info(f"Config copied to {full_config_save_path}")
        else:
            logging.getLogger(__name__).warning(f"Could not find config file to copy: {config_file_path}")
