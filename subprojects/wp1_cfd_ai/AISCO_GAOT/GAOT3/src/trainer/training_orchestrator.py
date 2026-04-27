"""TrainingOrchestrator is called with a config dict and runs the GAOT3 train/test flow."""

from __future__ import annotations

import logging
import math
import os

import pandas as pd
import torch
import torch.distributed as dist

from .Data_handler import DataHandler
from .WandB_logger import WandBLogger
from .checkpoint_handler import CheckpointHandler
from .loss_fn import LossFn
from .loss_plotter import LossPlotter
from .model_handler import ModelHandler
from .optimiser import Optimiser
from .path_handler import PathHandler
from .runtime_handler import RuntimeHandler
from .test_calculator import TestCalculator
from .validation_calculator import ValidationCalculator


class TrainingOrchestrator:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def run(self):
        self.config = PathHandler(self.config).prepare_paths_and_datarow()
        self._setup_logging()
        trainer = self._build_trainer()
        checkpoint_handler = CheckpointHandler(self.config)
        try:
            if self.config["setup"]["train"]:
                checkpoint_handler.load_for_training(trainer)
                trainer.fit()
                if self.config["setup"].get("rank", 0) == 0:
                    checkpoint_handler.save_model_config(trainer)
                    checkpoint_handler.copy_run_config()
            if self.config["setup"]["test"]:
                self._run_test(trainer, checkpoint_handler)
            if self.config["setup"].get("rank", 0) == 0:
                self._append_datarow()
        finally:
            if self.config["setup"].get("rank", 0) == 0 and hasattr(self, "tracker"):
                self.tracker.close()
            if dist.is_initialized():
                dist.destroy_process_group()
        return trainer

    def _build_trainer(self):
        if self.config["setup"]["trainer_name"] != "static3d":
            raise KeyError(self.config["setup"]["trainer_name"])
        RuntimeHandler(self.config).prepare_runtime(self)
        self.data_handler = DataHandler(self.config)
        self.data_handler.prepare_data(self)
        ModelHandler(self.config).build_model(self)
        self.loss_handler = LossFn(self.config)
        self.loss_fn = self.loss_handler.loss_fn
        params = self._prepare_loss_parameters()
        self.optimizer = {"adamw": Optimiser}[self.config["optimizer"]["name"]](params, self.config)
        self.full_config_dict = RuntimeHandler(self.config).serializable_config()
        self.tracker = WandBLogger(self.full_config_dict)
        self.tracker.watch_model(self.model)
        self._record_model_size()
        return self

    def _run_test(self, trainer, checkpoint_handler: CheckpointHandler) -> None:
        if self.config["setup"].get("rank", 0) != 0:
            if dist.is_initialized():
                dist.barrier()
            return
        if dist.is_initialized():
            dist.barrier()
        checkpoint_handler.load_for_testing(trainer)
        trainer.test()

    def _append_datarow(self) -> None:
        if os.path.exists(self.config["path"]["database_path"]):
            database = pd.read_csv(self.config["path"]["database_path"])
        else:
            database = pd.DataFrame(columns=self.config["datarow"].keys())
        database.loc[len(database)] = self.config["datarow"]
        database.to_csv(self.config["path"]["database_path"], index=False)

    def fit(self, _verbose=False):
        result = self.optimizer.optimise(self)
        self.config["datarow"]["training time"] = result["time"]
        if len(result["train"]["loss"]) == 0:
            if self.config["setup"]["rank"] == 0:
                self.test()
            return
        kwargs = {"epochs": result["train"]["epoch"], "losses": result["train"]["loss"]}
        if "valid" in result:
            kwargs["val_epochs"] = result["valid"]["epoch"]
            kwargs["val_losses"] = result["valid"]["loss"]
        if "best" in result:
            kwargs["best_epoch"] = result["best"]["epoch"]
            kwargs["best_loss"] = result["best"]["loss"]
        if self.config["setup"]["rank"] == 0:
            LossPlotter(self.config).save_loss_history(self, **kwargs)
            self.tracker.log_metrics(
                {"best/loss": result["best"]["loss"], "best/epoch": result["best"]["epoch"], "training_time": result["time"]}
            )
            self.save_ckpt()
            if self.config["setup"]["wandb"]:
                self.tracker.save_checkpoint_artifact(
                    epoch=result["best"]["epoch"],
                    train_loss=result["best"]["loss"],
                    val_loss=result.get("best", {}).get("loss"),
                    is_best=True,
                    config_dict=self.full_config_dict,
                )
            torch.cuda.empty_cache()

    def train_step(self, batch):
        return self.loss_handler.calculate_train_loss(self, batch)

    def validate(self, loader, epoch_name):
        return ValidationCalculator(self.config).validate(self, loader, epoch_name)

    def test(self):
        return TestCalculator(self.config).run_test(self)

    def visualize_sample(self, epoch: int) -> None:
        ValidationCalculator(self.config).visualize_samples(self, epoch)

    def save_ckpt(self, path: str | None = None):
        return CheckpointHandler(self.config).save_checkpoint(self, path)

    def load_ckpt(self, path: str | None = None):
        return CheckpointHandler(self.config).load_checkpoint(self, path)

    def load_from_artifact(self, artifact_name: str):
        return CheckpointHandler(self.config).load_artifact(self, artifact_name)

    def _setup_logging(self) -> None:
        level = os.environ.get("LOG_LEVEL", "INFO")
        log_file = os.environ.get("LOG_FILE", None) or self.config["path"].get("log_path", None)
        if isinstance(level, str):
            parsed_level = getattr(logging, level.upper(), logging.INFO)
        else:
            parsed_level = level
        handlers: list[logging.Handler] = [logging.StreamHandler()]
        if log_file is not None:
            try:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                handlers.append(logging.FileHandler(log_file))
            except Exception:
                pass
        logging.basicConfig(
            level=parsed_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
        )

    def _record_model_size(self) -> None:
        if self.config["setup"]["rank"] != 0:
            return
        nparam = sum([p.numel() * 2 if p.is_complex() else p.numel() for p in self.model.parameters()])
        nbytes = sum(
            [
                p.numel() * 2 * p.element_size() if p.is_complex() else p.numel() * p.element_size()
                for p in self.model.parameters()
            ]
        )
        logging.getLogger(__name__).info(f"Number of parameters: {nparam}")
        self.config["datarow"]["nparams"] = nparam
        self.config["datarow"]["nbytes"] = nbytes

    def _prepare_loss_parameters(self):
        self._uncertainty_weighting = self.config["dataset"]["uncertainty_weighting"]
        params = list(self.model.parameters())
        if not self._uncertainty_weighting:
            return params
        self.log_sigma_field = torch.nn.Parameter(torch.tensor([math.log(1)], device=self.device))
        self.log_sigma_scalar = torch.nn.Parameter(torch.tensor([math.log(1)], device=self.device))
        logging.getLogger(__name__).info("Uncertainty weighting enabled: log_sigma_field and log_sigma_scalar will be learned.")
        return [
            {"params": params},
            {
                "params": [self.log_sigma_field, self.log_sigma_scalar],
                "lr": self.config["optimizer"]["args"]["lr"],
                "weight_decay": 0.0,
            },
        ]
