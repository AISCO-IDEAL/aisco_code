import dataclasses
import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from ..dataloader.metadata import DATASET_METADATA

# Import the new W&B Tracker
from .logger import WandbTracker
from .optimizers import AdamOptimizer, AdamWOptimizer
from .utils.setup import load_ckpt, manual_seed, save_ckpt


class TrainerBase:
    """
    Base class for all trainers, define the init_dataset, initi_model,
    init_optimizer, train_step, validate, test for coreresponding trainers.

    Attributes:
    ----------
    """

    # Declared here so mypy knows about them; subclasses set them in init_model/init_dataset
    model: nn.Module
    train_loader: Any
    val_loader: Any
    test_loader: Any
    rigraph: Any
    optimizer: AdamOptimizer | AdamWOptimizer

    def __init__(self, args):
        # Config setup
        print("dasdado:", args)
        self.config = args
        self.setup_config = self.config.setup
        self.model_config = self.config.model
        self.dataset_config = self.config.dataset
        self.optimizer_config = self.config.optimizer
        self.path_config = self.config.path

        self.metadata = DATASET_METADATA[self.dataset_config["metaname"]]

        # initialization the distributed learning environment
        if self.setup_config["distributed"]:
            self.init_distributed_mode()
            torch.cuda.set_device(self.setup_config["local_rank"])
            self.device = torch.device("cuda", self.setup_config["local_rank"])
        else:
            self.setup_config["rank"] = 0
            self.setup_config["world_size"] = 1
            self.setup_config["local_rank"] = 0
            self.device = torch.device(self.setup_config["device"])

        manual_seed(self.setup_config["seed"] + self.setup_config["rank"])

        # 1. Handle dtypes and loss first
        if self.setup_config["dtype"] in ["float", "torch.float32", "torch.FloatTensor"]:
            self.dtype = torch.float32
        elif self.setup_config["dtype"] in ["double", "torch.float64", "torch.DoubleTensor"]:
            self.dtype = torch.float64
        else:
            raise ValueError("Invalid dtype:", self.setup_config["dtype"])
        self.loss_fn = nn.MSELoss()

        # 2. Initialize the model components
        self.init_dataset(self.dataset_config)
        self.init_model(self.model_config)
        self.init_optimizer(self.optimizer_config)

        # 3. Initialize the W&B Tracker and watch model
        self.full_config_dict = self._to_serializable_config()
        self.tracker = WandbTracker(self.setup_config, self.path_config, self.full_config_dict)
        self.tracker.watch_model(self.model)

        if self.setup_config["rank"] == 0:
            nparam = sum([p.numel() * 2 if p.is_complex() else p.numel() for p in self.model.parameters()])
            nbytes = sum(
                [
                    p.numel() * 2 * p.element_size() if p.is_complex() else p.numel() * p.element_size()
                    for p in self.model.parameters()
                ]
            )
            logging.getLogger(__name__).info(f"Number of parameters: {nparam}")
            args.datarow["nparams"] = nparam
            args.datarow["nbytes"] = nbytes

    # ------------ init ------------ #
    def init_dataset(self, dataset_config):
        raise NotImplementedError

    def init_model(self, model_config):
        raise NotImplementedError

    def init_optimizer(self, _optimizer_config):
        ### Initialize the optimizer
        self.optimizer = {"adam": AdamOptimizer, "adamw": AdamWOptimizer}[self.optimizer_config["name"]](  # type: ignore[assignment]
            self.model.parameters(), self.optimizer_config["args"]
        )

    def init_distributed_mode(self):
        logger = logging.getLogger(__name__)

        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.setup_config["rank"] = int(os.environ["RANK"])
            self.setup_config["world_size"] = int(os.environ["WORLD_SIZE"])
            self.setup_config["local_rank"] = int(
                os.environ["LOCAL_RANK"]
            )  # 移除默认值,如果没有设置应该报错 Translation by deepl:Remove the default value; if not set, an error should be reported.
        else:
            logger.info("Not using distributed mode")
            self.setup_config["distributed"] = False
            self.setup_config["rank"] = 0
            self.setup_config["world_size"] = 1
            self.setup_config["local_rank"] = 0
            return

        dist.init_process_group(
            backend=str(self.setup_config["backend"] + self.setup_config["nccl"]),
            init_method="env://",
            world_size=self.setup_config["world_size"],
            rank=self.setup_config["rank"],
        )
        dist.barrier()

    # ------------ utils ------------ #

    def _to_serializable_config(self) -> dict[str, Any]:
        """Convert config objects to a nested dict suitable for logging."""

        def to_dict(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return dataclasses.asdict(obj)
            if isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            return obj

        payload: dict[str, Any] = {}
        payload["setup"] = to_dict(self.setup_config)
        payload["model"] = to_dict(self.model_config)
        payload["optimizer"] = {"name": self.optimizer_config["name"], "args": to_dict(self.optimizer_config["args"])}
        payload["dataset"] = to_dict(self.dataset_config)
        payload["path"] = to_dict(self.path_config)
        return payload

    def to(self, device):
        self.model.to(device)

    def type(self, dtype):
        # TODO: check if this is necessary, dataloader does not have type method
        self.model.type(dtype)
        self.train_loader.type(dtype)
        self.val_loader.type(dtype)
        self.test_loader.type(dtype)

    def load_ckpt(self, path: str | None = None):
        if path is None:
            path = self.path_config["ckpt_path"]
        load_ckpt(path, model=self.model)
        return self

    def load_from_artifact(self, artifact_name: str) -> dict[str, Any] | None:
        """Uses Tracker to load from an artifact."""
        ckpt_path, metadata = self.tracker.download_artifact(artifact_name)
        if ckpt_path:
            self.load_ckpt(ckpt_path)
            logging.getLogger(__name__).info(f"Successfully loaded model from artifact: {artifact_name}")
        return metadata

    def save_ckpt(self, path: str | None = None):
        """Save checkpoint to the config.ckpt_path"""
        if path is None:
            path = self.path_config["ckpt_path"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_ckpt(path, model=self.model)
        return self

    def compute_test_errors(self):
        # TODO: compute test errors (need to modulate based on the type of dataset, based on metadata)
        raise NotImplementedError

    # ------------ train ------------ #
    def train_step(self, batch):
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
        x_batch, y_batch = x_batch.squeeze(1), y_batch.squeeze(1)
        pred = self.model(self.rigraph, x_batch)
        return self.loss_fn(pred, y_batch)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_in = x_batch.to(self.device).squeeze(1)
                y_in = y_batch.to(self.device).squeeze(1)
                pred = self.model(self.rigraph, x_in)
                loss = self.loss_fn(pred, y_in)
                total_loss += loss.item()
        return total_loss / len(loader)

    def fit(self, _verbose=False):
        result = self.optimizer.optimize(self)
        self.config.datarow["training time"] = result["time"]

        if len(result["train"]["loss"]) == 0:
            if self.setup_config["rank"] == 0:
                self.test()
        else:
            kwargs = {"epochs": result["train"]["epoch"], "losses": result["train"]["loss"]}

            if "valid" in result:
                kwargs["val_epochs"] = result["valid"]["epoch"]
                kwargs["val_losses"] = result["valid"]["loss"]

            if "best" in result:
                kwargs["best_epoch"] = result["best"]["epoch"]
                kwargs["best_loss"] = result["best"]["loss"]

            if self.setup_config["rank"] == 0:
                self.plot_losses(**kwargs)

                # Use Tracker for metrics logging
                self.tracker.log_metrics(
                    {"best/loss": result["best"]["loss"], "best/epoch": result["best"]["epoch"], "training_time": result["time"]}
                )

                self.save_ckpt()

                # Use Tracker for artifact saving
                if self.setup_config["wandb"]:
                    self.tracker.save_checkpoint_artifact(
                        epoch=result["best"]["epoch"],
                        train_loss=result["best"]["loss"],
                        val_loss=result.get("best", {}).get("loss"),
                        is_best=True,
                        config_dict=self.full_config_dict,
                    )
                torch.cuda.empty_cache()
                self.test()

        if self.setup_config["rank"] == 0:
            # Use Tracker to finish the run
            self.tracker.close()

    # ------------ plot ------------ #
    def plot_losses(self, epochs, losses, val_epochs=None, val_losses=None, best_epoch=None, best_loss=None):
        if val_losses is None:
            # plot only train loss
            _fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(epochs, losses)
            ax.scatter([best_epoch], [best_loss], c="r", marker="o", label="best loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Loss vs Epoch")
            ax.legend()
            ax.set_xlim(left=0)
            if (np.array(losses) > 0).all():
                ax.set_yscale("log")
            os.makedirs(os.path.dirname(self.path_config.loss_path), exist_ok=True)
            np.savez(self.path_config.loss_path[:-4] + ".npz", epochs=epochs, losses=losses)

        else:
            # also plot valid loss
            _fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            ax[0].plot(epochs, losses)
            ax[0].scatter([best_epoch], [best_loss], c="r", marker="o", label="best loss")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Loss")
            ax[0].set_title("Loss vs Epoch")
            ax[0].legend()
            ax[0].set_xlim(left=0)
            if (np.array(losses) > 0).all():
                ax[0].set_yscale("log")

            ax[1].plot(val_epochs, val_losses)
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("relative error")
            ax[1].set_title("Loss vs relative error")
            ax[1].legend()
            ax[1].set_xlim(left=0)
            if (np.array(val_losses) > 0).all():
                ax[1].set_yscale("log")
            os.makedirs(os.path.dirname(self.path_config["loss_path"]), exist_ok=True)
            plt.savefig(self.path_config["loss_path"])
            np.savez(
                self.path_config["loss_path"][:-4] + ".npz",
                epochs=epochs,
                losses=losses,
                val_epochs=val_epochs,
                val_losses=val_losses,
            )

    def plot_results(self):
        raise NotImplementedError

    # ------------ test ------------ #
    def variance_test(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
