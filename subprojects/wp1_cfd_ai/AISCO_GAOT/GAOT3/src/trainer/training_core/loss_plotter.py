"""LossPlotter is called with a config dict and writes loss PNG and NPZ history."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


class LossPlotter:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def save_loss_history(self, **kwargs) -> None:
        epochs = kwargs["epochs"]
        losses = kwargs["losses"]
        val_epochs = kwargs.get("val_epochs")
        val_losses = kwargs.get("val_losses")
        best_epoch = kwargs.get("best_epoch")
        best_loss = kwargs.get("best_loss")
        if val_losses is None:
            self._save_train_loss(epochs, losses, best_epoch, best_loss)
        else:
            self._save_train_val_loss(epochs, losses, val_epochs, val_losses, best_epoch, best_loss)

    def _save_train_loss(self, epochs, losses, best_epoch, best_loss) -> None:
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
        os.makedirs(os.path.dirname(self.config["path"]["loss_path"]), exist_ok=True)
        plt.savefig(self.config["path"]["loss_path"])
        np.savez(self.config["path"]["loss_path"][:-4] + ".npz", epochs=epochs, losses=losses)

    def _save_train_val_loss(self, epochs, losses, val_epochs, val_losses, best_epoch, best_loss) -> None:
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
        os.makedirs(os.path.dirname(self.config["path"]["loss_path"]), exist_ok=True)
        plt.savefig(self.config["path"]["loss_path"])
        np.savez(
            self.config["path"]["loss_path"][:-4] + ".npz",
            epochs=epochs,
            losses=losses,
            val_epochs=val_epochs,
            val_losses=val_losses,
        )
