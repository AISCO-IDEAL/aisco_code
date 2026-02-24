"""
Written by @clhuber
"""

import matplotlib.pyplot as plt
import pandas as pd


class LossTracker:
    __loss_history: dict[str, dict[str, float]]
    __loss_names: list[str]
    __sorted_epoch_names: list[str]

    def __init__(self, loss_names: list[str]):
        self.__loss_history = {name: {} for name in loss_names}
        self.__loss_history["total"] = {}
        self.__loss_names = loss_names
        self.__sorted_epoch_names = []

    def add_epoch(self, losses: dict, epoch_name: str = "-1") -> None:
        if epoch_name == "-1":
            if len(self.__sorted_epoch_names) == 0:
                epoch_name = "0"
            else:
                epoch_name = f"{int(self.__sorted_epoch_names[-1]) + 1}"

        for loss_name, loss_value in losses.items():
            if loss_name not in self.__loss_history:
                print(
                    f"Warning: Loss name '{loss_name}' not recognized. Creating new LossTracker entry."
                )
                self.__loss_history[loss_name] = {}
            self.__loss_history[loss_name][epoch_name] = loss_value
        self.__sorted_epoch_names.append(epoch_name)

    def save_values_to_csv(self, file_path: str) -> None:
        """
        Saves the tracked loss values to a CSV file at the specified file path.
        """
        df = pd.DataFrame(self.__loss_history)
        df.to_csv(file_path, index_label="epoch")

    def import_csv(self, file_path: str) -> None:
        """
        Imports loss values from a CSV file at the specified file path.
        """
        df = pd.read_csv(file_path, index_col="epoch")
        self.__loss_history = df.to_dict(orient="list")
        self.__sorted_epoch_names = list(df.index)

    def plot_losses(self) -> None:
        """
        Plots the tracked loss values over epochs.
        """
        for loss_name in self.__loss_names + ["total"]:
            if loss_name in self.__loss_history:
                epochs = list(self.__loss_history[loss_name].keys())
                values = list(self.__loss_history[loss_name].values())
                plt.plot(epochs, values, label=loss_name)

        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title("Loss Values Over Epochs")
        plt.legend()
        plt.show()

    def plot_normalized_losses(self) -> None:
        """
        Plots the normalized tracked loss values over epochs.
        They are normalized by the total loss at each epoch.
        """
        for loss_name in self.__loss_names:
            if loss_name in self.__loss_history:
                epochs = list(self.__loss_history[loss_name].keys())
                values = list(self.__loss_history[loss_name].values())
                total_values = [self.__loss_history["total"][epoch] for epoch in epochs]
                normalized_values = [
                    v / t if t != 0 else 0 for v, t in zip(values, total_values)
                ]
                plt.plot(epochs, normalized_values, label=loss_name)

        plt.xlabel("Epoch")
        plt.ylabel("Normalized Loss Value")
        plt.title("Normalized Loss Values Over Epochs")
        plt.legend()
        plt.show()
