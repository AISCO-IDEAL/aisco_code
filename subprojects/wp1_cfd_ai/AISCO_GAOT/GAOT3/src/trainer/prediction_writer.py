"""PredictionWriter is called with a config dict and writes prediction CSV arrays."""

from __future__ import annotations

import csv

import numpy as np


class PredictionWriter:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def write_predictions(self, values, header: str, path: str | None = None) -> None:
        np.savetxt(path or self._general_path(), values, delimiter=",", header=header, comments="")

    def write_field_metrics(self, rows: list[dict], path: str | None = None) -> None:
        with open(path or self._general_path(), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "n_nodes", "MSE", "MAE", "Rel_L2_pct"])
            writer.writeheader()
            writer.writerows(rows)

    def _general_path(self) -> str:
        return self.config["path"]["result_path"].replace(".png", "_predictions.csv")
