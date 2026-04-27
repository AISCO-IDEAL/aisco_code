"""ResultWriter is called with a config dict and appends the final datarow CSV."""

from __future__ import annotations

import os

import pandas as pd


class ResultWriter:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def append_datarow(self) -> None:
        if os.path.exists(self.config["path"]["database_path"]):
            database = pd.read_csv(self.config["path"]["database_path"])
        else:
            database = pd.DataFrame(columns=self.config["datarow"].keys())
        database.loc[len(database)] = self.config["datarow"]
        database.to_csv(self.config["path"]["database_path"], index=False)
