"""PathHandler is called with a config dict and prepares output paths and datarow values."""

from __future__ import annotations

import os
import time

import numpy as np


class PathHandler:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def prepare_paths_and_datarow(self) -> dict:
        self._apply_local_paths()
        basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for path_key in self.config["path"]:
            if os.path.isabs(self.config["path"][path_key]):
                self._create_parent_directory(self.config["path"][path_key])
                continue
            absolute_path = os.path.join(basepath, self.config["path"][path_key])
            self._create_parent_directory(absolute_path)
            self.config["path"][path_key] = absolute_path
        self.config["datarow"] = {key: value for key, value in self.config.items() if not key.startswith("_")}
        self.config["datarow"]["nbytes"] = -1
        self.config["datarow"]["nparams"] = -1
        self.config["datarow"]["p2r edges"] = -1
        self.config["datarow"]["r2r edges"] = -1
        self.config["datarow"]["r2p edges"] = -1
        self.config["datarow"]["training time"] = np.nan
        self.config["datarow"]["inference time"] = np.nan
        self.config["datarow"]["time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.config["datarow"]["relative error (direct)"] = np.nan
        self.config["datarow"]["relative error (auto2)"] = np.nan
        self.config["datarow"]["relative error (auto4)"] = np.nan
        return self.config

    def _apply_local_paths(self) -> None:
        if not self.config["setup"].get("run_local", False):
            return
        print("Running in local mode with config overrides:")
        self.config["path"]["ckpt_path"] = self.config["local_path"]["ckpt_path"]
        self.config["path"]["loss_path"] = self.config["local_path"]["loss_path"]
        self.config["path"]["result_path"] = self.config["local_path"]["result_path"]
        self.config["path"]["config_save_path"] = self.config["local_path"]["config_save_path"]
        self.config["path"]["database_path"] = self.config["local_path"]["database_path"]
        self.config["dataset"]["base_path"] = self.config["local_path"]["dataset"]
        self.config["dataset"]["val_files_path"] = "home/aisco/Documents/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/config/val_files.yaml"

    def _create_parent_directory(self, path: str) -> None:
        directory_path = os.path.dirname(path)
        if directory_path and not os.path.exists(directory_path):
            os.makedirs(directory_path)
