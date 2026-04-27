"""ConfigHandler is called by main.py with CLI arguments and returns config dictionaries."""

from __future__ import annotations

import argparse
import ast
import os
from typing import Any, cast

from omegaconf import OmegaConf


class ConfigHandler:
    def __init__(self, config: dict | None = None) -> None:
        self.config: dict = config or {}

    def parse_command_line(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", type=str, default=None, help="config file path")
        parser.add_argument("-f", "--folder", type=str, default=None, help="folder path")
        parser.add_argument("--debug", action="store_true", help="debug mode, to dispalce multiprocessing")
        parser.add_argument("--num_works_per_device", type=int, default=10, help="number of works per device")
        parser.add_argument("--visible_devices", nargs="*", type=int, default=None, help="visible devices")
        args, unknown = parser.parse_known_args()
        args.wandb_overrides = self._parse_wandb_overrides(unknown)
        assert args.config or args.folder, "Please specify --config or --folder"
        if args.visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.visible_devices))
        if args.config:
            args.arg_files = [args.config]
        else:
            args.arg_files = []
            for root, _dirs, files in os.walk(args.folder):
                for name in files:
                    if name.endswith((".toml", ".json", ".yaml")):
                        args.arg_files.append(os.path.join(root, name))
        return args

    def load_config(self, filename: str, overrides: dict[str, Any] | None = None) -> dict:
        if not filename.endswith((".toml", ".json", ".yaml")):
            raise NotImplementedError(f"File type {filename} not supported, currently only toml and json are supported.")
        with open(filename) as f:
            data = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
        if not isinstance(data, dict):
            raise TypeError(f"Expected config mapping in {filename}, got {type(data).__name__}")
        self.config = cast(dict[str, Any], data)
        self.config["_config_file_path"] = filename
        if overrides:
            self.apply_overrides(overrides)
        return self.config

    def apply_overrides(self, overrides: dict[str, Any]) -> dict:
        for key, value in overrides.items():
            self._set_nested_value(key, value)
        return self.config

    def _parse_wandb_overrides(self, unknown_args: list[str]) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        for arg in unknown_args:
            if not arg.startswith("--"):
                continue
            item = arg[2:]
            if "=" not in item:
                continue
            key, raw_value = item.split("=", 1)
            lower = raw_value.lower()
            value: Any
            if lower == "true":
                value = True
            elif lower == "false":
                value = False
            elif lower in ("none", "null"):
                value = None
            else:
                try:
                    value = ast.literal_eval(raw_value)
                except (ValueError, SyntaxError):
                    value = raw_value
            overrides[key] = value
        return overrides

    def _set_nested_value(self, dotted_key: str, value: Any) -> None:
        keys = dotted_key.split(".")
        current = self.config
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
