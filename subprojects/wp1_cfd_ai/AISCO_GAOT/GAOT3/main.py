import argparse
import ast
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from multiprocessing import Process
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from omegaconf import OmegaConf

# Add current directory to path for module discovery
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.trainer.stat import StaticTrainer3D


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """
    Initialize root logging config.
    - level: str or int, default INFO. Examples: "DEBUG", "INFO", "WARNING", "ERROR".
    - log_file: optional file path to store logs; if None, logs only to console
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)  # type: ignore[arg-type]
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            handlers.append(logging.FileHandler(log_file))
        except Exception:
            pass
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers
    )


def parse_wandb_overrides(unknown_args) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    for arg in unknown_args:
        if not arg.startswith("--"):
            continue

        item = arg[2:]
        if "=" not in item:
            continue

        key, raw_value = item.split("=", 1)

        lower = raw_value.lower()
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


def set_nested_value(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        set_nested_value(cfg, key, value)
    return cfg


class FileParser:
    def __init__(self, filename: str, overrides: dict[str, Any] | None = None) -> None:
        if filename.endswith((".toml", ".json", ".yaml")):
            with open(filename) as f:
                data = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
            if not isinstance(data, dict):
                raise TypeError(f"Expected config mapping in {filename}, got {type(data).__name__}")
            self.kwargs: dict[str, Any] = cast(dict[str, Any], data)
            if overrides:
                self.kwargs = apply_overrides(self.kwargs, overrides)
        else:
            raise NotImplementedError(f"File type {filename} not supported, currently only toml and json are supported.")

    def add_argument(self, *args, **kwargs):
        for raw_arg in args:
            key = raw_arg.removeprefix("--")
            if key not in self.kwargs:
                if "action" in kwargs:
                    self.kwargs[key] = False
                else:
                    self.kwargs[key] = kwargs.get("default")

    def parse_args(self):
        return argparse.Namespace(**self.kwargs)


def parse_cmd():
    parser = argparse.ArgumentParser()
    return [parser.parse_args()], True


def parse_files():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=None, help="config file path")
    parser.add_argument("-f", "--folder", type=str, default=None, help="folder path")
    parser.add_argument("--debug", action="store_true", help="debug mode, to dispalce multiprocessing")
    parser.add_argument("--num_works_per_device", type=int, default=10, help="number of works per device")
    parser.add_argument("--visible_devices", nargs="*", type=int, default=None, help="visible devices")
    args, unknown = parser.parse_known_args()
    args.wandb_overrides = parse_wandb_overrides(unknown)
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


def prepare_arg(arg):
    # make sure all paths are exist
    basepath = os.path.dirname(os.path.abspath(__file__))
    for _path in arg.path:
        if os.path.isabs(arg.path[_path]):
            continue
        _abspath = os.path.join(basepath, arg.path[_path])
        _dirpath = os.path.dirname(_abspath)
        # make sure the path directory exist
        if not os.path.exists(_dirpath):
            os.makedirs(_dirpath)
        # turn the relative path to abs path
        arg.path[_path] = _abspath
    arg.datarow = vars(arg).copy()
    arg.datarow["nbytes"] = -1
    arg.datarow["nparams"] = -1
    arg.datarow["p2r edges"] = -1
    arg.datarow["r2r edges"] = -1
    arg.datarow["r2p edges"] = -1
    arg.datarow["training time"] = np.nan
    arg.datarow["inference time"] = np.nan
    arg.datarow["time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    arg.datarow["relative error (direct)"] = np.nan
    arg.datarow["relative error (auto2)"] = np.nan
    arg.datarow["relative error (auto4)"] = np.nan

    return arg


def run_arg(arg):  # noqa: PLR0912, PLR0915
    arg = prepare_arg(arg)
    if arg.setup.get("run_local", False):
        print("Running in local mode with config overrides:")
        arg.path["ckpt_path"] = arg.local_path["ckpt_path"]
        arg.path["loss_path"] = arg.local_path["loss_path"]
        arg.path["result_path"] = arg.local_path["result_path"]
        arg.path["config_save_path"] = arg.local_path["config_save_path"]
        arg.path["database_path"] = arg.local_path["database_path"]
        arg.dataset["base_path"] = arg.local_path["dataset"]
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    log_file = os.environ.get("LOG_FILE", None) or arg.path.get("log_path", None)
    setup_logging(level=log_level, log_file=log_file)

    Trainer = {
        "static3d": StaticTrainer3D,
    }[arg.setup["trainer_name"]]
    t = Trainer(arg)

    if arg.setup["train"]:
        # Load model checkpoint with priority: artifact > resume_from_ckpt > ckpt
        if getattr(arg.setup, "resume_from_artifact", None) is not None:
            metadata = t.load_from_artifact(arg["setup"].resume_from_artifact)
            if metadata:
                logging.getLogger(__name__).info(f"Loaded from artifact: {metadata}")
            else:
                logging.getLogger(__name__).warning("Failed to load from artifact, skipping...")
        elif getattr(arg.setup, "resume_from_ckpt", None) is not None:
            t.load_ckpt(path=arg.setup["resume_from_ckpt"])
        elif arg.setup["ckpt"]:
            t.load_ckpt()
        t.fit()

        # Save the model config (config.yaml style dict) next to the checkpoint
        # so the architecture can always be reconstructed from it.
        if getattr(arg.setup, "rank", 0) == 0:
            ckpt_path = arg.path["ckpt_path"]
            config_save_path = ckpt_path.replace(".pt", "_config.yaml")

            # Get the actual config dict that GAOT3D was built with
            raw_model = t.model.module if hasattr(t.model, "module") else t.model
            model_config_dict = raw_model.config

            os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
            with open(config_save_path, "w") as f:
                yaml.dump(model_config_dict, f, default_flow_style=False, sort_keys=False)
            logging.getLogger(__name__).info(f"Model config saved to {config_save_path}")

            # Save the full training config to the RUN folder by copying the input config file
            if "config_save_path" in arg.path:
                full_config_save_path = arg.path["config_save_path"]
                # Find the config file path from sys.argv
                config_file_path = None
                for i, argv in enumerate(sys.argv):
                    if argv == "-c" and i + 1 < len(sys.argv):
                        config_file_path = sys.argv[i + 1]
                        break

                if config_file_path and os.path.exists(config_file_path):
                    os.makedirs(os.path.dirname(full_config_save_path), exist_ok=True)
                    shutil.copy(config_file_path, full_config_save_path)
                    logging.getLogger(__name__).info(f"Config copied to {full_config_save_path}")
                else:
                    logging.getLogger(__name__).warning(f"Could not find config file to copy: {config_file_path}")

    if arg.setup["test"]:
        # In DDP, only rank 0 runs test (it has the full test loader, no DistributedSampler)
        if getattr(arg.setup, "rank", 0) != 0:
            # Non-rank-0 processes wait, then skip test
            if dist.is_initialized():
                dist.barrier()
        else:
            # Rank 0: ensure checkpoint is fully written before loading
            if dist.is_initialized():
                dist.barrier()
            if getattr(arg.setup, "resume_from_artifact", None) is not None:
                metadata = t.load_from_artifact(arg.setup.resume_from_artifact)
                if metadata:
                    logging.getLogger(__name__).info(f"Loaded from artifact: {metadata}")
                else:
                    logging.getLogger(__name__).warning("Failed to load from artifact, skipping...")
            elif getattr(arg.setup, "resume_from_ckpt", None) is not None:
                t.load_ckpt(path=arg.setup.resume_from_ckpt)
            else:
                t.load_ckpt()
            t.test()

    if getattr(arg.setup, "rank", 0) == 0:
        if os.path.exists(arg.path["database_path"]):
            database = pd.read_csv(arg.path["database_path"])
        else:
            database = pd.DataFrame(columns=arg.datarow.keys())
        database.loc[len(database)] = arg.datarow
        database.to_csv(arg.path["database_path"], index=False)

    # Clean up DDP process group to release NCCL/GPU resources
    if dist.is_initialized():
        dist.destroy_process_group()

    return t


def run_arg_file_popen_handle(arg_file):
    command = f"python main.py -c {arg_file}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    if process.returncode == 0:
        print(f"Job {arg_file}: {out.decode('utf-8').strip()}")
    else:
        print(f"Job {arg_file} error: {err.decode('utf-8').strip()}")


def run_arg_files(arg_files, is_debug, num_works_per_device=3, overrides=None):  # noqa: PLR0912
    if len(arg_files) == 1:
        run_arg(FileParser(arg_files[0], overrides=overrides).parse_args())
    elif is_debug:
        for arg_file in arg_files:
            print("\n")
            print(arg_file, end="\n\n\n")
            run_arg(FileParser(arg_file, overrides=overrides).parse_args())
    elif platform.system() == "Windows":
        processes: list[Process] = []
        for arg_file in arg_files:
            arg = FileParser(arg_file, overrides=overrides).parse_args()
            p = Process(target=run_arg, args=(arg,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    elif platform.system() == "Linux":
        num_devices = torch.cuda.device_count()
        process_map: dict[str, list[Process]] = {"cpu": []}
        for i in range(num_devices):
            process_map[f"cuda:{i}"] = []
        for arg_file in arg_files:
            arg = FileParser(arg_file, overrides=overrides).parse_args()
            p = Process(target=run_arg_file_popen_handle, args=(arg_file,))
            if arg.device.startswith("cuda"):
                device_id = int(arg.device[-1])
                process_map[f"cuda:{device_id}"].append(p)
            else:
                process_map["cpu"].append(p)

        max_jobs = max([len(v) for k, v in process_map.items()])
        max_runs = (max_jobs + num_works_per_device - 1) // num_works_per_device
        for i in range(max_runs):
            for v in process_map.values():
                for p in v[i * num_works_per_device : (i + 1) * num_works_per_device]:
                    p.start()
            for v in process_map.values():
                for p in v[i * num_works_per_device : (i + 1) * num_works_per_device]:
                    p.join()
    else:
        raise NotImplementedError(f"Platform {platform.system()} not supported")


if __name__ == "__main__":
    config = parse_files()
    if getattr(config, "wandb_overrides", None):
        print("Applied W&B sweep overrides:")
        for k, v in config.wandb_overrides.items():
            print(f"  {k} = {v}")
    run_arg_files(config.arg_files, config.debug, config.num_works_per_device, getattr(config, "wandb_overrides", None))
    print("Finished")
