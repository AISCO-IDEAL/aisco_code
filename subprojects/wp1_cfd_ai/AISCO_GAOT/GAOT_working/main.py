import argparse
import logging
import os
import platform
import subprocess
import time
from multiprocessing import Process

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from omegaconf import OmegaConf

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


class FileParser:
    def __init__(self, filename):
        if filename.endswith((".toml", ".json", ".yaml")):
            with open(filename) as f:
                self.kwargs = OmegaConf.load(f)
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
    args = parser.parse_args()
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


def run_arg(arg):  # noqa: PLR0912
    arg = prepare_arg(arg)
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    log_file = os.environ.get("LOG_FILE", None) or arg.path.get("log_path", None)
    setup_logging(level=log_level, log_file=log_file)

    Trainer = {
        "static3d": StaticTrainer3D,
    }[arg.setup.trainer_name]
    t = Trainer(arg)

    if arg.setup.train:
        # Load model checkpoint with priority: artifact > resume_from_ckpt > ckpt
        if getattr(arg.setup, "resume_from_artifact", None) is not None:
            metadata = t.load_from_artifact(arg.setup.resume_from_artifact)
            if metadata:
                logging.getLogger(__name__).info(f"Loaded from artifact: {metadata}")
            else:
                logging.getLogger(__name__).warning("Failed to load from artifact, skipping...")
        elif getattr(arg.setup, "resume_from_ckpt", None) is not None:
            t.load_ckpt(path=arg.setup.resume_from_ckpt)
        elif arg.setup.ckpt:
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

    return t


def run_arg_file_popen_handle(arg_file):
    command = f"python main.py -c {arg_file}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    if process.returncode == 0:
        print(f"Job {arg_file}: {out.decode('utf-8').strip()}")
    else:
        print(f"Job {arg_file} error: {err.decode('utf-8').strip()}")


def run_arg_files(arg_files, is_debug, num_works_per_device=3):  # noqa: PLR0912
    if len(arg_files) == 1:
        run_arg(FileParser(arg_files[0]).parse_args())
    elif is_debug:
        for arg_file in arg_files:
            print("\n")
            print(arg_file, end="\n\n\n")
            run_arg(FileParser(arg_file).parse_args())
    elif platform.system() == "Windows":
        processes: list[Process] = []
        for arg_file in arg_files:
            arg = FileParser(arg_file).parse_args()
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
            arg = FileParser(arg_file).parse_args()
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
    run_arg_files(config.arg_files, config.debug, config.num_works_per_device)
    print("Finished")
