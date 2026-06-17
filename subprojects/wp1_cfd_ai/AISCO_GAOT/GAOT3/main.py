"""GAOT3 main is called as python main.py -c config.yaml and passes config dicts to TrainingOrchestrator."""

import os
import platform
import subprocess
import sys
from multiprocessing import Process
from pathlib import Path

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.trainer.handlers.config_handler import ConfigHandler
from src.trainer.training_orchestrator import TrainingOrchestrator

main_config: dict = {}


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge *overrides* into *base* in place."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _load_user_overrides() -> dict:
    """Return ~/AISCO_code/user_overrides.yaml[wp1] when enabled, else {}."""
    path = Path.home() / "AISCO_code" / "user_overrides.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not data.get("enabled", True):
        return {}
    return data.get("wp1", {})


def _expand_path_overrides(config: dict) -> None:
    """Expand ~ and substitute _NAME_/_RUN_ID_ placeholders in config['path'] strings.

    Must be called after user overrides are deep-merged so that tilde paths and
    template placeholders written in user_overrides.yaml are resolved against the
    actual run name and job ID that are already present in config['setup'].

    Args:
        config: The fully merged config dict, mutated in place.
    """
    path_section = config.get("path")
    if not isinstance(path_section, dict):
        return
    run_name: str = str(config.get("setup", {}).get("name", ""))
    run_id: str = str(config.get("setup", {}).get("id", ""))
    for key, raw in path_section.items():
        if not isinstance(raw, str):
            continue
        expanded = os.path.expanduser(raw)
        expanded = expanded.replace("_NAME_", run_name)
        expanded = expanded.replace("_RUN_ID_", run_id)
        path_section[key] = expanded


def run_config_file(config_file: str, overrides: dict | None = None):
    main_config.clear()
    main_config.update(ConfigHandler().load_config(config_file, overrides=overrides))
    _deep_merge(main_config, _load_user_overrides())
    _expand_path_overrides(main_config)
    return TrainingOrchestrator(main_config).run()


def run_config_file_popen_handle(config_file: str) -> None:
    command = [sys.executable, os.path.abspath(__file__), "-c", config_file]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    if process.returncode == 0:
        print(f"Job {config_file}: {out.decode('utf-8').strip()}")
    else:
        print(f"Job {config_file} error: {err.decode('utf-8').strip()}")


def run_config_files(
    config_files: list[str], is_debug: bool, num_works_per_device: int = 3, overrides: dict | None = None
) -> None:
    if len(config_files) == 1:
        run_config_file(config_files[0], overrides=overrides)
    elif is_debug:
        for config_file in config_files:
            print("\n")
            print(config_file, end="\n\n\n")
            run_config_file(config_file, overrides=overrides)
    elif platform.system() == "Windows":
        processes: list[Process] = []
        for config_file in config_files:
            process = Process(target=run_config_file, args=(config_file, overrides))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
    elif platform.system() == "Linux":
        _run_linux_config_files(config_files, num_works_per_device, overrides)
    else:
        raise NotImplementedError(f"Platform {platform.system()} not supported")


def _run_linux_config_files(config_files: list[str], num_works_per_device: int, overrides: dict | None) -> None:
    process_map = _build_linux_process_map(config_files, overrides)
    max_jobs = max(len(value) for value in process_map.values())
    max_runs = (max_jobs + num_works_per_device - 1) // num_works_per_device
    for index in range(max_runs):
        for processes in process_map.values():
            for process in processes[index * num_works_per_device : (index + 1) * num_works_per_device]:
                process.start()
        for processes in process_map.values():
            for process in processes[index * num_works_per_device : (index + 1) * num_works_per_device]:
                process.join()


def _build_linux_process_map(config_files: list[str], overrides: dict | None) -> dict[str, list[Process]]:
    process_map: dict[str, list[Process]] = {"cpu": []}
    for index in range(torch.cuda.device_count()):
        process_map[f"cuda:{index}"] = []
    for config_file in config_files:
        config = ConfigHandler().load_config(config_file, overrides=overrides)
        process = Process(target=run_config_file_popen_handle, args=(config_file,))
        device = config["setup"]["device"]
        process_key = f"cuda:{int(device[-1])}" if device.startswith("cuda") else "cpu"
        process_map[process_key].append(process)
    return process_map


if __name__ == "__main__":
    config = ConfigHandler().parse_command_line()
    if getattr(config, "wandb_overrides", None):
        print("Applied W&B sweep overrides:")
        for key, value in config.wandb_overrides.items():
            print(f"  {key} = {value}")
    run_config_files(config.arg_files, config.debug, config.num_works_per_device, getattr(config, "wandb_overrides", None))
    print("Finished")
