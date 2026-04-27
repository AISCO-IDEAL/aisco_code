"""GAOT3 main is called as python main.py -c config.yaml and passes config dicts to TrainingOrchestrator."""

import os
import platform
import subprocess
import sys
from multiprocessing import Process

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.trainer.config_handler import ConfigHandler

main_config: dict = {}


def run_config_file(config_file: str, overrides: dict | None = None):
    from src.trainer.training_orchestrator import TrainingOrchestrator

    main_config.clear()
    main_config.update(ConfigHandler().load_config(config_file, overrides=overrides))
    return TrainingOrchestrator(main_config).run()


def run_config_file_popen_handle(config_file: str) -> None:
    command = [sys.executable, os.path.abspath(__file__), "-c", config_file]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    if process.returncode == 0:
        print(f"Job {config_file}: {out.decode('utf-8').strip()}")
    else:
        print(f"Job {config_file} error: {err.decode('utf-8').strip()}")


def run_config_files(config_files: list[str], is_debug: bool, num_works_per_device: int = 3, overrides: dict | None = None) -> None:
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
        num_devices = torch.cuda.device_count()
        process_map: dict[str, list[Process]] = {"cpu": []}
        for index in range(num_devices):
            process_map[f"cuda:{index}"] = []
        for config_file in config_files:
            config = ConfigHandler().load_config(config_file, overrides=overrides)
            process = Process(target=run_config_file_popen_handle, args=(config_file,))
            if config["setup"]["device"].startswith("cuda"):
                device_id = int(config["setup"]["device"][-1])
                process_map[f"cuda:{device_id}"].append(process)
            else:
                process_map["cpu"].append(process)

        max_jobs = max([len(value) for value in process_map.values()])
        max_runs = (max_jobs + num_works_per_device - 1) // num_works_per_device
        for index in range(max_runs):
            for processes in process_map.values():
                for process in processes[index * num_works_per_device : (index + 1) * num_works_per_device]:
                    process.start()
            for processes in process_map.values():
                for process in processes[index * num_works_per_device : (index + 1) * num_works_per_device]:
                    process.join()
    else:
        raise NotImplementedError(f"Platform {platform.system()} not supported")


if __name__ == "__main__":
    config = ConfigHandler().parse_command_line()
    if getattr(config, "wandb_overrides", None):
        print("Applied W&B sweep overrides:")
        for key, value in config.wandb_overrides.items():
            print(f"  {key} = {value}")
    run_config_files(config.arg_files, config.debug, config.num_works_per_device, getattr(config, "wandb_overrides", None))
    print("Finished")
