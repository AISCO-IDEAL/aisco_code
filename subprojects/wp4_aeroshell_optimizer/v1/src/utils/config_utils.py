import json
from pathlib import Path

import yaml

import wandb


def load_config(v="v1", cfg_name="general_config.yaml") -> dict:
    """
    Load a YAML configuration file by giving hte name/path of the config file, located in the v{x}/configs/ folder.
    """
    config_path = Path.home() / "AISCO_code" / "subprojects" / "wp4_aeroshell_optimizer" / v / "configs" / cfg_name

    try:
        with config_path.open() as f:
            config = yaml.safe_load(f)
            if wandb.run is not None and cfg_name != "general_config.yaml":
                # general config should be logged already as the main part on wandb
                name = cfg_name.replace(".yaml", "")
                key = f"SubCFG_{name}"
                if key not in wandb.config:
                    wandb.config.update({key: config})
                # Add other configs as subkeys in the main config for better visibility in wandb
        return config  # noqa: TRY300
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found at {config_path}") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file at {config_path}: {e}") from e


def print_config(config: dict):
    print("\n=== Loaded Configuration ===")
    print("Config Contents:")
    print(json.dumps(config, indent=4))
    print("============================\n")
