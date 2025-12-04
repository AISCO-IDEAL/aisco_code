import json
from pathlib import Path

import yaml


def load_general_config(v="v1", cfg_name="general_config.yaml") -> dict:
    config_path = Path.home() / "AISCO_code" / "subprojects" / "wp4_aeroshell_optimizer" / v / "configs" / cfg_name

    try:
        with config_path.open() as f:
            config = yaml.safe_load(f)
            if config["general_info"].get("config_debugging") == "True":
                print(f"✓ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file at {config_path}: {e}")


def load_specific_config(v="v1", cfg_name="losses_v0_config.yaml") -> dict:
    config_path = Path.home() / "AISCO_code" / "subprojects" / "wp4_aeroshell_optimizer" / v / "configs" / cfg_name

    try:
        with config_path.open() as f:
            config = yaml.safe_load(f)
            if config["general_info"].get("config_debugging") == "True":
                print(f"✓ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file at {config_path}: {e}")


def print_config(config: dict):
    print("\n=== Loaded Configuration ===")
    print("Config Contents:")
    print(json.dumps(config, indent=4))
    print("============================\n")
