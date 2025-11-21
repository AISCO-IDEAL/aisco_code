
from pathlib import Path
import yaml
def load_config() -> dict:
    config_path = (
        Path.home()
        / "AISCO_code"
        / "subprojects"
        / "wp4_aeroshell_optimizer"
        / "v1"
        / "config"
        / "wp4_config.yaml"
    )
    ####################### DEBUG #############################
    print(f"Looking for config at: {config_path}")
    ###########################################################
    try:
        with config_path.open() as f:
            config = yaml.safe_load(f)
        print(f"✓ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"✗ Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"✗ Error parsing configuration file: {e}")
        raise


def print_config(config: dict):
    print("\n=== Loaded Configuration ===")
    print("Config Contents:")
    print(json.dumps(config, indent=4))
    print("============================\n")