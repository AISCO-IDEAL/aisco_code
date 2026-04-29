from pathlib import Path


def fetch_wp4_configs(ROOT_DIR: str) -> list[str]:
    """
    Gets fed the root directory (absolute path of AISCO_code)
    Returns a list of paths (as str) to all the configs in the config storage directory (inluding subdirectories)

    """
    cfg_path = Path(ROOT_DIR) / "subprojects" / "wp4_aeroshell_optimizer" / "v2" / "configs"
    # recursively get all .yaml files in cfg_path and subdirectories
    config_files = list(cfg_path.rglob("*.yaml"))
    # convert to string paths
    config_files_str = [str(cfg) for cfg in config_files]
    return config_files_str
