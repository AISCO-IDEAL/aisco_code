from pathlib import Path


def fetch_wp4_configs(ROOT_DIR: str, exclude_subfolders: tuple[str, ...] = ()) -> list[str]:
    """
    Gets fed the root directory (absolute path of AISCO_code).
    Returns a list of paths (as str) to all .yaml configs under wp4/v2/configs/, including subdirectories.

    Args:
        ROOT_DIR: absolute path to the AISCO_code repo root
        exclude_subfolders: subfolder names (relative to wp4/v2/configs/) to skip recursively (e.g. ("sweeps",))
    """
    cfg_path = Path(ROOT_DIR) / "subprojects" / "wp4_aeroshell_optimizer" / "v2" / "configs"
    config_files = list(cfg_path.rglob("*.yaml"))
    if exclude_subfolders:
        excluded = {(cfg_path / sub).resolve() for sub in exclude_subfolders}
        config_files = [
            cfg
            for cfg in config_files
            if not any(str(cfg.resolve()).startswith(str(ex) + "/") or cfg.resolve() == ex for ex in excluded)
        ]
    return [str(cfg) for cfg in config_files]
