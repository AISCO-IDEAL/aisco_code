"""CLI argument parsing and config environment setup for main_v2.py."""

import argparse
import os
import sys
from pathlib import Path


def _parse_cfg_args(argv: list[str] | None = None) -> tuple[Path | None, bool]:
    """Parse --cfg-folder and --uses-gui from argv.

    Unknown arguments are silently ignored so that existing callers that pass
    --save_model, --problem_id, etc. continue to work without change.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:] when None.

    Returns:
        Tuple of (cfg_folder path or None, uses_gui flag).
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cfg-folder", dest="cfg_folder", default=None)
    parser.add_argument("--uses-gui", dest="uses_gui", action="store_true", default=False)
    args, _ = parser.parse_known_args(argv if argv is not None else sys.argv[1:])
    cfg_folder = Path(args.cfg_folder) if args.cfg_folder is not None else None
    return cfg_folder, args.uses_gui


def _configure_cfg_env(cfg_folder: Path | None) -> str:
    """Expose cfg_folder to config_utils via AISCO_CFG_FOLDER env var.

    Must be called before any load_config / register_all_configs call so that
    _load_yaml can see the override path.

    Args:
        cfg_folder: Path to the session folder, or None if no override is active.

    Returns:
        config_source string: 'session_folder' or 'default'.
    """
    if cfg_folder is None:
        return "default"

    if not cfg_folder.exists():
        print(
            f"[WARN] run_start: --cfg-folder path does not exist: {cfg_folder}. Falling back to default configs.",
            file=sys.stderr,
        )
        return "default"

    os.environ["AISCO_CFG_FOLDER"] = str(cfg_folder)
    return "session_folder"
