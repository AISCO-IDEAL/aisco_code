"""
Config loading utilities with a sweep-aware registry.

When a wandb sweep is running, sweep parameters are named with the config
filename (without ``.yaml``) as prefix, e.g.::

    losses_v0_config.general_info.initial_lambda   → targets losses_v0_config.yaml
    training_v0_config.training_params.learning_rate → targets training_v0_config.yaml
    solar_loss_config.SolarLossConfigV3.method       → targets solar_loss_config.yaml

The registry is populated once at startup via :func:`register_all_configs` and
patched with sweep overrides via :func:`apply_sweep_overrides`.  After that,
every subsequent :func:`load_config` call returns the patched version instead
of re-reading the YAML from disk.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import socket
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Module-level config registry
# ---------------------------------------------------------------------------
_config_registry: dict[str, dict] = {}
_registry_initialised: bool = False
_registry_alias_index: dict[str, set[str]] = {}


def _cfg_key(cfg_name: str) -> str:
    """Normalise a config filename to a registry key (drop ``.yaml``)."""
    return cfg_name.replace(".yaml", "")


def _register_key_alias(alias: str, canonical_key: str) -> None:
    """Register one alias that resolves to a canonical config key."""
    _registry_alias_index.setdefault(alias, set()).add(canonical_key)


def _rebuild_alias_index() -> None:
    """Rebuild alias-to-config mapping from the current registry.

    Canonical keys remain the source of truth. Additional aliases allow sweep
    keys to reference subfolder configs robustly:
    - canonical: ``solar_configs/solar_loss_config``
    - dotted-path alias: ``solar_configs.solar_loss_config``
    - basename alias: ``solar_loss_config`` (legacy compatibility)
    """
    _registry_alias_index.clear()
    for canonical_key in _config_registry:
        _register_key_alias(canonical_key, canonical_key)
        _register_key_alias(canonical_key.replace("/", "."), canonical_key)
        _register_key_alias(Path(canonical_key).name, canonical_key)


def _resolve_sweep_target(key: str) -> tuple[dict | None, list[str]]:
    """Resolve a sweep key to a target config dict and remaining nested path.

    The key format is ``<config-id>.<nested.path>`` where ``<config-id>`` can
    be canonical (path-aware), dotted-path alias, or basename alias.

    If no config-id matches, this preserves old behaviour and falls back to
    applying on ``general_config`` using the full key path.
    """
    if "." not in key:
        return _config_registry.get("general_config"), [key]

    candidates: list[tuple[int, int, str, str, list[str]]] = []
    for alias, canonical_keys in _registry_alias_index.items():
        prefix = f"{alias}."
        if not key.startswith(prefix):
            continue

        remaining = key[len(prefix) :].split(".")
        if not remaining:
            continue

        # Prefer longer alias matches first. For ties, prefer canonical over
        # dotted-path alias over basename alias.
        alias_kind_priority = 0
        canonical_key = next(iter(sorted(canonical_keys)))
        if alias == canonical_key:
            alias_kind_priority = 2
        elif alias == canonical_key.replace("/", "."):
            alias_kind_priority = 1

        for canonical in sorted(canonical_keys):
            candidates.append((len(alias), alias_kind_priority, canonical, alias, remaining))  # noqa: PERF401, does not really matter to us, one time thing

    if candidates:
        best = max(candidates, key=lambda item: (item[0], item[1]))
        target_cfg = _config_registry.get(best[2])
        if target_cfg is not None:
            return target_cfg, best[4]

    # Backward-compatible fallback for keys that do not name a config.
    return _config_registry.get("general_config"), key.split(".")


def _load_yaml(v: str, cfg_name: str) -> dict:
    """
    Load a YAML configuration file from disk (no registry, no wandb side-effects) by giving the name/path of the config file, located in the v{x}/configs/ folder.
    """
    config_path = Path.home() / "AISCO_code" / "subprojects" / "wp4_aeroshell_optimizer" / v / "configs" / cfg_name
    try:
        with config_path.open() as f:
            config = yaml.safe_load(f)

        if cfg_name == "general_config.yaml" and isinstance(config, dict) and "general_info" in config:
            is_wsl = "WSL_DISTRO_NAME" in os.environ
            hostname = socket.gethostname()
            is_euler_node = hostname.startswith("eu-")
            is_slurm_job = "SLURM_JOB_ID" in os.environ

            if is_wsl:
                # Definitely local (WSL)
                config["general_info"]["run_locally"] = "True"
            elif is_euler_node or is_slurm_job:
                # Definitely cluster (HPC)
                config["general_info"]["run_locally"] = "False"
            else:
                # Default to Euler if uncertain
                config["general_info"]["run_locally"] = "False"

        return config  # noqa: TRY300
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found at {config_path}") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file at {config_path}: {e}") from e


# ---------------------------------------------------------------------------
# Registry helpers — called once from main before the training loop
# ---------------------------------------------------------------------------


def register_all_configs(general_config: dict, *, v: str = "v1") -> None:
    """Pre-load every sub-config referenced in *general_config* into the
    registry so that :func:`apply_sweep_overrides` can patch them and later
    :func:`load_config` calls return the patched version.

    Also loads standalone configs (``solar_loss_config``, ``logging``) that are
    used throughout the codebase but not listed in ``active_configs``.
    """
    global _registry_initialised  # noqa: PLW0603

    # Register the general config itself
    _config_registry[_cfg_key("general_config.yaml")] = general_config

    # Register every config referenced in active_configs
    for cfg_name in general_config.get("active_configs", {}).values():
        if cfg_name and _cfg_key(cfg_name) not in _config_registry:
            with contextlib.suppress(FileNotFoundError):  # e.g. sweep definition YAMLs that were removed — skip
                _config_registry[_cfg_key(cfg_name)] = _load_yaml(v, cfg_name)

    # Register standalone configs used elsewhere in the codebase
    _standalone = ["solar_configs/solar_loss_config.yaml", "logging.yaml", "solar_configs/australia.yaml"]
    for cfg_name in _standalone:
        key = _cfg_key(cfg_name)
        if key not in _config_registry:
            with contextlib.suppress(FileNotFoundError):  # optional config, skip if missing
                _config_registry[key] = _load_yaml(v, cfg_name)

    _rebuild_alias_index()

    _registry_initialised = True


def apply_sweep_overrides(sweep_params: dict) -> None:
    """Apply wandb sweep parameters to the registered configs.

    Each key is expected to be of the form
        ``<config_id>.<nested.path.to.value>`` where ``<config_id>`` can be:
        - canonical relative config path without ``.yaml``
            (e.g. ``solar_configs/solar_loss_config``)
        - dotted-path alias for folder separators
            (e.g. ``solar_configs.solar_loss_config``)
        - basename alias (legacy compatibility)
            (e.g. ``solar_loss_config``)

    Keys whose first segment does **not** match any registered config are
    applied to ``general_config`` as a fallback (backward-compatible with old
    flat keys)
    Args:
        sweep_params (dict): A dictionary of sweep parameters.
    Returns:
        None, the config values are updated in place
    """
    if not _registry_initialised:
        raise RuntimeError(
            "apply_sweep_overrides() called before register_all_configs(). Call register_all_configs(general_config) first."
        )

    for key, value in sweep_params.items():
        # Skip internal wandb keys and SubCFG_ keys logged by older code
        if key.startswith(("_wandb", "SubCFG_")):
            continue

        target_cfg, remaining = _resolve_sweep_target(key)

        if target_cfg is None:
            continue

        # Drill into nested dicts, creating intermediate dicts as needed
        d = target_cfg
        if not remaining:
            continue
        for segment in remaining[:-1]:
            if segment not in d or not isinstance(d[segment], dict):
                d[segment] = {}
            d = d[segment]
        d[remaining[-1]] = value


def get_master_config() -> dict:
    """Return a combined snapshot of all registered configs for wandb logging.

    The result is a deep copy so mutating it has no effect on the registry.
    """
    return copy.deepcopy(_config_registry)


def get_registered_config(cfg_name: str) -> dict | None:
    """Return a registered config by name (without ``.yaml``), or ``None``."""
    return _config_registry.get(_cfg_key(cfg_name))


# ---------------------------------------------------------------------------
# Public API — load_config (backward-compatible)
# ---------------------------------------------------------------------------


def load_config(v: str = "v1", cfg_name: str = "general_config.yaml") -> dict:
    """Load a config by name.

    If the registry has been initialised (i.e. we are inside a sweep-aware
    run), return the registered (potentially sweep-patched) version.
    Otherwise fall back to reading the YAML from disk.

    Sub-configs are still logged to ``wandb.config`` as ``SubCFG_<name>`` for
    visibility on the dashboard.
    """
    key = _cfg_key(cfg_name)

    # If the registry is populated, prefer the (possibly patched) version
    config = _config_registry[key] if _registry_initialised and key in _config_registry else _load_yaml(v, cfg_name)

    return config


def print_config(config: dict) -> None:
    print("\n=== Loaded Configuration ===")
    print("Config Contents:")
    print(json.dumps(config, indent=4))
    print("============================\n")
