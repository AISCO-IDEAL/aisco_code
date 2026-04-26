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
import re
from pathlib import Path
from warnings import warn

import yaml

from subprojects.wp4_aeroshell_optimizer.v2.src.utils.device_detection import get_config_path, get_run_locally_value

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

    Environment-aware path resolution is delegated to device_detection module.
    """
    config_path = Path.home() / "AISCO_code" / "subprojects" / "wp4_aeroshell_optimizer" / v / "configs" / cfg_name
    if Path.home() == Path("/home/aisco"):  # Checks if running on the Best and corrects Path if necessary
        config_path = (
            Path.home() / "Documents" / "AISCO_code" / "subprojects" / "wp4_aeroshell_optimizer" / v / "configs" / cfg_name
        )
    try:
        with config_path.open() as f:
            config = yaml.safe_load(f)

        if cfg_name == "general_config.yaml" and isinstance(config, dict) and "general_info" in config:
            config["general_info"]["run_locally"] = get_run_locally_value()

        return config  # noqa: TRY300
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found at {config_path}") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file at {config_path}: {e}") from e


# ---------------------------------------------------------------------------
# Registry helpers — called once from main before the training loop
# ---------------------------------------------------------------------------


def register_all_configs(general_config: dict, *, v: str = "v2") -> None:
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
    _standalone = [
        "solar_configs/solar_loss_config.yaml",
        "solar_configs/raytracing_config.yaml",
        "logging.yaml",
        "solar_configs/australia.yaml",
        "debug.yaml",
    ]
    for cfg_name in _standalone:
        key = _cfg_key(cfg_name)
        if key not in _config_registry:
            with contextlib.suppress(FileNotFoundError):  # optional config, skip if missing
                _config_registry[key] = _load_yaml(v, cfg_name)

    _rebuild_alias_index()

    _registry_initialised = True


def apply_sweep_overrides(sweep_params: dict) -> None:  # noqa: PLR0915
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
    flat keys).

    Grouped sweep keys are also supported with this syntax:
        ``group<n>.<path>.<param1>...<param_n>``

    where:
    - ``n`` is the group size,
    - ``path`` points to the shared config object,
    - trailing ``param`` names are mapped left-to-right to row values.

    Example:
        ``group2.solar_configs/raytracing_config.RaytracingConfigV2.sdf.step_size_factor.num_steps``
        with value ``[0.03, 64]`` maps to:
        - ``...sdf.step_size_factor = 0.03``
        - ``...sdf.num_steps = 64``

    Grouped rows with invalid shape are skipped with a warning.
    Multiple grouped keys in one run are forbidden to prevent accidental
    combinatorial coupling.
    Args:
        sweep_params (dict): A dictionary of sweep parameters.
    Returns:
        None, the config values are updated in place
    """
    if not _registry_initialised:
        raise RuntimeError(
            "apply_sweep_overrides() called before register_all_configs(). Call register_all_configs(general_config) first."
        )

    def _coerce_like_existing(raw_value: object, existing_value: object) -> object:  # noqa: PLR0911
        """Cast grouped values to match the existing config leaf type when possible."""
        # Keep grouped sweeps type-stable by mirroring the target leaf type.
        if isinstance(existing_value, bool):
            if isinstance(raw_value, str):
                lowered = raw_value.strip().lower()
                if lowered in {"true", "1", "yes", "on"}:
                    return True
                if lowered in {"false", "0", "no", "off"}:
                    return False
                return raw_value
            if isinstance(raw_value, (int, float)):
                return bool(raw_value)
            return raw_value

        if isinstance(existing_value, int) and not isinstance(existing_value, bool):
            if isinstance(raw_value, (str, int, float, bool)):
                try:
                    return int(raw_value)
                except (TypeError, ValueError):
                    return raw_value
            return raw_value

        if isinstance(existing_value, float):
            if isinstance(raw_value, (str, int, float, bool)):
                try:
                    return float(raw_value)
                except (TypeError, ValueError):
                    return raw_value
            return raw_value

        if isinstance(existing_value, str):
            return str(raw_value)

        return raw_value

    def _apply_single_override(full_key: str, raw_value: object, *, infer_type: bool = False) -> None:
        target_cfg, remaining = _resolve_sweep_target(full_key)

        if target_cfg is None:
            warn(f"Skipping override with unresolved target: {full_key}", stacklevel=2)
            return

        if not remaining:
            warn(f"Skipping override with empty path: {full_key}", stacklevel=2)
            return

        d = target_cfg
        # Create intermediate dicts so new nested keys can be injected safely.
        for segment in remaining[:-1]:
            if segment not in d or not isinstance(d[segment], dict):
                d[segment] = {}
            d = d[segment]

        leaf = remaining[-1]
        value_to_set = raw_value
        if infer_type and leaf in d:
            value_to_set = _coerce_like_existing(raw_value, d[leaf])

        d[leaf] = value_to_set

    grouped_key_pattern = re.compile(r"^group(\d+)\.(.+)$")
    # Guardrail: forbid multiple grouped dimensions in one run to prevent
    # accidental Cartesian coupling between grouped keys.
    grouped_keys = [key for key in sweep_params if isinstance(key, str) and grouped_key_pattern.fullmatch(key)]

    if len(grouped_keys) > 1:
        raise RuntimeError(
            f"Multiple grouped sweep keys are not supported in one run. Found: {grouped_keys}. Keep exactly one grouped key."
        )

    for key, value in sweep_params.items():
        # Skip internal wandb keys and SubCFG_ keys logged by older code
        if key.startswith(("_wandb", "SubCFG_")):
            continue

        grouped_match = grouped_key_pattern.fullmatch(key)
        if grouped_match:
            group_size = int(grouped_match.group(1))
            grouped_path = grouped_match.group(2)

            if group_size <= 0:
                warn(f"Skipping grouped key with non-positive size: {key}", stacklevel=2)
                continue

            path_parts = grouped_path.split(".")
            if len(path_parts) <= group_size:
                warn(
                    f"Skipping grouped key because it does not have enough path segments for group size {group_size}: {key}",
                    stacklevel=2,
                )
                continue

            if not isinstance(value, (list, tuple)):
                warn(
                    f"Skipping grouped key {key}: expected list/tuple row, got {type(value).__name__}",
                    stacklevel=2,
                )
                continue

            if len(value) != group_size:
                warn(
                    f"Skipping grouped row for {key}: expected length {group_size}, got {len(value)}",
                    stacklevel=2,
                )
                continue

            # group<n> consumes the last n dotted segments as parameter names.
            # Split grouped key into shared config path + grouped parameter suffix.
            base_path = ".".join(path_parts[:-group_size])
            param_names = path_parts[-group_size:]

            # W&B already selected one row for this run; we now fan it out into
            # individual flat overrides that reuse the existing config resolver.
            # Left-to-right mapping: param_names[i] gets value[i].
            for param_name, param_value in zip(param_names, value, strict=True):
                target_key = f"{base_path}.{param_name}"
                _apply_single_override(target_key, param_value, infer_type=True)
            continue

        _apply_single_override(key, value, infer_type=False)


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


def load_config(v: str = "v2", cfg_name: str = "general_config.yaml") -> dict:
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
