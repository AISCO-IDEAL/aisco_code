"""Utility functions for assembly ID resolution in loss functions."""

from __future__ import annotations

from typing import Any

import numpy as np

from subprojects.wp2_assembler.version4.components.nbz import max_assembly_id


def resolve_assembly_id(config: dict[str, Any]) -> int:
    """
    Resolve assembly ID from configuration with support for fixed and random IDs.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary containing:
        - "random_assembly_id": "True" or "False" (default: "False")
        - "assembly_id": integer for fixed ID (default: 1)

    Returns
    -------
    int
        The resolved assembly ID.

    Raises
    ------
    ValueError
        If random_assembly_id has an invalid value (not "True" or "False")
        or if generated random ID is out of valid range.

    Examples
    --------
    >>> # Fixed assembly ID
    >>> config = {"random_assembly_id": "False", "assembly_id": 5}
    >>> resolve_assembly_id(config)
    5

    >>> # Random assembly ID
    >>> config = {"random_assembly_id": "True"}
    >>> aid = resolve_assembly_id(config)
    >>> 0 <= aid <= max_assembly_id()
    True
    """
    random_assembly_id_str = config.get("random_assembly_id", "False")

    # Fixed assembly ID
    if random_assembly_id_str == "False":
        return int(config.get("assembly_id", 1))

    # Random assembly ID
    elif random_assembly_id_str == "True":
        max_id = max_assembly_id()
        assembly_id = int(np.random.default_rng().integers(0, max_id + 1))
        if not 0 <= assembly_id <= max_id:
            raise ValueError(f"assembly_id must be in range 0..{max_id}, got {assembly_id}")
        return assembly_id

    # Invalid value
    else:
        raise ValueError(f"Invalid value for random_assembly_id: {random_assembly_id_str} (expected 'True' or 'False')")
