from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


def resolve_bb_assembly_id(config: dict[str, Any], random_id_fn: Callable[[], int]) -> int:
    env_assembly_id = os.getenv("AISCO_ASSEMBLY_ID", "").strip()
    if env_assembly_id:
        return int(env_assembly_id)

    if str(config.get("random_assembly_id", "False")) == "True":
        return int(random_id_fn())

    configured_assembly_id = config.get("assembly_id")
    if configured_assembly_id is not None:
        return int(configured_assembly_id)

    raise ValueError("Unable to resolve assembly_id from AISCO_ASSEMBLY_ID, random_assembly_id, or assembly_id.")
