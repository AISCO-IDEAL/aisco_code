"""Global mesh cache shared across MeshSDFExtractor instances."""

from __future__ import annotations

import torch

_MESH_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_MESH_CACHE_MAX_ENTRIES = 16


def clear_mesh_cache() -> None:
    """Manually clear the global mesh cache.

    Rarely needed: the cache auto-invalidates on parameter updates via the
    `_version` counter. Use this when deliberately rerunning extraction with
    the exact same parameters (e.g. in tests).
    """
    _MESH_CACHE.clear()
