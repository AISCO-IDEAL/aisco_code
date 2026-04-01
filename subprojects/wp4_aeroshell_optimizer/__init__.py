"""wp4_aeroshell_optimizer — versioned public API.

Change ACTIVE_VERSION below to switch all callers between v1 and v2.
"""

import importlib
from typing import TYPE_CHECKING, Any

ACTIVE_VERSION = "v2"  # change to "v2" to switch globally

# For static type checking, import the chosen version so mypy
# sees a single consistent API. Use TYPE_CHECKING so only a
# single version is seen by the type checker.
if TYPE_CHECKING:
    from .v2 import *  # noqa: F403
    from .v2 import CONFIGS_DIR, SLURM_SCRIPTS_DIR, __all__
else:
    # Runtime: import only the selected version and forward attribute access.
    _mod = importlib.import_module(f"{__name__}.{ACTIVE_VERSION}")

    # Expose commonly used constants eagerly for tooling/import-time access.
    CONFIGS_DIR = getattr(_mod, "CONFIGS_DIR", None)
    SLURM_SCRIPTS_DIR = getattr(_mod, "SLURM_SCRIPTS_DIR", None)

    def __getattr__(name: str) -> Any:  # PEP 562
        return getattr(_mod, name)

    def __dir__() -> list[str]:  # PEP 562
        return list(getattr(_mod, "__all__", []))


VERSION = ACTIVE_VERSION
