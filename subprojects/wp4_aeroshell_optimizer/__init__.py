"""wp4_aeroshell_optimizer — versioned public API.

Change ACTIVE_VERSION below to switch all callers between v1 and v2.
"""

import importlib
from typing import TYPE_CHECKING, Any

ACTIVE_VERSION = "v1"  # change to "v2" to switch globally

# For static type checking, import the chosen version so mypy
# sees a single consistent API. Use TYPE_CHECKING so only a
# single version is seen by the type checker.
if TYPE_CHECKING:
    from .v2 import *  # noqa: F403
    from .v2 import CONFIGS_DIR, SLURM_SCRIPTS_DIR, __all__
else:
    # Runtime: lazy-load only when attributes are requested.
    _mod: Any | None = None

    def _get_mod() -> Any:
        global _mod  # noqa: PLW0603
        if _mod is None:
            _mod = importlib.import_module(f"{__name__}.{ACTIVE_VERSION}")
        return _mod

    def __getattr__(name: str) -> Any:  # PEP 562
        return getattr(_get_mod(), name)

    def __dir__() -> list[str]:  # PEP 562
        return list(getattr(_get_mod(), "__all__", []))


VERSION = ACTIVE_VERSION
