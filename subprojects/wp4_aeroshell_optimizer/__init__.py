"""wp4_aeroshell_optimizer — versioned public API.

Change ACTIVE_VERSION below to switch all callers between v1 and v2.
"""

ACTIVE_VERSION = "v1"  # change to "v2" to switch globally

if ACTIVE_VERSION == "v1":
    from .v1 import *  # noqa: F403
    from .v1 import CONFIGS_DIR, SLURM_SCRIPTS_DIR, __all__
elif ACTIVE_VERSION == "v2":
    from .v2 import *  # type: ignore[assignment, no-redef]  # noqa: F403
    from .v2 import CONFIGS_DIR, SLURM_SCRIPTS_DIR, __all__  # type: ignore[assignment, no-redef]
else:
    raise ValueError(f"Unknown ACTIVE_VERSION: {ACTIVE_VERSION!r}. Must be 'v1' or 'v2'.")

VERSION = ACTIVE_VERSION
