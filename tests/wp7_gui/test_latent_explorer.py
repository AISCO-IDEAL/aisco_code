"""TDD tests for subprojects.wp7_gui.v3.shared.latent_explorer.

Run BEFORE implementing the service so tests fail with ImportError/ModuleNotFoundError.
After implementation they should all pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# This import will fail (ModuleNotFoundError) before the service is created.
from subprojects.wp7_gui.v3.shared.latent_explorer import (
    find_last_epoch_checkpoint,
    is_checkpoint_path_allowed,
)

# ---------------------------------------------------------------------------
# is_checkpoint_path_allowed
# ---------------------------------------------------------------------------

_ALLOWED = Path("/cluster/work/fuge/AISCO/temp_/6166b4e953/models/run_epoch1000.pth")
_DISALLOWED_TMP = Path("/tmp/evil.pth")
_DISALLOWED_HOME = Path("/root/evil.pth")


def test_allowed_path_under_aisco_data():
    """Paths under /cluster/work/fuge/AISCO/ must be allowed."""
    assert is_checkpoint_path_allowed(_ALLOWED) is True


def test_disallowed_path_outside_allowlist():
    """/tmp paths must NOT be allowed."""
    assert is_checkpoint_path_allowed(_DISALLOWED_TMP) is False


def test_disallowed_path_home():
    """/root paths must NOT be allowed."""
    assert is_checkpoint_path_allowed(_DISALLOWED_HOME) is False


# ---------------------------------------------------------------------------
# find_last_epoch_checkpoint
# ---------------------------------------------------------------------------

_EXAMPLE_CHECKPOINT = Path(
    "/cluster/work/fuge/AISCO/temp_/6166b4e953/39_26423_feasible-monkey/models/39_26423_feasible-monkey_epoch4200.pth"
)

_EXAMPLE_MODELS_DIR = Path("/cluster/work/fuge/AISCO/temp_/6166b4e953/39_26423_feasible-monkey/models")


def test_find_last_epoch_checkpoint_from_file():
    """When given a direct .pth file path, returns it unchanged."""
    if not _EXAMPLE_CHECKPOINT.exists():
        pytest.skip("reference checkpoint not on this machine")
    result = find_last_epoch_checkpoint(_EXAMPLE_CHECKPOINT)
    assert result == _EXAMPLE_CHECKPOINT


def test_find_last_epoch_checkpoint_from_models_dir():
    """When given a models/ directory, returns the highest-epoch .pth file."""
    if not _EXAMPLE_MODELS_DIR.exists():
        pytest.skip("reference run not on this machine")
    result = find_last_epoch_checkpoint(_EXAMPLE_MODELS_DIR)
    assert result.suffix == ".pth"
    # epoch4200 should be the last checkpoint in this run
    assert "epoch4200" in result.stem


def test_find_last_epoch_checkpoint_from_run_dir():
    """When given the run root (parent of models/), finds the highest-epoch .pth."""
    run_dir = _EXAMPLE_MODELS_DIR.parent
    if not run_dir.exists():
        pytest.skip("reference run not on this machine")
    result = find_last_epoch_checkpoint(run_dir)
    assert result.suffix == ".pth"
    assert "epoch" in result.stem
