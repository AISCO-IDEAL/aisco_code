"""TDD tests for Plan 041 Milestone 4 — Pretraining Vis CVM2-Only.

Red phase: all tests must fail before implementation.
Green phase: all tests must pass after implementation.
"""

from __future__ import annotations

import inspect
import pathlib

from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_latent import (
    _write_epoch_vis_snapshot,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import (
    NullVisManager,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import (
    CentralVisManagerV2,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_V2_DIR = pathlib.Path(__file__).parents[3] / "subprojects" / "wp4_aeroshell_optimizer" / "v2"
_PRETRAIN_SRC = _V2_DIR / "src" / "main_files" / "pretraining_latent.py"
_CVM2_SRC = _V2_DIR / "src" / "sampling_visualization" / "central_vis_manager_v2.py"
_NULL_SRC = _V2_DIR / "src" / "sampling_visualization" / "central_vis_manager_null.py"


# ---------------------------------------------------------------------------
# T1: CentralVisManagerV2 exposes latent_pretraining_cfg property
# ---------------------------------------------------------------------------
class TestCVM2LatentPretrainingCfg:
    def test_cvm2_has_latent_pretraining_cfg_property(self) -> None:
        """CentralVisManagerV2 must expose a latent_pretraining_cfg property."""
        assert hasattr(CentralVisManagerV2, "latent_pretraining_cfg"), (
            "CentralVisManagerV2 must have a latent_pretraining_cfg property"
        )

    def test_cvm2_latent_pretraining_cfg_returns_dict(self, tmp_path: pathlib.Path) -> None:
        """latent_pretraining_cfg must return a dict."""
        cvm2 = CentralVisManagerV2(run_dir=str(tmp_path), run_id="test-001")
        result = cvm2.latent_pretraining_cfg
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    def test_cvm2_latent_pretraining_cfg_contains_cadence_key(self, tmp_path: pathlib.Path) -> None:
        """latent_pretraining_cfg must contain vis_cadence_pretraining key from vis_config.yaml."""
        cvm2 = CentralVisManagerV2(run_dir=str(tmp_path), run_id="test-001")
        cfg = cvm2.latent_pretraining_cfg
        assert "vis_cadence_pretraining" in cfg, (
            f"latent_pretraining_cfg must have 'vis_cadence_pretraining' key; got keys: {list(cfg.keys())}"
        )


# ---------------------------------------------------------------------------
# T2: NullVisManager exposes latent_pretraining_cfg property
# ---------------------------------------------------------------------------
class TestNullVisManagerLatentPretrainingCfg:
    def test_null_has_latent_pretraining_cfg_property(self) -> None:
        """NullVisManager must expose latent_pretraining_cfg as an empty dict."""
        nm = NullVisManager()
        assert hasattr(nm, "latent_pretraining_cfg"), "NullVisManager must have latent_pretraining_cfg"

    def test_null_latent_pretraining_cfg_returns_empty_dict(self) -> None:
        """NullVisManager.latent_pretraining_cfg must return an empty dict."""
        nm = NullVisManager()
        assert nm.latent_pretraining_cfg == {}, (
            f"NullVisManager.latent_pretraining_cfg must be empty dict, got {nm.latent_pretraining_cfg!r}"
        )


# ---------------------------------------------------------------------------
# T3: _write_epoch_vis_snapshot has no `vis` parameter (SamplingVisualizer removed)
# ---------------------------------------------------------------------------
class TestWriteEpochVisSnapshotSignature:
    def test_snapshot_has_no_vis_parameter(self) -> None:
        """`_write_epoch_vis_snapshot` must no longer accept a `vis` parameter."""
        sig = inspect.signature(_write_epoch_vis_snapshot)
        assert "vis" not in sig.parameters, (
            "_write_epoch_vis_snapshot still has 'vis' parameter. Remove SamplingVisualizer dual-path as per Plan 041 M4."
        )

    def test_snapshot_has_vis_manager_parameter(self) -> None:
        """`_write_epoch_vis_snapshot` must keep `vis_manager` parameter."""
        sig = inspect.signature(_write_epoch_vis_snapshot)
        assert "vis_manager" in sig.parameters, "_write_epoch_vis_snapshot must have 'vis_manager' parameter"


# ---------------------------------------------------------------------------
# T4: pretraining_latent.py no longer constructs epoch_vis SamplingVisualizer
# ---------------------------------------------------------------------------
class TestNoEpochVisConstruction:
    def test_no_sampling_visualizer_instantiation_in_pretraining_latent(self) -> None:
        """pretraining_latent() must not construct a SamplingVisualizer (epoch_vis)."""
        src = _PRETRAIN_SRC.read_text()
        # Check that epoch_vis = SamplingVisualizer(...) is not in the source
        assert "epoch_vis = SamplingVisualizer(" not in src, (
            "pretraining_latent.py still instantiates SamplingVisualizer for epoch_vis. Remove this per Plan 041 M4."
        )

    def test_no_epoch_vis_save_snapshot_call(self) -> None:
        """_write_epoch_vis_snapshot must not call vis.save_snapshot()."""
        src = _PRETRAIN_SRC.read_text()
        assert "vis.save_snapshot(" not in src, (
            "pretraining_latent.py still calls vis.save_snapshot(). Remove SamplingVisualizer calls per Plan 041 M4."
        )


# ---------------------------------------------------------------------------
# T5: _write_epoch_vis_snapshot calls vis_manager.vis_mesh_assembly_v4
# ---------------------------------------------------------------------------
class TestSnapshotCallsAssemblyVis:
    def test_snapshot_calls_vis_mesh_assembly_v4(self) -> None:
        """_write_epoch_vis_snapshot must call vis_manager.vis_mesh_assembly_v4()."""
        src = _PRETRAIN_SRC.read_text()
        assert "vis_manager.vis_mesh_assembly_v4(" in src, (
            "_write_epoch_vis_snapshot does not call vis_manager.vis_mesh_assembly_v4(). Add this call per Plan 041 M4."
        )


# ---------------------------------------------------------------------------
# T6: vis_cadence_pretraining gate in pretraining_latent()
# ---------------------------------------------------------------------------
class TestVisCadenceGate:
    def test_vis_cadence_pretraining_read_from_vis_manager(self) -> None:
        """pretraining_latent.py must read vis_cadence_pretraining from vis_manager config."""
        src = _PRETRAIN_SRC.read_text()
        assert "vis_cadence_pretraining" in src, (
            "pretraining_latent.py does not reference 'vis_cadence_pretraining'. "
            "Read from vis_manager.latent_pretraining_cfg per Plan 041 M4."
        )

    def test_no_epoch_vis_in_snapshot_gate(self) -> None:
        """Training loop gate must not reference epoch_vis."""
        src = _PRETRAIN_SRC.read_text()
        assert "epoch_vis is not None" not in src, (
            "pretraining_latent.py still references 'epoch_vis is not None' in the training loop gate. Remove per Plan 041 M4."
        )
