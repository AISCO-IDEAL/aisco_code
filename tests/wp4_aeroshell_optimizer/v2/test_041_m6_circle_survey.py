"""TDD tests for Plan 041 Milestone 6: CVM2 vis_latent_circle_survey + interpolator removal.

Tests are written BEFORE implementation (Red phase).

Scope:
- T1: CentralVisManagerV2 has vis_latent_circle_survey method
- T2: NullVisManager has vis_latent_circle_survey no-op
- T3: main_v2.py does not import setup_latent_interpolator / run_latent_interpolation
- T4: main_v2.py does not call setup_latent_interpolator / run_latent_interpolation
- T5: vis_latent_circle_survey signature has model, z_radius, n_samples params
- T6: finalize() source references latent_space_exploration gate
"""

from __future__ import annotations

import inspect
from pathlib import Path

from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import NullVisManager
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import CentralVisManagerV2

CENTRAL_VIS_V2_PATH = Path(
    "/cluster/home/kdiehl/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/src/sampling_visualization/central_vis_manager_v2.py"
)
CENTRAL_VIS_NULL_PATH = Path(
    "/cluster/home/kdiehl/AISCO_code/subprojects/wp4_aeroshell_optimizer"
    "/v2/src/sampling_visualization/central_vis_manager_null.py"
)
MAIN_V2_PATH = Path("/cluster/home/kdiehl/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/src/main_files/main_v2.py")


def _load_source(path: Path) -> str:
    return path.read_text()


class TestCVM2VisLatentCircleSurvey:
    """T1, T5: CVM2 has vis_latent_circle_survey with expected signature."""

    def test_cvm2_has_vis_latent_circle_survey(self) -> None:
        """CentralVisManagerV2 must have vis_latent_circle_survey method."""
        assert hasattr(CentralVisManagerV2, "vis_latent_circle_survey"), (
            "CentralVisManagerV2 missing vis_latent_circle_survey method. "
            "Add: def vis_latent_circle_survey(self, model, z_radius, n_samples, ...) -> None"
        )

    def test_vis_latent_circle_survey_has_expected_params(self) -> None:
        """vis_latent_circle_survey must accept model, z_radius, n_samples params."""
        sig = inspect.signature(CentralVisManagerV2.vis_latent_circle_survey)
        params = set(sig.parameters.keys())
        for required in ("model", "z_radius", "n_samples"):
            assert required in params, f"vis_latent_circle_survey missing '{required}' parameter. Signature has: {params}"

    def test_cvm2_source_calls_write_ply_in_survey(self) -> None:
        """vis_latent_circle_survey must submit PLY writes for each sample."""
        source = _load_source(CENTRAL_VIS_V2_PATH)
        assert "vis_latent_circle_survey" in source, "vis_latent_circle_survey not in CVM2 source"
        # Find method body
        survey_idx = source.find("def vis_latent_circle_survey")
        next_def = source.find("\n    def ", survey_idx + 1)
        body = source[survey_idx:next_def] if next_def != -1 else source[survey_idx:]
        assert "write_ply_task" in body or "_submit_ply" in body or "_pending_ply" in body, (
            "vis_latent_circle_survey does not submit PLY writes. "
            "Add a write_ply_task or _submit_ply call inside the survey loop."
        )

    def test_finalize_references_latent_space_exploration_gate(self) -> None:
        """finalize() must be gated by latent_space_exploration.enabled from vis_config."""
        source = _load_source(CENTRAL_VIS_V2_PATH)
        assert "latent_space_exploration" in source, (
            "CVM2 source does not reference latent_space_exploration config key. "
            "Gate vis_latent_circle_survey in finalize() on "
            "vis_config.latent_pretraining.latent_space_exploration.enabled"
        )


class TestNullVisManagerCircleSurvey:
    """T2: NullVisManager has vis_latent_circle_survey no-op."""

    def test_null_vis_manager_has_vis_latent_circle_survey(self) -> None:
        """NullVisManager must have vis_latent_circle_survey no-op."""
        assert hasattr(NullVisManager, "vis_latent_circle_survey"), (
            "NullVisManager missing vis_latent_circle_survey. "
            "Add: def vis_latent_circle_survey(self, *args, **kwargs) -> None: pass"
        )


class TestInterpolatorRemoval:
    """T3, T4: Interpolator imports and calls removed from main_v2.py."""

    def test_main_v2_does_not_import_setup_latent_interpolator(self) -> None:
        """main_v2.py must not import setup_latent_interpolator."""
        source = _load_source(MAIN_V2_PATH)
        assert "setup_latent_interpolator" not in source, (
            "main_v2.py still imports/calls setup_latent_interpolator. Remove the import and all call sites (M6 task)."
        )

    def test_main_v2_does_not_import_run_latent_interpolation(self) -> None:
        """main_v2.py must not import run_latent_interpolation."""
        source = _load_source(MAIN_V2_PATH)
        assert "run_latent_interpolation" not in source, (
            "main_v2.py still imports/calls run_latent_interpolation. Remove the import and all call sites (M6 task)."
        )

    def test_main_v2_does_not_import_cleanup_latent_interpolator(self) -> None:
        """main_v2.py must not import cleanup_latent_interpolator."""
        source = _load_source(MAIN_V2_PATH)
        assert "cleanup_latent_interpolator" not in source, (
            "main_v2.py still imports/calls cleanup_latent_interpolator. Remove the import and all call sites (M6 task)."
        )


class TestCircleSurveyCallsite:
    """Circle survey must be called and write to a dedicated output dir."""

    def test_main_v2_calls_vis_latent_circle_survey(self) -> None:
        """main_v2.py must call vis_latent_circle_survey after training ends."""
        source = _load_source(MAIN_V2_PATH)
        assert "vis_latent_circle_survey" in source, (
            "main_v2.py does not call vis_latent_circle_survey. "
            "Add a call after the training loop, before finalize(), "
            "gated on latent_space and vis_config latent_optimisation.vis_latent_circle_survey.enabled."
        )

    def test_circle_survey_does_not_write_to_mesh_assembly_dir(self) -> None:
        """vis_latent_circle_survey must NOT call vis_mesh_assembly_v4 (wrong output dir).

        It must use a dedicated circle_survey/ output path so it does not
        corrupt the optimisation-phase mesh_assembly_v4 GIF.
        """
        source = _load_source(CENTRAL_VIS_V2_PATH)
        survey_idx = source.find("def vis_latent_circle_survey")
        assert survey_idx != -1, "vis_latent_circle_survey not found in CVM2 source"
        next_def = source.find("\n    def ", survey_idx + 1)
        body = source[survey_idx:next_def] if next_def != -1 else source[survey_idx:]

        # Must NOT use mesh_assembly_v4 as the output target
        assert "vis_mesh_assembly_v4" not in body, (
            "vis_latent_circle_survey calls vis_mesh_assembly_v4(), which shares the "
            "'mesh_assembly_v4/' output directory with optimization-phase vis. "
            "Use a dedicated vis_type key (e.g. 'circle_survey') instead."
        )
        # Must reference a circle_survey output key (f-string or literal)
        assert "circle_survey" in body, (
            "vis_latent_circle_survey must write to a 'circle_survey/' subdirectory. "
            "Otherwise survey PLYs corrupt or collide with optimization-phase assembly GIF."
        )

    def test_circle_survey_mesh_resolution_is_configurable(self) -> None:
        """vis_latent_circle_survey must use mesh_resolution from its vis_config section."""
        source = _load_source(CENTRAL_VIS_V2_PATH)
        survey_idx = source.find("def vis_latent_circle_survey")
        next_def = source.find("\n    def ", survey_idx + 1)
        body = source[survey_idx:next_def] if next_def != -1 else source[survey_idx:]
        assert "mesh_resolution" in body or "resolution" in body, (
            "vis_latent_circle_survey must read mesh_resolution from the vis_config "
            "'latent_optimisation.vis_latent_circle_survey' section. "
            "Currently it ignores this setting and uses make_vis_extractor() defaults."
        )
