"""Regression tests for OpenFOAM SLURM helper scripts."""

from pathlib import Path

SLURM_DIR = Path(__file__).parents[1] / "slurm"


class TestOpenfoamSolveArrayScript:
    def test_can_optionally_stream_iteration_progress(self) -> None:
        """Solve script exposes an opt-in iteration-progress stream in SLURM stdout."""
        content = (SLURM_DIR / "openfoam_solve_array.slurm").read_text()

        assert "SHOW_ITERATIONS" in content
        assert "ITERATION_LOG_INTERVAL" in content
        assert "tee log.simpleFoam" in content
        assert "Time =" in content
