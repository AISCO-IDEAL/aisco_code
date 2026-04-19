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

    def test_reads_proc_count_from_decomposepardict(self) -> None:
        """Solve script reads numberOfSubdomains from decomposeParDict instead of hardcoding."""
        content = (SLURM_DIR / "openfoam_solve_array.slurm").read_text()

        assert "numberOfSubdomains" in content
        assert "decomposeParDict" in content
        # Must NOT hardcode mpirun -np 128 (should use variable)
        assert 'mpirun -np "$ATTEMPT_PROCS"' in content

    def test_has_mpi_crash_retry_cascade(self) -> None:
        """Solve script retries at reduced proc counts on MPI crash."""
        content = (SLURM_DIR / "openfoam_solve_array.slurm").read_text()

        assert "RETRY_TIERS" in content
        assert "MPI_CRASH" in content
        assert "reduced_procs" in content
        assert "All retry tiers exhausted" in content

    def test_has_lustre_safe_cleanup(self) -> None:
        """Solve script uses Lustre-tolerant processor dir cleanup."""
        content = (SLURM_DIR / "openfoam_solve_array.slurm").read_text()

        assert "lustre_safe_cleanup" in content

    def test_has_structured_error_logging(self) -> None:
        """Solve script emits structured JSON error events to stderr."""
        content = (SLURM_DIR / "openfoam_solve_array.slurm").read_text()

        assert "log_error" in content
        assert "log_event" in content
        assert "error_class" in content
        assert "run_id" in content


class TestOpenfoamMeshArrayScript:
    def test_has_cell_count_sanity_check(self) -> None:
        """Mesh script validates cell count against STL face count heuristic."""
        content = (SLURM_DIR / "openfoam_mesh_array.slurm").read_text()

        assert "STL_FACES" in content
        assert "CELL_COUNT" in content
        assert "50" in content  # lower bound of expected ratio
        assert "300" in content  # upper bound of expected ratio

    def test_has_structured_error_logging(self) -> None:
        """Mesh script emits structured JSON error events."""
        content = (SLURM_DIR / "openfoam_mesh_array.slurm").read_text()

        assert "log_error" in content
        assert "log_event" in content
        assert "error_class" in content

    def test_has_lustre_safe_cleanup(self) -> None:
        """Mesh script uses Lustre-tolerant processor dir cleanup."""
        content = (SLURM_DIR / "openfoam_mesh_array.slurm").read_text()

        assert "lustre_safe_cleanup" in content

    def test_has_reconstructparmesh_crash_tolerance(self) -> None:
        """Mesh script tolerates reconstructParMesh crashes when mesh files exist."""
        content = (SLURM_DIR / "openfoam_mesh_array.slurm").read_text()

        assert "constant/polyMesh/owner" in content
        assert "Mesh OK" in content
        assert "checkMesh_verify" in content


class TestSubmitPipelineScript:
    def test_has_numeric_guard_on_max_parallel(self) -> None:
        """submit_pipeline.sh guards against non-numeric $2 corrupting MAX_PARALLEL (Bug G)."""
        content = (SLURM_DIR / "submit_pipeline.sh").read_text()

        # Must use a numeric regex to validate $2 before using it as MAX_PARALLEL
        assert "^[0-9]+$" in content
        # Must detect --no-extract via loop before reading $2
        assert '--no-extract' in content
        assert "SKIP_EXTRACT=1" in content

    def test_extract_dependency_is_afterany(self) -> None:
        """submit_pipeline.sh uses afterany (not afterok) for extract dependency (Bug E)."""
        content = (SLURM_DIR / "submit_pipeline.sh").read_text()

        assert '--dependency="afterany:${SOLVE_JOBID}"' in content
        # Echo message must match the actual dependency type
        assert "afterok:${SOLVE_JOBID}" not in content
