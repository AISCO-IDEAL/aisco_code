"""Regression tests for the OpenFOAM case template physics parameters.

These tests read the actual template files on disk and assert that key
physics values are present and correct.  They act as a regression guard:
any accidental removal of a parameter (e.g. curvatureCorrection) causes
an immediate, obvious failure here rather than a silent physics change.
"""

from pathlib import Path

TEMPLATE_DIR = Path(__file__).parents[1] / "case_template"


class TestTemplateBoundaryConditions:
    def test_inlet_velocity(self) -> None:
        """Inlet U uses __VINF__ placeholder for velocity."""
        content = (TEMPLATE_DIR / "0" / "U").read_text()
        assert "uniform (-__VINF__ 0 0)" in content

    def test_ground_moving_wall_velocity(self) -> None:
        """Ground is a moving wall using __VINF__ placeholder."""
        content = (TEMPLATE_DIR / "0" / "U").read_text()
        assert "movingWallVelocity" in content
        assert "uniform (-__VINF__ 0 0)" in content

    def test_car_surface_no_slip(self) -> None:
        """car_surface patch has noSlip BC for velocity."""
        content = (TEMPLATE_DIR / "0" / "U").read_text()
        assert "noSlip" in content

    def test_k_inlet_value(self) -> None:
        """Turbulent kinetic energy inlet uses __K_INLET__ placeholder."""
        content = (TEMPLATE_DIR / "0" / "k").read_text()
        assert "__K_INLET__" in content

    def test_omega_inlet_value(self) -> None:
        """Specific dissipation rate inlet uses __OMEGA_INLET__ placeholder."""
        content = (TEMPLATE_DIR / "0" / "omega").read_text()
        assert "__OMEGA_INLET__" in content

    def test_car_surface_k_wall_function(self) -> None:
        """car_surface uses kqRWallFunction for k."""
        content = (TEMPLATE_DIR / "0" / "k").read_text()
        assert "kqRWallFunction" in content

    def test_car_surface_omega_wall_function(self) -> None:
        """car_surface uses omegaWallFunction for omega."""
        content = (TEMPLATE_DIR / "0" / "omega").read_text()
        assert "omegaWallFunction" in content

    def test_six_boundary_patches_in_u(self) -> None:
        """All 6 required boundary patches are defined in U."""
        content = (TEMPLATE_DIR / "0" / "U").read_text()
        for patch in ("inlet", "outlet", "car_surface", "ground", "top", "sides"):
            assert patch in content, f"Patch '{patch}' missing from 0/U"


class TestTemplateTransportProperties:
    def test_kinematic_viscosity(self) -> None:
        """Kinematic viscosity is nu = mu/rho = 1.813e-5 / 1.225 = 1.480e-5."""
        content = (TEMPLATE_DIR / "constant" / "transportProperties").read_text()
        assert "1.480e-5" in content

    def test_newtonian_model(self) -> None:
        """Transport model is Newtonian."""
        content = (TEMPLATE_DIR / "constant" / "transportProperties").read_text()
        assert "Newtonian" in content


class TestTemplateTurbulenceProperties:
    def test_turbulence_model_komegasst(self) -> None:
        """Turbulence model is kOmegaSST."""
        content = (TEMPLATE_DIR / "constant" / "turbulenceProperties").read_text()
        assert "kOmegaSST" in content

    def test_curvature_correction_enabled(self) -> None:
        """curvatureCorrection is present (matching Fluent k-omega SST setup)."""
        content = (TEMPLATE_DIR / "constant" / "turbulenceProperties").read_text()
        assert "curvatureCorrection" in content

    def test_ras_simulation_type(self) -> None:
        """simulationType is RAS."""
        content = (TEMPLATE_DIR / "constant" / "turbulenceProperties").read_text()
        assert "RAS" in content


class TestTemplateControlDict:
    def test_application_simple_foam(self) -> None:
        """Solver application is simpleFoam."""
        content = (TEMPLATE_DIR / "system" / "controlDict").read_text()
        assert "simpleFoam" in content

    def test_force_function_objects_use_valid_openfoam_syntax(self) -> None:
        """Force-related function objects keep the whitespace-delimited syntax OpenFOAM expects."""
        content = (TEMPLATE_DIR / "system" / "controlDict").read_text()
        assert "libs            (forces);" in content
        assert "patches         (car_surface);" in content
        assert "CofR            (0 0 0);" in content
        assert "liftDir         (0 0 1);" in content
        assert "dragDir         (-1 0 0);" in content
        assert "pitchAxis       (0 1 0);" in content
        assert "libs            (fieldFunctionObjects);" in content

    def test_end_time_3000(self) -> None:
        """endTime is 3000 to ensure all cases converge (residualControl stops easy ones early)."""
        content = (TEMPLATE_DIR / "system" / "controlDict").read_text()
        assert "endTime         3000;" in content

    def test_write_interval_3000(self) -> None:
        """writeInterval is 3000 so only the final converged state is written by default."""
        content = (TEMPLATE_DIR / "system" / "controlDict").read_text()
        assert "writeInterval   3000;" in content

    def test_force_coeffs_function_object(self) -> None:
        """forceCoeffs function object is defined for Cd extraction."""
        content = (TEMPLATE_DIR / "system" / "controlDict").read_text()
        assert "forceCoeffs" in content
        assert "car_surface" in content


class TestTemplateNumerics:
    def test_decompose_par_nprocs_placeholder(self) -> None:
        """decomposeParDict uses __NPROCS__ placeholder and scotch method."""
        content = (TEMPLATE_DIR / "system" / "decomposeParDict").read_text()
        assert "__NPROCS__" in content
        assert "scotch" in content

    def test_div_scheme_linear_upwind_for_u(self) -> None:
        """Divergence scheme for U uses linearUpwind (second-order)."""
        content = (TEMPLATE_DIR / "system" / "fvSchemes").read_text()
        assert "linearUpwind" in content

    def test_p_relaxation_factor(self) -> None:
        """Pressure relaxation factor is 0.3."""
        content = (TEMPLATE_DIR / "system" / "fvSolution").read_text()
        assert "0.3" in content

    def test_u_relaxation_factor(self) -> None:
        """Velocity relaxation factor is 0.7."""
        content = (TEMPLATE_DIR / "system" / "fvSolution").read_text()
        assert "0.7" in content

    def test_fvsolution_keeps_relaxation_and_residual_blocks(self) -> None:
        """fvSolution retains the expected residualControl and relaxationFactors blocks."""
        content = (TEMPLATE_DIR / "system" / "fvSolution").read_text()
        assert "residualControl" in content
        assert "p               1e-5;" in content
        assert "U               1e-4;" in content
        assert "k               1e-4;" in content
        assert "omega           1e-4;" in content
        assert "relaxationFactors" in content
        assert "fields" in content
        assert "equations" in content
        assert "p               0.3;" in content
        assert "U               0.7;" in content

    def test_snappy_boi_placeholders(self) -> None:
        """snappyHexMeshDict contains all 6 BOI coordinate placeholders."""
        content = (TEMPLATE_DIR / "system" / "snappyHexMeshDict").read_text()
        for placeholder in (
            "__BOI_XMIN__",
            "__BOI_XMAX__",
            "__BOI_YMIN__",
            "__BOI_YMAX__",
            "__BOI_ZMIN__",
            "__BOI_ZMAX__",
        ):
            assert placeholder in content, f"Placeholder '{placeholder}' missing from snappyHexMeshDict"

    def test_snappy_surface_refinement_is_increased(self) -> None:
        """The car surface and feature-edge refinement are raised one notch for better surface resolution."""
        content = (TEMPLATE_DIR / "system" / "snappyHexMeshDict").read_text()
        assert "level   5;" in content
        assert "level   (4 5);" in content

    def test_snappy_boundary_layers_enabled(self) -> None:
        """Boundary layers are active with 8 nSurfaceLayers for car and ground (v040 upgrade)."""
        content = (TEMPLATE_DIR / "system" / "snappyHexMeshDict").read_text()
        assert "addLayers       true;" in content
        assert "nSurfaceLayers  8;" in content
