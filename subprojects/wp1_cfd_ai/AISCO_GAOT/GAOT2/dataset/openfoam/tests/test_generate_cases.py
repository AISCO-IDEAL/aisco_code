"""Tests for generate_cases.py — OpenFOAM case instantiation."""

import json
from pathlib import Path

import pytest
import trimesh

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.dataset.openfoam.generate_cases import (
    compute_boi_box,
    compute_domain_extents,
    generate_all_cases,
    instantiate_case,
    _MIN_UPSTREAM_EXTENT,
)

# Path to the case template in the repo
_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "case_template"


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_bounds() -> tuple[list[float], list[float]]:
    """Realistic car bounds in meters (after unit normalization)."""
    return [-2.5, -0.9, 0.0], [2.5, 0.9, 1.5]


@pytest.fixture
def temp_dirs(tmp_path: Path) -> dict[str, Path]:
    """Create temp directories for cases and geometry."""
    output_dir = tmp_path / "CASES"
    output_dir.mkdir()
    geom_dir = tmp_path / "GEOMETRY"
    geom_dir.mkdir()
    return {"output": output_dir, "geometry": geom_dir}


@pytest.fixture
def sample_manifest(tmp_path: Path, temp_dirs: dict[str, Path]) -> Path:
    """Create a sample geometry manifest with 2 converted runs."""
    geom_dir = temp_dirs["geometry"]
    entries = []
    for i in [1, 2]:
        # Create a real STL file
        box = trimesh.creation.box(extents=[4.0, 1.8, 1.5])
        stl_path = geom_dir / f"RUN_{i:03d}.stl"
        box.export(str(stl_path))
        entries.append(
            {
                "run_id": f"RUN_{i:03d}",
                "stl_path": str(stl_path),
                "status": "converted",
                "detected_unit": "mm",
                "bounds_min": [-2.0, -0.9, -0.75],
                "bounds_max": [2.0, 0.9, 0.75],
            }
        )
    # Add an excluded run
    entries.append(
        {
            "run_id": "RUN_003",
            "stl_path": "",
            "status": "excluded",
            "exclusion_reason": "Not watertight",
        }
    )
    manifest_path = tmp_path / "geometry_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(entries, f)
    return manifest_path


# ---------------------------------------------------------------------------
#  compute_domain_extents
# ---------------------------------------------------------------------------


class TestComputeDomainExtents:
    def test_domain_extends_upstream_and_downstream(self, sample_bounds: tuple) -> None:
        """Domain extends ~2 car-lengths upstream, ~5 downstream."""
        bmin, bmax = sample_bounds
        domain = compute_domain_extents(bmin, bmax)
        car_length = bmax[0] - bmin[0]  # 5.0m
        # Upstream: xMin should be well before geometry
        assert domain["xMin"] < bmin[0] - car_length
        # Downstream: xMax should be well after geometry
        assert domain["xMax"] > bmax[0] + 2 * car_length

    def test_domain_extends_laterally(self, sample_bounds: tuple) -> None:
        """Domain extends ~2 car-widths on each side."""
        bmin, bmax = sample_bounds
        domain = compute_domain_extents(bmin, bmax)
        car_width = bmax[1] - bmin[1]  # 1.8m
        assert domain["yMin"] < bmin[1] - car_width
        assert domain["yMax"] > bmax[1] + car_width

    def test_domain_ground_at_zero(self, sample_bounds: tuple) -> None:
        """Ground is always at z=0."""
        bmin, bmax = sample_bounds
        domain = compute_domain_extents(bmin, bmax)
        assert domain["zMin"] == 0.0

    def test_domain_has_cell_counts(self, sample_bounds: tuple) -> None:
        """Domain has integer cell counts based on ~0.164m cell size."""
        bmin, bmax = sample_bounds
        domain = compute_domain_extents(bmin, bmax)
        assert isinstance(domain["xCells"], int)
        assert isinstance(domain["yCells"], int)
        assert isinstance(domain["zCells"], int)
        assert domain["xCells"] > 0
        assert domain["yCells"] > 0
        assert domain["zCells"] > 0

    def test_minimum_upstream_floor_applied_for_small_geometry(self) -> None:
        """Small geometries get at least _MIN_UPSTREAM_EXTENT upstream (not just 2×car_length).

        Regression test for analysis 062: geometries shorter than 0.75m had
        proportional upstream extents so short that locationInMesh fell outside
        the blockMesh domain, crashing snappyHexMesh.
        """
        # RUN_220 archetype: 0.216m cube (the smallest real failing case)
        bmin = [-0.108, 0.042, 0.1]
        bmax = [0.108, 0.258, 0.316]
        car_length = bmax[0] - bmin[0]  # 0.216m — well below 0.75m threshold
        proportional_upstream = 2.0 * car_length  # 0.432m
        assert proportional_upstream < _MIN_UPSTREAM_EXTENT, (
            "Test precondition: proportional upstream must be less than minimum floor"
        )
        domain = compute_domain_extents(bmin, bmax)
        actual_upstream = bmin[0] - domain["xMin"]
        assert actual_upstream >= _MIN_UPSTREAM_EXTENT, (
            f"Upstream extent {actual_upstream:.4f}m is below the minimum "
            f"{_MIN_UPSTREAM_EXTENT}m floor for a {car_length:.3f}m car"
        )

    def test_normal_geometry_uses_proportional_upstream(self) -> None:
        """Standard-size geometries use proportional upstream (floor does not activate)."""
        # Typical solar car: 3.5m long
        bmin = [-1.75, -0.9, 0.0]
        bmax = [1.75, 0.9, 1.5]
        car_length = bmax[0] - bmin[0]  # 3.5m
        proportional_upstream = 2.0 * car_length  # 7.0m — well above floor
        domain = compute_domain_extents(bmin, bmax)
        actual_upstream = bmin[0] - domain["xMin"]
        assert abs(actual_upstream - proportional_upstream) < 0.001, (
            f"Normal geometry should use proportional upstream {proportional_upstream:.3f}m, "
            f"got {actual_upstream:.4f}m"
        )


# ---------------------------------------------------------------------------
#  compute_boi_box
# ---------------------------------------------------------------------------


class TestComputeBoiBox:
    def test_boi_extends_with_padding(self, sample_bounds: tuple) -> None:
        """BOI box is larger than geometry bounds in all directions."""
        bmin, bmax = sample_bounds
        boi = compute_boi_box(bmin, bmax)
        assert boi["xMin"] < bmin[0]
        assert boi["xMax"] > bmax[0]
        assert boi["yMin"] < bmin[1]
        assert boi["yMax"] > bmax[1]
        assert boi["zMax"] > bmax[2]

    def test_boi_zmin_at_ground(self, sample_bounds: tuple) -> None:
        """BOI box starts at ground level."""
        bmin, bmax = sample_bounds
        boi = compute_boi_box(bmin, bmax)
        assert boi["zMin"] == 0.0


# ---------------------------------------------------------------------------
#  instantiate_case
# ---------------------------------------------------------------------------


class TestInstantiateCase:
    def test_creates_case_directory(self, temp_dirs: dict[str, Path]) -> None:
        """Creates a complete case directory from template."""
        geom_dir = temp_dirs["geometry"]
        box = trimesh.creation.box(extents=[4.0, 1.8, 1.5])
        stl_path = geom_dir / "RUN_001.stl"
        box.export(str(stl_path))

        result = instantiate_case(
            run_id="RUN_001",
            stl_path=stl_path,
            bounds_min=[-2.0, -0.9, -0.75],
            bounds_max=[2.0, 0.9, 0.75],
            template_dir=_TEMPLATE_DIR,
            output_dir=temp_dirs["output"],
        )
        case_dir = Path(result["case_path"])
        assert case_dir.exists()
        assert (case_dir / "system" / "controlDict").exists()
        assert (case_dir / "system" / "blockMeshDict").exists()
        assert (case_dir / "system" / "snappyHexMeshDict").exists()
        assert (case_dir / "0" / "U").exists()
        assert (case_dir / "constant" / "triSurface" / "car_surface.stl").exists()

    def test_placeholders_replaced_in_blockmesh(self, temp_dirs: dict[str, Path]) -> None:
        """blockMeshDict has no remaining __PLACEHOLDER__ tokens."""
        geom_dir = temp_dirs["geometry"]
        box = trimesh.creation.box(extents=[4.0, 1.8, 1.5])
        stl_path = geom_dir / "RUN_001.stl"
        box.export(str(stl_path))

        result = instantiate_case(
            run_id="RUN_001",
            stl_path=stl_path,
            bounds_min=[-2.0, -0.9, -0.75],
            bounds_max=[2.0, 0.9, 0.75],
            template_dir=_TEMPLATE_DIR,
            output_dir=temp_dirs["output"],
        )
        bmd = Path(result["case_path"]) / "system" / "blockMeshDict"
        content = bmd.read_text()
        assert "__" not in content, f"Unreplaced placeholders in blockMeshDict: {content}"

    def test_placeholders_replaced_in_snappy(self, temp_dirs: dict[str, Path]) -> None:
        """snappyHexMeshDict has no remaining __PLACEHOLDER__ tokens."""
        geom_dir = temp_dirs["geometry"]
        box = trimesh.creation.box(extents=[4.0, 1.8, 1.5])
        stl_path = geom_dir / "RUN_001.stl"
        box.export(str(stl_path))

        result = instantiate_case(
            run_id="RUN_001",
            stl_path=stl_path,
            bounds_min=[-2.0, -0.9, -0.75],
            bounds_max=[2.0, 0.9, 0.75],
            template_dir=_TEMPLATE_DIR,
            output_dir=temp_dirs["output"],
        )
        shmd = Path(result["case_path"]) / "system" / "snappyHexMeshDict"
        content = shmd.read_text()
        assert "__" not in content, f"Unreplaced placeholders in snappyHexMeshDict: {content}"

    def test_location_in_mesh_not_on_cell_face_for_symmetric_geometry(
        self, temp_dirs: dict[str, Path]
    ) -> None:
        """locationInMesh y-coordinate must not coincide with a blockMesh cell face.

        Regression test for RUN_018: symmetric geometry (y_center=0) caused
        locationInMesh to land exactly on cell boundary, triggering snappyHexMesh
        FOAM FATAL ERROR: 'Point is not inside the mesh'.
        """
        from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.dataset.openfoam.generate_cases import (
            _CELL_SIZE,
            compute_domain_extents,
        )

        geom_dir = temp_dirs["geometry"]
        box = trimesh.creation.box(extents=[3.5, 1.2374, 1.6371])
        stl_path = geom_dir / "RUN_018.stl"
        box.export(str(stl_path))

        # Symmetric geometry: bounds symmetric about y=0 (solar car-like)
        bounds_min = [-1.7917, -0.6137, 0.1]
        bounds_max = [1.7329, 0.6249, 1.7371]

        result = instantiate_case(
            run_id="RUN_018",
            stl_path=stl_path,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            template_dir=_TEMPLATE_DIR,
            output_dir=temp_dirs["output"],
        )

        shmd = Path(result["case_path"]) / "system" / "snappyHexMeshDict"
        content = shmd.read_text()

        # Extract locationInMesh y value from the dict
        import re

        m = re.search(r"locationInMesh\s+\(\s*[-\d.]+\s+([-\d.]+)\s+[-\d.]+\s*\)", content)
        assert m is not None, "Could not find locationInMesh in snappyHexMeshDict"
        loc_y = float(m.group(1))

        # Verify the y point is not on a cell face boundary
        domain = compute_domain_extents(bounds_min, bounds_max)
        y_offset_from_min = loc_y - domain["yMin"]
        cell_fraction = (y_offset_from_min / _CELL_SIZE) % 1.0
        # Must be at least 10% of a cell away from any face
        assert cell_fraction > 0.1 and cell_fraction < 0.9, (
            f"locationInMesh y={loc_y} is too close to a cell face boundary "
            f"(cell_fraction={cell_fraction:.4f}). snappyHexMesh will fail."
        )

    def test_location_in_mesh_inside_domain_for_small_geometry(
        self, temp_dirs: dict[str, Path]
    ) -> None:
        """locationInMesh x-coordinate is inside the domain for small geometries.

        Regression test for analysis 062 / Bug F: geometries shorter than 0.5m
        had loc_x = bounds_min[0] - 1.0 which fell outside the blockMesh domain
        (domain xMin = bounds_min[0] - 2*car_length), crashing snappyHexMesh.
        With Option B the domain has a minimum upstream floor and loc_x is
        derived as the midpoint between domain["xMin"] and bounds_min[0].
        """
        import re

        geom_dir = temp_dirs["geometry"]
        # RUN_220 archetype: 0.216m cube — the worst failing case
        box = trimesh.creation.box(extents=[0.216, 0.216, 0.216])
        stl_path = geom_dir / "RUN_SMALL.stl"
        box.export(str(stl_path))
        bounds_min = [-0.108, 0.042, 0.1]
        bounds_max = [0.108, 0.258, 0.316]

        result = instantiate_case(
            run_id="RUN_SMALL",
            stl_path=stl_path,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            template_dir=_TEMPLATE_DIR,
            output_dir=temp_dirs["output"],
        )

        shmd = Path(result["case_path"]) / "system" / "snappyHexMeshDict"
        snappy_content = shmd.read_text()
        bmd = Path(result["case_path"]) / "system" / "blockMeshDict"
        bmd_content = bmd.read_text()

        # Extract domain xMin from blockMeshDict
        m_xmin = re.search(r"xMin\s+([\-\d.]+)", bmd_content)
        assert m_xmin is not None
        domain_xmin = float(m_xmin.group(1))

        # Extract locationInMesh x from snappyHexMeshDict
        m_loc = re.search(r"locationInMesh\s+\(\s*([\-\d.]+)", snappy_content)
        assert m_loc is not None
        loc_x = float(m_loc.group(1))

        assert loc_x > domain_xmin, (
            f"locationInMesh x={loc_x} is outside domain xMin={domain_xmin} "
            "for small geometry. snappyHexMesh will crash."
        )
        assert loc_x < bounds_min[0], (
            f"locationInMesh x={loc_x} must be upstream of geometry (bounds_min[0]={bounds_min[0]})"
        )


# ---------------------------------------------------------------------------
#  generate_all_cases
# ---------------------------------------------------------------------------


class TestGenerateAllCases:
    def test_generates_cases_for_converted_runs_only(self, sample_manifest: Path, temp_dirs: dict[str, Path]) -> None:
        """Only creates cases for runs with status='converted'."""
        results = generate_all_cases(
            manifest_path=sample_manifest,
            template_dir=_TEMPLATE_DIR,
            output_dir=temp_dirs["output"],
        )
        # 2 converted + 1 excluded → 2 cases generated
        assert len(results) == 2
        assert all(r["status"] == "created" for r in results)

    def test_skips_excluded_runs(self, sample_manifest: Path, temp_dirs: dict[str, Path]) -> None:
        """Excluded runs are not instantiated."""
        results = generate_all_cases(
            manifest_path=sample_manifest,
            template_dir=_TEMPLATE_DIR,
            output_dir=temp_dirs["output"],
        )
        run_ids = {r["run_id"] for r in results}
        assert "RUN_003" not in run_ids

    def test_writes_cases_manifest(self, sample_manifest: Path, temp_dirs: dict[str, Path]) -> None:
        """Writes a cases_manifest.json."""
        cases_manifest = temp_dirs["output"].parent / "cases_manifest.json"
        generate_all_cases(
            manifest_path=sample_manifest,
            template_dir=_TEMPLATE_DIR,
            output_dir=temp_dirs["output"],
            cases_manifest_path=cases_manifest,
        )
        assert cases_manifest.exists()
        with open(cases_manifest) as f:
            data = json.load(f)
        assert len(data) == 2
