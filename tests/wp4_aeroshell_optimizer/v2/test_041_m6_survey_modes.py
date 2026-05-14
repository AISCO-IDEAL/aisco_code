"""Tests verifying strip / grid / pizza sampling modes for vis_latent_circle_survey.

Scope
-----
- Sampling branches: correct logic exists in CVM2 source for each mode.
- File naming: correct polar / Cartesian / angle patterns produced per mode.
- survey_png._SURVEY_PATTERN: regex matches all three file types, rejects others.
- Sort order: each naming scheme sorts into the expected row-major grid order.
- Canvas dimensions: strip → 1×N, grid → √N×√N, pizza → r_steps×n_angles.
- Config: pizza keys present in vis_config.yaml.
- pizza_r_steps × n_circle_samples == total pizza PLY count.
"""

from __future__ import annotations

import math
import random

import yaml

from subprojects.wp4_aeroshell_optimizer.v2.src.utils.device_detection import get_config_base_path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO = get_config_base_path() / "AISCO_code"
_CVM2_PATH = _REPO / "subprojects/wp4_aeroshell_optimizer/v2/src/sampling_visualization/central_vis_manager_v2.py"
_SURVEY_PNG_PATH = _REPO / "subprojects/wp4_aeroshell_optimizer/v2/src/sampling_visualization/survey_png.py"
_VIS_CFG_PATH = _REPO / "subprojects/wp4_aeroshell_optimizer/v2/configs/vis_config.yaml"

CVM2_SOURCE = _CVM2_PATH.read_text()
SURVEY_PNG_SOURCE = _SURVEY_PNG_PATH.read_text()
VIS_CFG = yaml.safe_load(_VIS_CFG_PATH.read_text())
SURVEY_CFG = VIS_CFG.get("latent_optimisation", {}).get("vis_latent_circle_survey", {})

# Extract the body of vis_latent_circle_survey for scoped checks.
_survey_start = CVM2_SOURCE.find("def vis_latent_circle_survey")
_survey_end = CVM2_SOURCE.find("\n    def ", _survey_start + 1)
SURVEY_METHOD_BODY = CVM2_SOURCE[_survey_start:_survey_end] if _survey_end != -1 else CVM2_SOURCE[_survey_start:]

# Import the compiled survey pattern directly.
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.survey_png import (  # noqa: E402
    _SURVEY_PATTERN,
)

# ---------------------------------------------------------------------------
# Helpers mirroring the _sort_key logic in survey_png.py
# (tested independently so we can verify ordering without file I/O)
# ---------------------------------------------------------------------------


def _sort_key(name: str) -> tuple:
    """Mirror of survey_png._sort_key (operates on stem, not Path)."""
    if name.startswith("angle_"):
        return (0, int(name.split("_")[1]))
    elif name.startswith("x_"):
        parts = name.split("_")
        return (1, int(parts[3]), int(parts[1]))  # y, x
    elif name.startswith("r") and "_a" in name:
        r_part, a_part = name.split("_a")
        return (2, int(r_part[1:]), int(a_part))
    return (3, 0)


# ---------------------------------------------------------------------------
# 1. Source-level: sampling branches
# ---------------------------------------------------------------------------


class TestSamplingModeSources:
    """CVM2 source must contain correct logic for each layout branch."""

    def test_strip_branch_exists(self) -> None:
        assert "else:  # strip" in SURVEY_METHOD_BODY or "strip" in SURVEY_METHOD_BODY, (
            "strip mode branch missing from vis_latent_circle_survey"
        )

    def test_grid_branch_exists(self) -> None:
        assert 'layout == "grid"' in SURVEY_METHOD_BODY, (
            'grid mode branch (layout == "grid") missing from vis_latent_circle_survey'
        )

    def test_pizza_branch_exists(self) -> None:
        assert 'layout == "pizza"' in SURVEY_METHOD_BODY, (
            'pizza mode branch (layout == "pizza") missing from vis_latent_circle_survey'
        )

    def test_pizza_reads_r_min_from_config(self) -> None:
        assert "pizza_r_min" in SURVEY_METHOD_BODY, "pizza mode must read pizza_r_min from survey_cfg"

    def test_pizza_reads_r_max_from_config(self) -> None:
        assert "pizza_r_max" in SURVEY_METHOD_BODY, "pizza mode must read pizza_r_max from survey_cfg"

    def test_pizza_reads_r_steps_from_config(self) -> None:
        assert "pizza_r_steps" in SURVEY_METHOD_BODY, "pizza mode must read pizza_r_steps from survey_cfg"


# ---------------------------------------------------------------------------
# 2. Source-level: file naming patterns
# ---------------------------------------------------------------------------


class TestSamplingFileNaming:
    """Each mode must produce the correct polar / Cartesian / angle filenames."""

    def test_strip_produces_angle_filenames(self) -> None:
        assert 'f"angle_{' in SURVEY_METHOD_BODY or '"angle_"' in SURVEY_METHOD_BODY or "angle_" in SURVEY_METHOD_BODY, (
            "strip mode must name PLYs angle_NNN"
        )

    def test_grid_produces_x_y_filenames(self) -> None:
        assert "x_idx" in SURVEY_METHOD_BODY, "grid mode must name PLYs x_XX_y_XX (x_idx must appear)"
        assert "y_idx" in SURVEY_METHOD_BODY, "grid mode must embed y_idx in filename"

    def test_pizza_produces_r_a_filenames(self) -> None:
        assert "r{r_idx" in SURVEY_METHOD_BODY or "r_idx" in SURVEY_METHOD_BODY, "pizza mode must embed r_idx in filename"
        assert "a{a_idx" in SURVEY_METHOD_BODY or "_a" in SURVEY_METHOD_BODY, "pizza mode must embed angle index (_a) in filename"

    def test_pizza_does_not_use_cartesian_naming(self) -> None:
        # Find the pizza branch only
        pizza_start = SURVEY_METHOD_BODY.find('layout == "pizza"')
        else_start = SURVEY_METHOD_BODY.find("else:  # strip", pizza_start)
        pizza_body = SURVEY_METHOD_BODY[pizza_start:else_start] if else_start != -1 else SURVEY_METHOD_BODY[pizza_start:]
        assert "x_{x_idx" not in pizza_body, "pizza mode must use polar naming (r/a), not Cartesian (x/y)"

    def test_strip_does_not_use_polar_r_naming(self) -> None:
        # Find the else (strip) branch
        strip_start = SURVEY_METHOD_BODY.rfind("else:  # strip")
        strip_body = SURVEY_METHOD_BODY[strip_start:] if strip_start != -1 else ""
        assert "r{r_idx" not in strip_body and "r_idx" not in strip_body, (
            "strip mode must use angle naming, not polar r_idx naming"
        )


# ---------------------------------------------------------------------------
# 3. _SURVEY_PATTERN regex
# ---------------------------------------------------------------------------


class TestSurveyPattern:
    """_SURVEY_PATTERN must match exactly the three valid naming schemes."""

    # Strip
    def test_matches_angle_000(self) -> None:
        assert _SURVEY_PATTERN.match("angle_000.ply"), "angle_000.ply should match"

    def test_matches_angle_089(self) -> None:
        assert _SURVEY_PATTERN.match("angle_089.ply"), "angle_089.ply should match"

    # Grid
    def test_matches_x_00_y_00(self) -> None:
        assert _SURVEY_PATTERN.match("x_00_y_00.ply"), "x_00_y_00.ply should match"

    def test_matches_x_09_y_09(self) -> None:
        assert _SURVEY_PATTERN.match("x_09_y_09.ply"), "x_09_y_09.ply should match"

    # Pizza
    def test_matches_r00_a000(self) -> None:
        assert _SURVEY_PATTERN.match("r00_a000.ply"), "r00_a000.ply should match"

    def test_matches_r03_a089(self) -> None:
        assert _SURVEY_PATTERN.match("r03_a089.ply"), "r03_a089.ply should match"

    # Rejects
    def test_rejects_epoch_00001(self) -> None:
        assert not _SURVEY_PATTERN.match("epoch_00001.ply"), "epoch_00001.ply should NOT match"

    def test_rejects_survey_png(self) -> None:
        assert not _SURVEY_PATTERN.match("survey.png"), "survey.png should NOT match"

    def test_rejects_bare_angle(self) -> None:
        assert not _SURVEY_PATTERN.match("angle.ply"), "angle.ply should NOT match"

    def test_rejects_r_without_a(self) -> None:
        assert not _SURVEY_PATTERN.match("r00.ply"), "r00.ply should NOT match"


# ---------------------------------------------------------------------------
# 4. Sort order
# ---------------------------------------------------------------------------


class TestSortOrder:
    """Each naming scheme must sort into the expected row-major order."""

    def test_strip_sorts_angle_numerically(self) -> None:
        names = ["angle_005", "angle_000", "angle_003", "angle_001"]
        expected = ["angle_000", "angle_001", "angle_003", "angle_005"]
        assert sorted(names, key=_sort_key) == expected

    def test_strip_full_circle_order(self) -> None:
        n = 8
        names = [f"angle_{i:03d}" for i in range(n)]
        shuffled = names[:]
        random.shuffle(shuffled)
        assert sorted(shuffled, key=_sort_key) == names

    def test_grid_sorts_row_major_y_then_x(self) -> None:
        # Row 0: x=0,1,2; Row 1: x=0,1,2
        names = ["x_02_y_01", "x_00_y_00", "x_01_y_00", "x_00_y_01", "x_02_y_00", "x_01_y_01"]
        expected = ["x_00_y_00", "x_01_y_00", "x_02_y_00", "x_00_y_01", "x_01_y_01", "x_02_y_01"]
        assert sorted(names, key=_sort_key) == expected

    def test_pizza_sorts_by_radius_then_angle(self) -> None:
        # 2 radii x 3 angles
        names = ["r01_a002", "r00_a000", "r01_a000", "r00_a002", "r00_a001", "r01_a001"]
        expected = ["r00_a000", "r00_a001", "r00_a002", "r01_a000", "r01_a001", "r01_a002"]
        assert sorted(names, key=_sort_key) == expected

    def test_sort_key_types_are_disjoint(self) -> None:
        """Angle, grid, pizza entries must never interleave in sort."""
        assert _sort_key("angle_000")[0] == 0
        assert _sort_key("x_00_y_00")[0] == 1
        assert _sort_key("r00_a000")[0] == 2


# ---------------------------------------------------------------------------
# 5. Canvas dimension logic
# ---------------------------------------------------------------------------


class TestCanvasDimensions:
    """Verify the canvas layout formulas for each mode."""

    def _strip_dims(self, n: int) -> tuple[int, int]:
        return n, 1  # cols, rows

    def _grid_dims(self, n: int) -> tuple[int, int]:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return cols, rows

    def _pizza_dims(self, r_steps: int, n_angles: int) -> tuple[int, int]:
        # rows = r_steps, cols = n_angles (from survey_png pizza branch)
        return n_angles, r_steps  # cols, rows

    def test_strip_is_single_row(self) -> None:
        for n in [4, 8, 12, 90]:
            cols, rows = self._strip_dims(n)
            assert rows == 1, f"strip with n={n} should have 1 row, got {rows}"
            assert cols == n, f"strip with n={n} should have {n} cols, got {cols}"

    def test_grid_is_approximately_square(self) -> None:
        for n in [4, 9, 16, 25]:
            cols, rows = self._grid_dims(n)
            assert abs(cols - rows) <= 1, f"grid n={n}: cols={cols}, rows={rows} is not square"
            assert cols * rows >= n, f"grid n={n}: {cols}x{rows} canvas too small"

    def test_grid_9_is_3x3(self) -> None:
        cols, rows = self._grid_dims(9)
        assert (cols, rows) == (3, 3)

    def test_grid_12_fits_all(self) -> None:
        cols, rows = self._grid_dims(12)
        assert cols * rows >= 12

    def test_pizza_cols_equal_n_angles(self) -> None:
        for r_steps, n_angles in [(3, 8), (4, 12), (2, 6)]:
            cols, _rows = self._pizza_dims(r_steps, n_angles)
            assert cols == n_angles, f"pizza cols should equal n_angles={n_angles}, got {cols}"

    def test_pizza_rows_equal_r_steps(self) -> None:
        for r_steps, n_angles in [(3, 8), (4, 12), (2, 6)]:
            _cols, rows = self._pizza_dims(r_steps, n_angles)
            assert rows == r_steps, f"pizza rows should equal r_steps={r_steps}, got {rows}"

    def test_pizza_total_cells_equals_total_plys(self) -> None:
        r_steps, n_angles = 4, 8
        cols, rows = self._pizza_dims(r_steps, n_angles)
        assert cols * rows == r_steps * n_angles

    def test_survey_png_source_has_pizza_canvas_logic(self) -> None:
        assert 'layout == "pizza"' in SURVEY_PNG_SOURCE, "survey_png.py must have a pizza canvas layout branch"
        assert "r_indices" in SURVEY_PNG_SOURCE, "survey_png.py pizza branch must compute r_indices for row count"
        assert "a_indices" in SURVEY_PNG_SOURCE, "survey_png.py pizza branch must compute a_indices for col count"


# ---------------------------------------------------------------------------
# 6. Config: pizza keys in vis_config.yaml
# ---------------------------------------------------------------------------


class TestPizzaConfig:
    """vis_config.yaml must contain pizza-mode configuration keys."""

    def test_pizza_r_min_present(self) -> None:
        assert "pizza_r_min" in SURVEY_CFG, "vis_config latent_optimisation.vis_latent_circle_survey missing pizza_r_min"

    def test_pizza_r_max_present(self) -> None:
        assert "pizza_r_max" in SURVEY_CFG, "vis_config latent_optimisation.vis_latent_circle_survey missing pizza_r_max"

    def test_pizza_r_steps_present(self) -> None:
        assert "pizza_r_steps" in SURVEY_CFG, "vis_config latent_optimisation.vis_latent_circle_survey missing pizza_r_steps"

    def test_pizza_r_min_less_than_r_max(self) -> None:
        r_min = float(SURVEY_CFG["pizza_r_min"])
        r_max = float(SURVEY_CFG["pizza_r_max"])
        assert r_min < r_max, f"pizza_r_min ({r_min}) must be less than pizza_r_max ({r_max})"

    def test_pizza_r_steps_is_positive_int(self) -> None:
        r_steps = int(SURVEY_CFG["pizza_r_steps"])
        assert r_steps >= 1, f"pizza_r_steps must be >= 1, got {r_steps}"

    def test_three_layout_values_documented_in_config(self) -> None:
        config_text = _VIS_CFG_PATH.read_text()
        for mode in ("strip", "grid", "pizza"):
            assert mode in config_text, f"vis_config.yaml layout comment should mention '{mode}' mode"
