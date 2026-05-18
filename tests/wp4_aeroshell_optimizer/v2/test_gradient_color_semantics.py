from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import matplotlib.colors
import numpy as np


def _load_gradient_semantics_module():
    root = Path(__file__).parents[3]
    module_path = (
        root
        / "subprojects"
        / "wp4_aeroshell_optimizer"
        / "v2"
        / "src"
        / "sampling_visualization"
        / "utils"
        / "gradient_color_semantics.py"
    )
    spec = importlib.util.spec_from_file_location("gradient_color_semantics", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rdylgn_rgb(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    cmap = matplotlib.colormaps["RdYlGn"]
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return cmap(norm(values))[:, :3]


def _assert_red(rgb: np.ndarray) -> None:
    assert rgb[0] > rgb[1]
    assert rgb[0] > rgb[2]


def _assert_green(rgb: np.ndarray) -> None:
    assert rgb[1] > rgb[0]
    assert rgb[1] > rgb[2]


def _assert_yellow(rgb: np.ndarray) -> None:
    assert rgb[0] > 0.8
    assert rgb[1] > 0.8
    assert rgb[2] < 0.8


def test_mesh_gradient_colors_optimizer_update_direction() -> None:
    semantics = _load_gradient_semantics_module()
    normals = np.tile([0.0, 0.0, 1.0], (3, 1))

    outward_raw_gradient = semantics.optimizer_update_normal_scalar(normals, np.tile([0.0, 0.0, 1.0], (3, 1)))
    inward_raw_gradient = semantics.optimizer_update_normal_scalar(normals, np.tile([0.0, 0.0, -1.0], (3, 1)))
    zero_gradient = semantics.optimizer_update_normal_scalar(normals, np.zeros((3, 3)))

    rgb = _rdylgn_rgb(
        np.array([outward_raw_gradient[0], inward_raw_gradient[0], zero_gradient[0]], dtype=np.float32),
        vmin=-1.0,
        vmax=1.0,
    )

    _assert_red(rgb[0])
    _assert_green(rgb[1])
    _assert_yellow(rgb[2])


def test_mesh_gradient_relative_scale_keeps_zero_neutral() -> None:
    semantics = _load_gradient_semantics_module()

    normalized = semantics.normalize_signed_to_unit_interval(np.array([-2.0, 0.0, 4.0], dtype=np.float32))
    rgb = _rdylgn_rgb(normalized, vmin=0.0, vmax=1.0)

    _assert_red(rgb[0])
    _assert_yellow(rgb[1])
    _assert_green(rgb[2])


def test_sdf_influence_colors_optimizer_update_displacement() -> None:
    semantics = _load_gradient_semantics_module()

    displacement = semantics.optimizer_update_sdf_displacement(np.array([1.0, -1.0, 0.0], dtype=np.float32))
    rgb = _rdylgn_rgb(displacement, vmin=-1.0, vmax=1.0)

    _assert_red(rgb[0])
    _assert_green(rgb[1])
    _assert_yellow(rgb[2])
