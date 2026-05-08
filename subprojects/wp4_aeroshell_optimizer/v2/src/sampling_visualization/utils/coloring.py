from __future__ import annotations

import numpy as np

from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.scalar_color_mapping import (
    map_boolean_mask_to_rgba,
    map_scalar_to_rgb_mpl,
)


def scalar_to_rgb(
    values: np.ndarray,
    colormap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Map scalar values to RGB using a named matplotlib colormap.

    Args:
        values: Scalar values of any shape — flattened to 1-D internally.
        colormap: Named matplotlib colormap (e.g. "coolwarm", "viridis").
        vmin: Fixed lower bound for normalisation. Inferred from data if None.
        vmax: Fixed upper bound for normalisation. Inferred from data if None.

    Returns:
        (N, 3) float32 RGB array in [0, 1]. Non-finite values → neutral grey.
    """
    return map_scalar_to_rgb_mpl(values, colormap, vmin=vmin, vmax=vmax)


def boolean_to_rgb(
    mask: np.ndarray,
    true_rgb: tuple[float, float, float] = (0.2, 0.2, 0.8),
    false_rgb: tuple[float, float, float] = (1.0, 0.5, 0.0),
) -> np.ndarray:
    """Map a boolean mask to per-entry RGB colors.

    Args:
        mask: Boolean-like array of any shape.
        true_rgb: RGB color for True entries, float in [0, 1].
        false_rgb: RGB color for False entries, float in [0, 1].

    Returns:
        (N, 3) float32 RGB array in [0, 1].
    """
    true_rgba = tuple(int(c * 255) for c in true_rgb) + (255,)
    false_rgba = tuple(int(c * 255) for c in false_rgb) + (255,)
    rgba = map_boolean_mask_to_rgba(mask, true_rgba=true_rgba, false_rgba=false_rgba)  # type: ignore[arg-type]
    return rgba[:, :3].astype(np.float32) / 255.0


def solid_rgb(n_verts: int, rgb: tuple[float, float, float]) -> np.ndarray:
    """Return a uniform color array for all vertices.

    Args:
        n_verts: Number of vertices.
        rgb: RGB color, float in [0, 1].

    Returns:
        (N, 3) float32 array with the same color repeated for every vertex.
    """
    return np.tile(np.array(rgb, dtype=np.float32), (n_verts, 1))
