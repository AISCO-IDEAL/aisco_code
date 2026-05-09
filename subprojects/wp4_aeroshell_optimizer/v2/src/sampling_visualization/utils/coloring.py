from __future__ import annotations

from collections.abc import Iterable

import matplotlib
import matplotlib.colors
import numpy as np


def _as_uint8_rgba(color: Iterable[int], *, name: str) -> np.ndarray:
    rgba = np.asarray(list(color), dtype=np.int64)
    if rgba.shape != (4,):
        raise ValueError(f"Invalid {name}: expected 4 channels (R,G,B,A), got shape {rgba.shape}.")
    if np.any(rgba < 0) or np.any(rgba > 255):
        raise ValueError(f"Invalid {name}: channel values must be in [0, 255].")
    return rgba.astype(np.uint8)


def map_scalar_to_rgba(
    values: np.ndarray,
    *,
    min_val: float | None = None,
    max_val: float | None = None,
    alpha: float = 1.0,
    low_rgba: tuple[int, int, int, int] | None = None,
    high_rgba: tuple[int, int, int, int] | None = None,
    nan_rgba: tuple[int, int, int, int] = (127, 127, 127, 255),
) -> np.ndarray:
    """Map continuous scalar values to per-value RGBA colors using a blue-to-red gradient.

    Args:
        values: Scalar values as a NumPy array of any shape.
        min_val: Lower normalisation bound. Inferred from finite values if None.
        max_val: Upper normalisation bound. Inferred from finite values if None.
        alpha: Global alpha in [0, 1] applied to finite mapped values.
        low_rgba: RGBA at the low endpoint. Defaults to blue (0, 0, 255, 255).
        high_rgba: RGBA at the high endpoint. Defaults to red (255, 0, 0, 255).
        nan_rgba: RGBA fallback for NaN or non-finite entries.

    Returns:
        (N, 4) uint8 RGBA array where N = values.size.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Invalid alpha={alpha}. Expected a value in [0, 1].")

    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    finite_mask = np.isfinite(flat)
    if not np.any(finite_mask):
        raise ValueError("No finite scalar values were provided.")

    finite = flat[finite_mask]
    v_min = float(np.min(finite)) if min_val is None else float(min_val)
    v_max = float(np.max(finite)) if max_val is None else float(max_val)
    if v_max < v_min:
        raise ValueError(f"max_val ({v_max}) must be >= min_val ({v_min}).")

    denom = v_max - v_min
    rgba = np.zeros((flat.size, 4), dtype=np.uint8)
    low_color = _as_uint8_rgba((0, 0, 255, 255) if low_rgba is None else low_rgba, name="low_rgba")
    high_color = _as_uint8_rgba((255, 0, 0, 255) if high_rgba is None else high_rgba, name="high_rgba")

    normalized = np.full(finite.shape, 0.5, dtype=np.float32) if denom == 0.0 else np.clip((finite - v_min) / denom, 0.0, 1.0)
    mapped_rgb = low_color[:3].astype(np.float32) + normalized[:, None] * (high_color[:3] - low_color[:3]).astype(np.float32)
    rgba[finite_mask, :3] = np.round(mapped_rgb).astype(np.uint8)
    rgba[finite_mask, 3] = round(alpha * 255.0)
    rgba[~finite_mask] = _as_uint8_rgba(nan_rgba, name="nan_rgba")
    return rgba


def map_boolean_mask_to_rgba(
    mask: np.ndarray,
    *,
    true_rgba: tuple[int, int, int, int] = (51, 51, 204, 255),
    false_rgba: tuple[int, int, int, int] = (255, 128, 0, 255),
) -> np.ndarray:
    """Map a boolean mask to per-entry RGBA values.

    Args:
        mask: Boolean-like mask array. Non-zero values are treated as True.
        true_rgba: RGBA assigned to True entries.
        false_rgba: RGBA assigned to False entries.

    Returns:
        (N, 4) uint8 RGBA array where N = mask.size.
    """
    flat_mask = np.asarray(mask).reshape(-1).astype(bool)
    rgba_true = _as_uint8_rgba(true_rgba, name="true_rgba")
    rgba_false = _as_uint8_rgba(false_rgba, name="false_rgba")
    out = np.tile(rgba_false, (flat_mask.size, 1))
    out[flat_mask] = rgba_true
    return out.astype(np.uint8)


def map_scalar_to_rgb_mpl(
    values: np.ndarray,
    colormap: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Map scalar values to RGB using a named matplotlib colormap.

    Args:
        values: Scalar values as a NumPy array of any shape.
        colormap: A valid matplotlib colormap name (e.g. "coolwarm", "viridis").
        vmin: Fixed lower bound for colour normalisation. Inferred if None.
        vmax: Fixed upper bound for colour normalisation. Inferred if None.

    Returns:
        (N, 3) float32 array with values in [0, 1]. Non-finite values → grey (0.5).

    Raises:
        KeyError: If colormap is not a registered matplotlib colormap.
        ValueError: If colormap is an empty string.
    """
    if not colormap:
        raise ValueError("colormap must be a non-empty string.")

    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    finite_mask = np.isfinite(flat)
    rgb = np.full((flat.size, 3), 0.5, dtype=np.float32)

    if np.any(finite_mask):
        finite = flat[finite_mask]
        v_min = float(np.min(finite)) if vmin is None else float(vmin)
        v_max = float(np.max(finite)) if vmax is None else float(vmax)
        cmap = matplotlib.colormaps[colormap]
        norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max, clip=True)
        rgb[finite_mask] = cmap(norm(finite))[:, :3].astype(np.float32)

    return rgb


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
    true_rgba = tuple(int(c * 255) for c in true_rgb) + (255,)  # noqa: RUF005 : Weird unpacking thing
    false_rgba = tuple(int(c * 255) for c in false_rgb) + (255,)  # noqa: RUF005
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
