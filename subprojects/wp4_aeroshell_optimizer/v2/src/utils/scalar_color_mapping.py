"""Scalar-to-RGBA mapping utilities for unified visualization exports.

This module provides deterministic, dependency-light mappings for:
1. Continuous scalar fields (e.g., pressure/error/gradient magnitude).
2. Boolean masks (e.g., solar selection).
3. Matplotlib-colormap-based RGB mapping (Plan 012 / M2).
"""

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
    """Map continuous scalar values to per-value RGBA colors.

    The mapping uses a deterministic blue-to-red gradient:
    1. Low endpoint -> blue.
    2. High endpoint -> red.

    Args:
        values: Scalar values as a NumPy array of any shape.
        min_val: Lower normalization bound. If None, inferred from finite values.
        max_val: Upper normalization bound. If None, inferred from finite values.
        alpha: Global alpha in [0, 1] applied to finite mapped values.
        low_rgba: RGBA color used at the low normalization endpoint.
            Defaults to blue ``(0, 0, 255, 255)`` to preserve legacy behavior.
        high_rgba: RGBA color used at the high normalization endpoint.
            Defaults to red ``(255, 0, 0, 255)`` to preserve legacy behavior.
        nan_rgba: RGBA fallback used for NaN or non-finite entries.

    Returns:
        A ``(N, 4)`` uint8 RGBA array where ``N = values.size``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Invalid alpha={alpha}. Expected a value in [0, 1].")

    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    finite_mask = np.isfinite(flat)
    if not np.any(finite_mask):
        raise ValueError("No finite scalar values were provided. Provide at least one finite value to map.")

    finite = flat[finite_mask]
    v_min = float(np.min(finite)) if min_val is None else float(min_val)
    v_max = float(np.max(finite)) if max_val is None else float(max_val)

    if v_max < v_min:
        raise ValueError(f"Invalid normalization bounds: max_val ({v_max}) must be >= min_val ({v_min}).")

    denom = v_max - v_min
    rgba = np.zeros((flat.size, 4), dtype=np.uint8)
    low_color = _as_uint8_rgba((0, 0, 255, 255) if low_rgba is None else low_rgba, name="low_rgba")
    high_color = _as_uint8_rgba((255, 0, 0, 255) if high_rgba is None else high_rgba, name="high_rgba")

    normalized = np.full(finite.shape, 0.5, dtype=np.float32) if denom == 0.0 else (finite - v_min) / denom
    normalized = np.clip(normalized, 0.0, 1.0)

    rgb_low = low_color[:3].astype(np.float32)
    rgb_high = high_color[:3].astype(np.float32)
    mapped_rgb = rgb_low[None, :] + normalized[:, None] * (rgb_high - rgb_low)[None, :]
    rgba[finite_mask, :3] = np.round(mapped_rgb).astype(np.uint8)
    rgba[finite_mask, 3] = round(alpha * 255.0)

    fallback = _as_uint8_rgba(nan_rgba, name="nan_rgba")
    rgba[~finite_mask] = fallback
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
        A ``(N, 4)`` uint8 RGBA array where ``N = mask.size``.
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

    Non-finite values are assigned a neutral grey (0.5, 0.5, 0.5).

    Args:
        values: Scalar values as a NumPy array of any shape.  Flattened to 1-D
            internally; the caller is responsible for passing a 1-D array.
        colormap: A valid matplotlib colormap name (e.g. ``"coolwarm"``,
            ``"viridis"``). Raises ``KeyError`` if the name is not recognised.
        vmin: Fixed lower bound for colour normalisation. If ``None``, the
            minimum over finite values is used.
        vmax: Fixed upper bound for colour normalisation. If ``None``, the
            maximum over finite values is used.

    Returns:
        A ``(N, 3)`` float32 array with values in ``[0, 1]``.

    Raises:
        KeyError: If ``colormap`` is not a registered matplotlib colormap.
        ValueError: If ``colormap`` is an empty string or otherwise invalid.
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

        cmap = matplotlib.colormaps[colormap]  # raises KeyError for unknown names
        norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max, clip=True)
        mapped = cmap(norm(finite))  # (M, 4) float64
        rgb[finite_mask] = mapped[:, :3].astype(np.float32)

    return rgb
