from __future__ import annotations

import numpy as np


def optimizer_update_normal_scalar(vertex_normals: np.ndarray, dL_dv: np.ndarray) -> np.ndarray:
    """Return signed normal displacement for gradient descent updates."""
    raw_dot = np.einsum("ij,ij->i", vertex_normals, dL_dv).astype(np.float32)
    return np.nan_to_num(-raw_dot, nan=0.0, posinf=0.0, neginf=0.0)


def optimizer_update_sdf_displacement(values: np.ndarray) -> np.ndarray:
    """Convert positive-outside SDF deltas to signed surface displacement."""
    return -np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize_signed_to_unit_interval(values: np.ndarray) -> np.ndarray:
    """Map signed values to [0, 1] with zero fixed at 0.5."""
    values = np.asarray(values, dtype=np.float32)
    value_min = float(np.min(values)) if values.size else 0.0
    value_max = float(np.max(values)) if values.size else 0.0
    normalized = np.full_like(values, 0.5, dtype=np.float32)
    if value_max > 0.0:
        normalized = np.where(values > 0.0, 0.5 + 0.5 * values / value_max, normalized)
    if value_min < 0.0:
        normalized = np.where(values < 0.0, 0.5 + 0.5 * values / (-value_min), normalized)
    return normalized
