"""Regression tests for Fluent orientation handling."""

import numpy as np

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.dataset.euler.process_fluent_pressure import (
    apply_surface_orientation,
)


def test_apply_surface_orientation_preserves_original_frame_when_disabled() -> None:
    """Disabling canonicalization must leave positions and normals unchanged."""
    pos = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)
    normals = np.array([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]], dtype=np.float32)

    pos_out, normals_out = apply_surface_orientation(pos.copy(), normals.copy(), canonicalize_orientation=False)

    np.testing.assert_array_equal(pos_out, pos)
    np.testing.assert_array_equal(normals_out, normals)


def test_apply_surface_orientation_flips_only_x_when_enabled() -> None:
    """Canonicalization must negate x only, matching the current Fluent export convention."""
    pos = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)
    normals = np.array([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]], dtype=np.float32)

    pos_out, normals_out = apply_surface_orientation(pos.copy(), normals.copy(), canonicalize_orientation=True)

    expected_pos = np.array([[-1.0, 2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
    expected_normals = np.array([[-0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype=np.float32)

    np.testing.assert_array_equal(pos_out, expected_pos)
    np.testing.assert_array_equal(normals_out, expected_normals)
