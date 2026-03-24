"""Differentiable visibility utilities for solar shading analysis."""

from subprojects.wp3_solar_estimator.raytracing.diffable.version1.mesh_ray_visibility import (
	mesh_visibility as mesh_visibility,
)
from subprojects.wp3_solar_estimator.raytracing.diffable.version1.sdf_ray_visibility import (
	sdf_visibility as sdf_visibility,
)
from subprojects.wp3_solar_estimator.raytracing.diffable.version2.mesh_ray_visibility import (
	mesh_visibility as mesh_visibility_v2,
)
from subprojects.wp3_solar_estimator.raytracing.diffable.version2.sdf_ray_visibility import (
	sdf_visibility as sdf_visibility_v2,
)

__all__ = ["mesh_visibility", "mesh_visibility_v2", "sdf_visibility", "sdf_visibility_v2"]
