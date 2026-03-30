"""Version 3 SDF visibility wrapper.

Currently delegates to v2 implementation while v3 mesh visibility evolves.
Keeping this wrapper allows unified raytracing_version="v3" selection in the
solar loss pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import torch

from subprojects.wp3_solar_estimator.raytracing.diffable.version2.sdf_ray_visibility_v2 import sdf_visibility_v2


# Wrapper since V3 of sdf visibility does not work yet
def sdf_visibility_v3(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    vertices: torch.Tensor,
    faces: torch.Tensor,
    sun_dir: torch.Tensor,
    *,
    normal_source: Literal["mesh", "sdf"] = "mesh",
    # Sigmoid method params
    sdf_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Delegate to v2 SDF visibility implementation."""

    return sdf_visibility_v2(
        sdf_fn=sdf_fn,
        vertices=vertices,
        faces=faces,
        sun_dir=sun_dir,
        normal_source=normal_source,
        sdf_cfg=sdf_cfg,
    )
