from __future__ import annotations

from typing import Any

import numpy as np


class NullVisManager:
    """Drop-in replacement for CentralVisManagerV2 that does nothing.

    Use via ``vis_manager: null`` in general_config.yaml for dry-run training
    with zero visualization overhead. Every named method is a silent no-op.
    """

    @property
    def latent_pretraining_cfg(self) -> dict[str, Any]:
        """Return empty dict — NullVisManager disables all vis."""
        return {}

    @property
    def latent_optimisation_cfg(self) -> dict[str, Any]:
        """Return empty dict — NullVisManager disables all vis."""
        return {}

    def vis_latent_circle_survey(self, *args: Any, **kwargs: Any) -> None:
        pass

    def vis_mesh_shape(self, epoch: int, verts: Any, faces: Any) -> None:
        pass

    def vis_aero_cp_field(self, epoch: int, verts: Any, faces: Any, cp_values: Any) -> None:
        pass

    def vis_aero_sdf_influence(self, epoch: int, method: str, variant: str, verts: Any, faces: Any, values: Any) -> None:
        pass

    def vis_solar_selection(self, epoch: int, verts: Any, faces: Any, weights: Any) -> None:
        pass

    def vis_solar_shading(self, epoch: int, verts: Any, faces: Any, shading: Any) -> None:
        pass

    def vis_solar_penalty(self, epoch: int, verts: Any, faces: Any, penalty: Any) -> None:
        pass

    def vis_solar_numerical_factor(self, epoch: int, verts: Any, faces: Any, numerical_factors: Any) -> None:
        pass

    def vis_solar_quality(self, epoch: int, verts: Any, faces: Any, quality: Any) -> None:
        pass

    def vis_solar_receiver_mask(self, epoch: int, verts: Any, faces: Any, receiver_mask: Any) -> None:
        pass

    def vis_solar_vision_cone_mask(self, epoch: int, verts: Any, faces: Any, mask: Any) -> None:
        pass

    def vis_bb_constraint(self, epoch: int, verts: Any, distances: Any) -> None:
        pass

    def vis_mesh_gradient(
        self,
        epoch: int,
        loss_key: str,
        verts: Any,
        faces: Any,
        dL_dv: Any,
        true_scale: bool = True,
    ) -> None:
        pass

    def vis_latent_sampling(
        self,
        epoch: int,
        points: Any,
        slot_idx: int | None = None,
        slot_name: str = "",
        secondary_points: Any | None = None,
        tertiary_points: Any | None = None,
        mesh_verts: np.ndarray | None = None,
        mesh_faces: np.ndarray | None = None,
    ) -> None:
        pass

    def set_assembly_meshes(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def make_vis_extractor(self, _device: str = "cpu") -> None:
        return None

    def is_enabled(self, _visual_type: str) -> bool:
        return False

    def is_due(self, _visual_type: str, _epoch: int) -> bool:
        return False

    def vis_mesh_assembly_v4(self, epoch: int, sdf_verts: Any, sdf_faces: Any) -> None:
        pass

    def save_ply(self, visual_type: str, epoch: int, *_args: Any, **_kwargs: Any) -> None:
        pass

    def save_ply_points(self, visual_type: str, epoch: int, *_args: Any, **_kwargs: Any) -> None:
        pass

    def set_active_slot(self, slot_idx: int | None) -> None:
        pass

    def finalize(self) -> None:
        pass

    def __del__(self) -> None:
        pass
