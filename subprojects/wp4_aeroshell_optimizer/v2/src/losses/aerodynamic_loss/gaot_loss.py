from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.model_loader import ModelLoader as ModelLoader2
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.model_loader import ModelLoader as ModelLoader3
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT_working.model_loader import ModelLoader as ModelLoader1
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.device_detection import is_local_machine

if TYPE_CHECKING:
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import NullVisManager
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import CentralVisManagerV2


class GaotLoss(ParentLoss):
    # Set externally by TotalLoss after _init_active_losses(); propagated to
    # self.gaot before each forward call so ModelLoader can route its
    # Cp-field visualisation through vis_* methods.
    _vis_manager: CentralVisManagerV2 | NullVisManager | None = None

    def _setup(self) -> None:
        version = self.losses_config["cfd"]["version"] if self.losses_config["cfd"]["version"] is not None else 1

        self.gaot: ModelLoader1 | ModelLoader2 | ModelLoader3
        if version == 1:
            warnings.warn("You are using an old Version of the GAOT", stacklevel=2)
            self.gaot = ModelLoader1(cfd_config=self.losses_config["cfd"], device=self.device)
        elif version == 2:
            warnings.warn("You are using an old Version of the GAOT", stacklevel=2)
            self.gaot = ModelLoader2(cfd_config=self.losses_config["cfd"], device=self.device)
        elif version == 3:
            if is_local_machine():
                self.gaot = ModelLoader3(cfd_config=self.losses_config["cfd_local"], device=self.device)
            else:
                self.gaot = ModelLoader3(cfd_config=self.losses_config["cfd"], device=self.device)
        else:
            raise ValueError(f"Gaot Loss Version {version} is not implemented")

    def _loss(self, model) -> torch.Tensor:
        # Propagate CentralVisManager to the underlying model loader so it can
        # route its Cp-field visualisation through save_ply() instead of wandb PNG.
        if self._vis_manager is not None:
            # Propagate to the underlying model loader (GAOT3 only in practice;
            # setattr avoids mypy union-attr error on ModelLoader1/2).
            setattr(self.gaot, "vis_manager", self._vis_manager)  # noqa: B010
        if isinstance(self.gaot, ModelLoader3):
            learning_rate = self.training_config.get("training_params", {}).get("learning_rate", 1e-3)
            self.gaot.sdf_influence_learning_rate = float(learning_rate)
        result = self.gaot.forward(sdf_model=model, epoch_name=self.epoch_name)
        # Propagate diagnostics from the underlying GAOT model loader
        self._last_diagnostics = getattr(self.gaot, "_last_diagnostics", {})
        return result
