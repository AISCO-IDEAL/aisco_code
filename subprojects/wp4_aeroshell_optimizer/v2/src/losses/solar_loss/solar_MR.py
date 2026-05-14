"""
Solar MR wrapper: K solar directions averaged, no raytracing.

Returns solar power in watts. Gradients are preserved through the mesh_sdf surrogate path.
Solar vectors are loaded once in _setup from RouteV3 (race mode) or a custom yaml vector set.
"""

from __future__ import annotations

from dataclasses import replace as _replace

import torch

from subprojects.wp3_solar_estimator.rollout_losses.MR_SolarLoss import solar_loss_mr
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import NullVisManager
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor, load_mesh_extraction_config


class SolarMR(ParentLoss):
    """Solar MR wrapper — K solar directions, mesh_sdf extraction, no raytracing."""

    def _setup(self) -> None:
        gen_cfg = load_config(v="v2", cfg_name="general_config.yaml")
        cfg_path = gen_cfg["active_configs"]["solar"]
        cfg = load_config(v="v2", cfg_name=cfg_path)["SolarLossConfigMR"]

        self.area_budget_mm2 = float(cfg["area_budget_mm2"])
        self.effective_irradiance_W_m2 = float(cfg.get("effective_irradiance_W_m2", 250.0))
        self.sigmoid_sharpness = float(cfg.get("sigmoid_sharpness", 10.0))
        self.soft_mask_selection_threshold = float(cfg.get("soft_mask_selection_threshold", 0.5))
        self.soft_mask_full_selection_threshold = float(cfg.get("soft_mask_full_selection_threshold", 0.95))
        self.epsilon = float(cfg.get("eps", 1e-12))
        self.smooth_gradient_iters = int(cfg.get("smooth_gradient_iters", 0))

        source = cfg.get("solar_vector_source", "route_v3")
        if source == "route_v3":
            from subprojects.wp3_solar_estimator.australia.route_v3 import RouteV3  # noqa: PLC0415 , avoid unnecessary import

            route_mode = cfg.get("route_mode", "race")
            route = RouteV3(cfg_path=cfg.get("route_cfg_path", "solar_configs/australia.yaml"), mode=route_mode)
            self.solar_vectors = route.solar_vectors(subsample=cfg.get("subsample_n") or None)
        elif source == "custom":
            test_cfg = load_config(v="v2", cfg_name="solar_configs/test_solar_vectors.yaml")
            self.solar_vectors = torch.tensor(test_cfg[cfg["custom_vectors_key"]], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown solar_vector_source: {source!r}")

        mesh_cfg = load_mesh_extraction_config()
        mesh_cfg = _replace(
            mesh_cfg,
            device=self.device,
            bbox=cfg["bbox"],
            isovalue=float(cfg.get("isovalue", 0.0)),
        )
        self.extractor = MeshSDFExtractor(config=mesh_cfg)
        self.extractor.set_grid(cfg["bbox"], mesh_cfg.base_resolution)

        self._pending_diagnostics: dict = {}

    def _loss(self, model, skip_telemetry: bool = False) -> torch.Tensor:
        vis_manager = self._vis_manager if self._vis_manager is not None else NullVisManager()
        solar_vectors = self.solar_vectors.to(self.device)

        def loss_fn(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
            loss, diagnostics = solar_loss_mr(
                vertices,
                faces,
                solar_vectors,
                area_budget_mm2=self.area_budget_mm2,
                effective_irradiance_W_m2=self.effective_irradiance_W_m2,
                sigmoid_sharpness=self.sigmoid_sharpness,
                soft_mask_selection_threshold=self.soft_mask_selection_threshold,
                soft_mask_full_selection_threshold=self.soft_mask_full_selection_threshold,
                epsilon=self.epsilon,
                vis_manager=vis_manager if not skip_telemetry else NullVisManager(),
                epoch_name=self.epoch_name,
            )
            self._pending_diagnostics = diagnostics
            return loss

        _vertices, _faces, loss, surrogate_loss, _dL_dv = self.extractor.extract_and_compute_loss(
            sdf_model=model,
            loss_fn=loss_fn,
            enable_backward=True,
            smooth_vertex_gradients_iters=self.smooth_gradient_iters,
        )

        with torch.no_grad():
            self._last_diagnostics = {
                "loss_value_watts": loss.item(),
                **self._pending_diagnostics,
            }

        return surrogate_loss + (loss - surrogate_loss).detach()
