"""
Solar SRRT wrapper: single fixed sun direction with mesh raytracing (v3).

Returns solar power in watts. Gradients are preserved through the mesh_sdf surrogate path.
Raytracing visibility is computed per-face using differentiable Möller-Trumbore intersection.
"""

from __future__ import annotations

from dataclasses import replace as _replace

import torch

from subprojects.wp3_solar_estimator.raytracing.diffable.version4.mesh_ray_visibility_v4 import mesh_visibility_v4
from subprojects.wp3_solar_estimator.rollout_losses.SRRT_SolarLoss import solar_loss_srrt
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import NullVisManager
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor, load_mesh_extraction_config


class SolarSRRT(ParentLoss):
    """Solar SRRT wrapper — single fixed sun direction, mesh_sdf extraction, mesh raytracing v3."""

    def _setup(self) -> None:
        gen_cfg = load_config(v="v2", cfg_name="general_config.yaml")
        cfg_path = gen_cfg["active_configs"]["solar"]
        cfg = load_config(v="v2", cfg_name=cfg_path)["SolarLossConfigSRRT"]

        self.area_budget_mm2 = float(cfg["area_budget_mm2"])
        self.sun_dir = torch.tensor(cfg["sun_dir"], dtype=torch.float32)
        self.effective_irradiance_W_m2 = float(cfg.get("effective_irradiance_W_m2", 250.0))
        self.sigmoid_sharpness = float(cfg.get("sigmoid_sharpness", 10.0))
        self.soft_mask_selection_threshold = float(cfg.get("soft_mask_selection_threshold", 0.5))
        self.soft_mask_full_selection_threshold = float(cfg.get("soft_mask_full_selection_threshold", 0.95))
        self.shading_penalty_weight = float(cfg.get("shading_penalty_weight", 0.0))
        self.kappa = float(cfg.get("kappa", 1.0))
        self._lambda = float(cfg.get("_lambda", 1.0))
        self.epsilon = float(cfg.get("eps", 1e-12))
        self.use_activation_checkpointing = bool(cfg.get("use_activation_checkpointing", True))
        self.smooth_gradient_iters = int(cfg.get("smooth_gradient_iters", 0))

        mesh_cfg_extract = load_mesh_extraction_config()
        mesh_cfg_extract = _replace(
            mesh_cfg_extract,
            device=self.device,
            bbox=cfg["bbox"],
            isovalue=float(cfg.get("isovalue", 0.0)),
        )
        self.extractor = MeshSDFExtractor(config=mesh_cfg_extract)
        self.extractor.set_grid(cfg["bbox"], mesh_cfg_extract.base_resolution)

        # Load raytracing config from raytracing_config.yaml (single source of truth)
        raytracing_version = cfg.get("raytracing_version", "v3")
        rt_cfg = load_config(v="v2", cfg_name="solar_configs/raytracing_config.yaml")
        key = f"RaytracingConfig{raytracing_version.upper()}"
        if key not in rt_cfg:
            raise ValueError(f"Unknown raytracing_version '{raytracing_version}' in raytracing_config.yaml")
        self.raytracing_cfg = dict(rt_cfg[key])
        self._pending_diagnostics: dict = {}

        self._mesh_visibility_fn = mesh_visibility_v4 if raytracing_version == "v4" else None

    def _loss(self, model, skip_telemetry: bool = False) -> torch.Tensor:
        vis_manager = self._vis_manager if self._vis_manager is not None else NullVisManager()
        sun_dir = self.sun_dir.to(self.device)

        def loss_fn(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
            loss, diagnostics = solar_loss_srrt(
                vertices,
                faces,
                sun_dir,
                area_budget_mm2=self.area_budget_mm2,
                raytracing_cfg=self.raytracing_cfg,
                effective_irradiance_W_m2=self.effective_irradiance_W_m2,
                sigmoid_sharpness=self.sigmoid_sharpness,
                soft_mask_selection_threshold=self.soft_mask_selection_threshold,
                soft_mask_full_selection_threshold=self.soft_mask_full_selection_threshold,
                shading_penalty_weight=self.shading_penalty_weight,
                kappa=self.kappa,
                _lambda=self._lambda,
                epsilon=self.epsilon,
                use_activation_checkpointing=self.use_activation_checkpointing,
                vis_manager=vis_manager if not skip_telemetry else NullVisManager(),
                epoch_name=self.epoch_name,
                mesh_visibility_fn=self._mesh_visibility_fn,
            )
            self._pending_diagnostics = diagnostics
            return loss

        _, _, loss, surrogate_loss, _ = self.extractor.extract_and_compute_loss(
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
