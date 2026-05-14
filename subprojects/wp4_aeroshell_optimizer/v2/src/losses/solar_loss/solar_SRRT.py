from __future__ import annotations

from dataclasses import replace as _replace

import torch

from subprojects.wp3_solar_estimator.rollout_losses.SRRT_SolarLoss import solar_loss_srrt
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor, load_mesh_extraction_config


class SolarSRRT(ParentLoss):
    def _setup(self) -> None:
        gen_cfg = load_config(v="v2", cfg_name="general_config.yaml")
        cfg_path = gen_cfg["active_configs"]["solar"]
        cfg = load_config(v="v2", cfg_name=cfg_path)["SolarLossConfigSRRT"]

        mesh_extraction_cfg = load_mesh_extraction_config()
        mesh_extraction_cfg = _replace(
            mesh_extraction_cfg,
            device=self.device,
            bbox=cfg["bbox"],
            isovalue=float(cfg.get("isovalue", 0.0)),
        )
        self.mesh_extractor = MeshSDFExtractor(config=mesh_extraction_cfg)
        self.mesh_extractor.set_grid(cfg["bbox"], int(cfg["resolution"]))

        self.area_budget_mm2: float = float(cfg["area_budget_mm2"])
        self.effective_irradiance_W_m2: float = float(cfg.get("effective_irradiance_W_m2", 250.0))
        self.sigmoid_sharpness: float = float(cfg.get("sigmoid_sharpness", 10.0))
        self.shading_penalty_weight: float = float(cfg.get("shading_penalty_weight", 0.0))
        self.kappa: float = float(cfg.get("kappa", 1.0))
        self._lambda_val: float = float(cfg.get("_lambda", 1.0))
        self.epsilon: float = float(cfg.get("eps", 1e-12))
        self.use_activation_checkpointing: bool = bool(cfg.get("use_activation_checkpointing", True))
        self.smooth_gradient_iters: int = int(cfg.get("smooth_gradient_iters", 0))
        self.sun_dir = torch.tensor(cfg["sun_dir"], dtype=torch.float32)

        raytracing_version: str = cfg.get("raytracing_version", "v3")
        rt_all = load_config(v="v2", cfg_name="solar_configs/raytracing_config.yaml")
        self.raytracing_cfg: dict = rt_all[f"RaytracingConfig{raytracing_version.upper()}"]

    def _loss(self, model, skip_telemetry: bool = False) -> torch.Tensor:
        vertices, faces = self.mesh_extractor.extract_mesh(model)
        sun_dir = self.sun_dir.to(vertices.device)

        raw_loss, diagnostics = solar_loss_srrt(
            vertices=vertices,
            faces=faces,
            sun_dir=sun_dir,
            area_budget_mm2=self.area_budget_mm2,
            raytracing_cfg=self.raytracing_cfg,
            effective_irradiance_W_m2=self.effective_irradiance_W_m2,
            sigmoid_sharpness=self.sigmoid_sharpness,
            shading_penalty_weight=self.shading_penalty_weight,
            kappa=self.kappa,
            _lambda=self._lambda_val,
            epsilon=self.epsilon,
            use_activation_checkpointing=self.use_activation_checkpointing,
            smooth_gradient_iters=self.smooth_gradient_iters,
            vis_manager=getattr(self, "_vis_manager", None) if not skip_telemetry else None,
            epoch_name=self.epoch_name,
        )

        if not skip_telemetry:
            self._last_diagnostics = diagnostics

        return raw_loss
