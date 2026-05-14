from __future__ import annotations

from dataclasses import replace as _replace

import torch

from subprojects.wp3_solar_estimator.rollout_losses.MR_SolarLoss import solar_loss_mr
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor, load_mesh_extraction_config


class SolarMR(ParentLoss):
    def _setup(self) -> None:
        gen_cfg = load_config(v="v2", cfg_name="general_config.yaml")
        cfg_path = gen_cfg["active_configs"]["solar"]
        cfg = load_config(v="v2", cfg_name=cfg_path)["SolarLossConfigMR"]

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
        self.smooth_gradient_iters: int = int(cfg.get("smooth_gradient_iters", 0))

        source = cfg.get("solar_vector_source", "route_v2")
        if source == "route_v2":
            from subprojects.wp3_solar_estimator.australia.route_v3 import RouteV3

            route = RouteV3(cfg_path=cfg.get("route_cfg_path", "solar_configs/australia.yaml"), mode="race")
            subsample = cfg.get("subsample_n") or None
            self.solar_vectors = route.solar_vectors(subsample=subsample)
        elif source == "custom":
            test_vecs_cfg = load_config(v="v2", cfg_name="solar_configs/test_solar_vectors.yaml")
            self.solar_vectors = torch.tensor(test_vecs_cfg[cfg["custom_vectors_key"]], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown solar_vector_source: {source!r}")

    def _loss(self, model, skip_telemetry: bool = False) -> torch.Tensor:
        vertices, faces = self.mesh_extractor.extract_mesh(model)
        solar_vectors = self.solar_vectors.to(vertices.device)

        raw_loss, diagnostics = solar_loss_mr(
            vertices=vertices,
            faces=faces,
            solar_vectors=solar_vectors,
            area_budget_mm2=self.area_budget_mm2,
            effective_irradiance_W_m2=self.effective_irradiance_W_m2,
            sigmoid_sharpness=self.sigmoid_sharpness,
            smooth_gradient_iters=self.smooth_gradient_iters,
            vis_manager=getattr(self, "_vis_manager", None) if not skip_telemetry else None,
            epoch_name=self.epoch_name,
        )

        if not skip_telemetry:
            self._last_diagnostics = diagnostics

        return raw_loss
