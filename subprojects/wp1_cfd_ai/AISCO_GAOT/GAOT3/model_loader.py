from __future__ import annotations

import ast
import copy
import os
import warnings
from dataclasses import replace as _replace
from typing import TYPE_CHECKING

import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch, Data

import wandb
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.sdf_influence import compute_sdf_parameter_influence
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.shear_force import ShearForce
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.dataloader.metadata import DATASET_METADATA
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.dataloader.pyg_transforms import RescalePosition, RescalePositionNew
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.model.gaot_3d import GAOT3D
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.utils.scale import rescale, rescale_new
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_baseclass import BaseModel
from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor, load_mesh_extraction_config

if TYPE_CHECKING:
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import NullVisManager
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import CentralVisManagerV2


class ModelLoader:
    def __init__(self, cfd_config: dict, device: str) -> None:
        self.cfd_config = cfd_config
        self.device = device
        self.vis_manager: CentralVisManagerV2 | NullVisManager | None
        self.vis_manager = None
        self.sdf_influence_learning_rate = float(cfd_config.get("sdf_influence_learning_rate", 1e-3))
        self.sdf_influence_chunk_size = int(cfd_config.get("sdf_influence_chunk_size", 8192))
        self.shear_force_calculator = ShearForce(unit="mm^2")

        self.optimization_state: dict[str, torch.Tensor | None] = {
            "vertices": None,
            "faces": None,
            "vertice_grads": None,
        }

        self.model_path = self._get_checkpoint_path()
        self._load_model()

        mesh_cfg = load_mesh_extraction_config()
        mesh_cfg = _replace(
            mesh_cfg,
            device=self.device,
            bbox=[[-4e3, 4e3], [-4e3, 4e3], [-4e3, 4e3]],
            isovalue=0.0,
            target_num_vertices=int(cfd_config["mesh_resolution"]),
            vertex_count_tolerance=float(cfd_config.get("mesh_resolution_tolerance", 0.1)),
        )
        self.extractor = MeshSDFExtractor(
            config=mesh_cfg,
            gradient_max=float(cfd_config["gradient_max"]),
        )

    def _get_checkpoint_path(self) -> str:
        try:
            df = pd.read_csv("/cluster/work/fuge/AISCO/wp1_cfd_ai/results/metrics.csv")
        except FileNotFoundError:
            df = pd.read_csv("/home/aisco/results/metrics.csv")
        for _, row in df.iterrows():
            setup_dict = ast.literal_eval(row["setup"])
            if str(setup_dict.get("id")) == str(self.cfd_config["model_id"]):
                path_dict = ast.literal_eval(row["path"])
                return path_dict.get("ckpt_path")
        raise ValueError("Wrong ID for GAOT-Model")

    def _load_model(self) -> None:
        config_path = self.model_path.replace("model.pt", "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config not found at {config_path}. Expected config.yaml next to model.pt.")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        self._og_config = copy.deepcopy(config)

        model_args = config["model"]["args"]

        model_args["magno"]["precompute_edges"] = False

        norm_eps = model_args["transformer"].get("norm_eps")
        model_args["transformer"]["norm_eps"] = float(norm_eps)

        self.gaot = GAOT3D(config=model_args).to(self.device)
        self.gaot.eval()

        state = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if "model" in state:
            sd = state["model"]
        elif "state_dict" in state:
            sd = state["state_dict"]
        else:
            sd = state

        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        missing, unexpected = self.gaot.load_state_dict(sd, strict=False)
        if missing or unexpected:
            warnings.warn(
                f"Checkpoint/state_dict mismatch. Missing keys: {missing}. Unexpected keys: {unexpected}.", stacklevel=2
            )

        # Reconstruct latent tokens from training context instead of using self.gaot.latent_tokens
        dataset_cfg = self._og_config.get("dataset", {})
        self._use_rescale_new = dataset_cfg.get("use_rescale_new", False)

        metaname = dataset_cfg.get("metaname")
        if metaname and metaname in DATASET_METADATA:
            self._metadata = DATASET_METADATA[metaname]
            self._phy_domain = self._metadata.domain_x
        else:
            norm_domain = model_args.get("norm_domain")
            if norm_domain is None:
                raise KeyError(
                    "Could not determine physical domain. Expected dataset.metaname in DATASET_METADATA "
                    "or model.args.norm_domain."
                )
            self._phy_domain = norm_domain

        latent_token_size = model_args.get("latent_tokens")
        if latent_token_size is None:
            raise KeyError("Expected model.args.latent_tokens in saved config.")

        x_min, y_min, z_min = self._phy_domain[0]
        x_max, y_max, z_max = self._phy_domain[1]

        meshgrid = torch.meshgrid(
            torch.linspace(x_min, x_max, latent_token_size[0]),
            torch.linspace(y_min, y_max, latent_token_size[1]),
            torch.linspace(z_min, z_max, latent_token_size[2]),
            indexing="ij",
        )
        latent_queries = torch.stack(meshgrid, dim=-1).reshape(-1, 3)

        if self._use_rescale_new:
            self._latent_tokens = rescale_new(latent_queries, (-1, 1), self._phy_domain).to(self.device)
        else:
            self._latent_tokens = rescale(latent_queries, (-1, 1)).to(self.device)

    def forward(self, sdf_model: BaseModel, epoch_name: int) -> torch.Tensor:
        self.epoch_name = epoch_name
        self.sdf_model = sdf_model
        verts, faces, loss, surrogate_loss, dL_dvertices = self.extractor.extract_and_compute_loss(
            sdf_model=sdf_model, loss_fn=self.loss_fn, smooth_vertex_gradients_iters=5
        )
        self.extractor.last_dL_dv = dL_dvertices  # type: ignore[attr-defined]
        self.extractor.last_verts = verts  # type: ignore[attr-defined]
        self.extractor.last_faces = faces  # type: ignore[attr-defined]
        self._visualize_sdf_influence(
            sdf_model=sdf_model,
            surrogate_loss=surrogate_loss,
            mesh_verts=verts,
            mesh_faces=faces,
        )

        cuppled_loss = loss.detach() + surrogate_loss - surrogate_loss.detach()
        return cuppled_loss

    def _due_sdf_influence_streams(self, epoch: int) -> list[tuple[str, str]]:
        if not self.cfd_config["visualisations"]:
            return []
        _vm = getattr(self, "vis_manager", None)
        is_due = getattr(_vm, "is_due", None)
        if not callable(is_due):
            return []
        streams = [
            ("jvp", "raw_gradient"),
            ("jvp", "update_direction"),
            ("finite", "raw_gradient"),
            ("finite", "update_direction"),
        ]
        return [(method, variant) for method, variant in streams if is_due(f"aero/sdf_influence/{method}/{variant}", epoch)]

    def _collect_sdf_parameter_grads(
        self,
        sdf_model: BaseModel,
        surrogate_loss: torch.Tensor,
    ) -> dict[str, torch.Tensor | None] | None:
        if not surrogate_loss.requires_grad:
            return None
        trainable_params = [(name, param) for name, param in sdf_model.named_parameters() if param.requires_grad]
        if not trainable_params:
            return None
        names = [name for name, _param in trainable_params]
        params = [param for _name, param in trainable_params]
        grads = torch.autograd.grad(
            surrogate_loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        grad_map: dict[str, torch.Tensor | None] = dict(zip(names, grads, strict=True))
        if all(grad is None for grad in grad_map.values()):
            return None
        return grad_map

    def _visualize_sdf_influence(
        self,
        sdf_model: BaseModel,
        surrogate_loss: torch.Tensor,
        mesh_verts: torch.Tensor,
        mesh_faces: torch.Tensor,
    ) -> None:
        if mesh_verts is None or mesh_faces is None or mesh_verts.numel() == 0 or mesh_faces.numel() == 0:
            return
        epoch = int(self.epoch_name)
        streams = self._due_sdf_influence_streams(epoch)
        if not streams:
            return

        _vm = getattr(self, "vis_manager", None)
        if _vm is None:
            return

        try:
            param_grads = self._collect_sdf_parameter_grads(sdf_model, surrogate_loss)
            if param_grads is None:
                return
            fields = compute_sdf_parameter_influence(
                sdf_model,
                mesh_verts,
                param_grads,
                learning_rate=self.sdf_influence_learning_rate,
                chunk_size=self.sdf_influence_chunk_size,
            )
            for method, variant in streams:
                values = fields.get(f"{method}/{variant}")
                if values is not None:
                    _vm.vis_aero_sdf_influence(epoch, method, variant, mesh_verts, mesh_faces, values)
        except Exception as exc:
            warnings.warn(f"Skipping GAOT SDF-influence visualisation after error: {exc}", stacklevel=2)

    def _visualize_pressure_field(
        self,
        pressure_field: torch.Tensor,
        mesh_verts: torch.Tensor,
        mesh_faces: torch.Tensor,
    ) -> None:
        if not self.cfd_config["visualisations"]:
            return
        if pressure_field is None:
            return

        _vm = getattr(self, "vis_manager", None)
        if _vm is not None:
            if pressure_field.dim() == 2 and pressure_field.size(-1) == 1:
                pressure_field = pressure_field.squeeze(-1)
            _vm.vis_aero_cp_field(self.epoch_name, mesh_verts, mesh_faces, pressure_field)
            return

    def _pressure_backprop_mode(self) -> str:
        mode = str(self.cfd_config.get("pressure_backprop_mode", "full"))
        valid_modes = {"full", "geometry_only", "geometry_area_only", "geometry_normal_only"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown pressure_backprop_mode={mode!r}. Expected one of {sorted(valid_modes)}.")
        return mode

    def _integrate_pressure_drag(self, pressure_field: torch.Tensor, batch: Batch, mesh_faces: torch.Tensor) -> torch.Tensor:
        p = pressure_field.squeeze(-1) if pressure_field.dim() > 1 else pressure_field
        face = mesh_faces.T
        pos = batch.pos_raw
        normals = batch.c
        pressure_backprop_mode = self._pressure_backprop_mode()
        cp_smoothing_cfg = self.cfd_config.get("cp_smoothing", {})
        detach_integration_geometry = (
            pressure_backprop_mode == "full"
            and isinstance(cp_smoothing_cfg, dict)
            and bool(cp_smoothing_cfg.get("detach_integration_geometry", False))
        )
        if detach_integration_geometry:
            pos = pos.detach()
            normals = normals.detach()

        v0 = pos[face[0]]
        v1 = pos[face[1]]
        v2 = pos[face[2]]
        raw_area_normals = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1)
        area = raw_area_normals.norm(dim=-1, keepdim=True)
        unit_normals = raw_area_normals / area.clamp_min(torch.finfo(raw_area_normals.dtype).eps)

        outward_n = (normals[face[0]] + normals[face[1]] + normals[face[2]]) / 3.0
        dot_prod = (unit_normals * outward_n).sum(dim=-1, keepdim=True)
        unit_normals = torch.where(dot_prod < 0, -unit_normals, unit_normals)

        if pressure_backprop_mode == "geometry_area_only":
            unit_normals = unit_normals.detach()
        elif pressure_backprop_mode == "geometry_normal_only":
            area = area.detach()
        area_normals = area * unit_normals

        q_inf = 0.5 * self.cfd_config["rho"] * self.cfd_config["v_inf"] ** 2
        p_face = q_inf * (p[face[0]] + p[face[1]] + p[face[2]]) / 3.0
        return (-p_face * area_normals[:, 0]).sum()

    def _compute_sdf_vertex_normals(
        self,
        mesh_verts: torch.Tensor,
        detach_normals: bool = True,
    ) -> torch.Tensor:
        verts_for_normals = mesh_verts.detach().requires_grad_(True) if detach_normals else mesh_verts
        if not verts_for_normals.requires_grad:
            verts_for_normals = verts_for_normals.detach().requires_grad_(True)

        sdf_values = self.sdf_model(verts_for_normals)
        if sdf_values.dim() > 1:
            sdf_values = sdf_values.squeeze(-1)

        (sdf_grads,) = torch.autograd.grad(
            sdf_values,
            verts_for_normals,
            torch.ones_like(sdf_values),
            create_graph=not detach_normals,
        )
        normals = torch.nn.functional.normalize(sdf_grads, dim=-1)
        normals = normals.clone()
        normals[:, 0] *= -1.0
        return normals.detach() if detach_normals else normals

    def _integrate_pressure_drag_sdf_normals(
        self,
        pressure_field: torch.Tensor,
        mesh_verts: torch.Tensor,
        mesh_faces: torch.Tensor,
        *,
        detach_normals: bool = True,
    ) -> torch.Tensor:
        p = pressure_field.squeeze(-1) if pressure_field.dim() > 1 else pressure_field
        if p.shape[0] != mesh_verts.shape[0]:
            raise ValueError(f"Pressure field length {p.shape[0]} does not match vertex count {mesh_verts.shape[0]}.")

        normals = self._compute_sdf_vertex_normals(mesh_verts, detach_normals=detach_normals)

        verts_m = mesh_verts * 1e-3
        v0 = verts_m[mesh_faces[:, 0]]
        v1 = verts_m[mesh_faces[:, 1]]
        v2 = verts_m[mesh_faces[:, 2]]
        face_areas = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)
        area_per_vertex = (2.0 * face_areas.mean()).detach()

        q_inf = 0.5 * self.cfd_config["rho"] * self.cfd_config["v_inf"] ** 2
        pressure_pa = q_inf * p
        return (-pressure_pa * normals[:, 0] * area_per_vertex).sum()

    def _smooth_cp_field_if_enabled(  # Noqa:PLR0915,PLR0912
        self,
        pressure_field: torch.Tensor,
        batch: Batch,
        mesh_verts: torch.Tensor,
        mesh_faces: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfd_config.get("cp_smoothing", {})
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            return pressure_field

        original_shape = pressure_field.shape
        p = pressure_field.squeeze(-1) if pressure_field.dim() > 1 else pressure_field
        if p.shape[0] != mesh_verts.shape[0]:
            raise ValueError(f"Pressure field length {p.shape[0]} does not match vertex count {mesh_verts.shape[0]}.")

        iterations = max(0, int(cfg.get("iterations", 3)))
        if iterations == 0 or mesh_faces.numel() == 0:
            return pressure_field

        min_strength = max(0.0, min(1.0, float(cfg.get("min_strength", 0.0))))
        max_strength = max(0.0, min(1.0, float(cfg.get("max_strength", 0.35))))
        if min_strength > max_strength:
            min_strength, max_strength = max_strength, min_strength
        curvature_quantile = max(0.0, min(1.0, float(cfg.get("curvature_quantile", 0.95))))
        curvature_power = max(0.0, float(cfg.get("curvature_power", 2.0)))
        detach_smoothing_weights = bool(cfg.get("detach_smoothing_weights", cfg.get("detach_integration_geometry", False)))

        faces = mesh_faces.to(device=p.device, dtype=torch.long)
        src = torch.cat([faces[:, 0], faces[:, 1], faces[:, 1], faces[:, 2], faces[:, 2], faces[:, 0]], dim=0)
        dst = torch.cat([faces[:, 1], faces[:, 0], faces[:, 2], faces[:, 1], faces[:, 0], faces[:, 2]], dim=0)

        num_verts = p.shape[0]
        degree = torch.zeros(num_verts, device=p.device, dtype=p.dtype)
        degree = degree.index_add(0, dst, torch.ones_like(src, dtype=p.dtype)).clamp_min(1.0)

        normals = batch.c_raw if hasattr(batch, "c_raw") and batch.c_raw is not None else batch.c
        normals = torch.nn.functional.normalize(normals.to(device=p.device, dtype=p.dtype), dim=-1)
        if detach_smoothing_weights:
            normals = normals.detach()
        edge_curvature = (1.0 - (normals[src] * normals[dst]).sum(dim=-1)).clamp_min(0.0)
        curvature = torch.zeros(num_verts, device=p.device, dtype=p.dtype)
        curvature = curvature.index_add(0, dst, edge_curvature) / degree

        curvature_scale = torch.quantile(curvature.detach().float(), curvature_quantile).to(device=p.device, dtype=p.dtype)
        curvature_scale = curvature_scale.clamp_min(torch.finfo(p.dtype).eps)
        curvature_norm = (curvature / curvature_scale).clamp(0.0, 1.0)
        smoothing_strength = min_strength + (max_strength - min_strength) * (1.0 - curvature_norm).pow(curvature_power)
        if detach_smoothing_weights:
            smoothing_strength = smoothing_strength.detach()

        p_smoothed = p
        for _ in range(iterations):
            neighbour_sum = torch.zeros_like(p_smoothed)
            neighbour_sum = neighbour_sum.index_add(0, dst, p_smoothed[src])
            neighbour_mean = neighbour_sum / degree
            p_smoothed = p_smoothed + smoothing_strength * (neighbour_mean - p_smoothed)

        if bool(cfg.get("preserve_drag", True)):
            q_inf = 0.5 * self.cfd_config["rho"] * self.cfd_config["v_inf"] ** 2
            pressure_integration_mode = self.cfd_config.get("pressure_integration_mode", "face")
            if pressure_integration_mode == "face":
                face = faces.T
                pos = batch.pos_raw.to(device=p.device, dtype=p.dtype)
                integration_normals = batch.c.to(device=p.device, dtype=p.dtype)
                if bool(cfg.get("detach_integration_geometry", False)):
                    pos = pos.detach()
                    integration_normals = integration_normals.detach()
                v0 = pos[face[0]]
                v1 = pos[face[1]]
                v2 = pos[face[2]]
                area_normals = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1)
                outward_n = (integration_normals[face[0]] + integration_normals[face[1]] + integration_normals[face[2]]) / 3.0
                dot_prod = (area_normals * outward_n).sum(dim=-1, keepdim=True)
                area_normals = torch.where(dot_prod < 0, -area_normals, area_normals)
                per_face_weight = -q_inf * area_normals[:, 0] / 3.0
                drag_weights = torch.zeros(num_verts, device=p.device, dtype=p.dtype)
                drag_weights = drag_weights.index_add(0, face[0], per_face_weight)
                drag_weights = drag_weights.index_add(0, face[1], per_face_weight)
                drag_weights = drag_weights.index_add(0, face[2], per_face_weight)
            elif pressure_integration_mode == "sdf_vertex_constant_area":
                normals_sdf = self._compute_sdf_vertex_normals(
                    mesh_verts,
                    detach_normals=bool(self.cfd_config.get("pressure_integration_detach_sdf_normals", True)),
                ).to(device=p.device, dtype=p.dtype)
                verts_m = mesh_verts.to(device=p.device, dtype=p.dtype) * 1e-3
                v0 = verts_m[faces[:, 0]]
                v1 = verts_m[faces[:, 1]]
                v2 = verts_m[faces[:, 2]]
                face_areas = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)
                area_per_vertex = (2.0 * face_areas.mean()).detach()
                drag_weights = -q_inf * normals_sdf[:, 0] * area_per_vertex
            else:
                drag_weights = None

            if drag_weights is not None:
                drag_weight_norm = (drag_weights * drag_weights).sum()
                if torch.isfinite(drag_weight_norm) and drag_weight_norm.detach().abs().item() > torch.finfo(p.dtype).eps:
                    raw_drag = (p * drag_weights).sum()
                    smoothed_drag = (p_smoothed * drag_weights).sum()
                    p_smoothed = p_smoothed + (raw_drag - smoothed_drag) * drag_weights / drag_weight_norm

        if len(original_shape) == 2 and original_shape[-1] == 1:
            return p_smoothed.unsqueeze(-1)
        return p_smoothed.reshape(original_shape)

    def loss_fn(self, mesh_verts: torch.Tensor, mesh_faces: torch.Tensor) -> torch.Tensor:

        self._check_watertightness(mesh_verts, mesh_faces)

        wandb.log({"cfd/Aero-resolution": mesh_verts.shape[0]}, step=self.epoch_name)

        batch = self._reformat_data(mesh_verts, mesh_faces)

        tokens_pos = self._latent_tokens

        with torch.autograd.set_detect_anomaly(True):
            gaot_loss = self.gaot.forward(batch=batch, tokens_pos=tokens_pos)

        self.optimization_state = {"vertices": mesh_verts, "faces": mesh_faces}

        pressure_field = gaot_loss["field"]
        cp_smoothing_cfg = self.cfd_config.get("cp_smoothing", {})
        if isinstance(cp_smoothing_cfg, dict) and bool(cp_smoothing_cfg.get("enabled", False)):
            pressure_field = self._smooth_cp_field_if_enabled(pressure_field, batch, mesh_verts, mesh_faces)

        self._visualize_pressure_field(
            pressure_field=pressure_field,
            mesh_verts=mesh_verts,
            mesh_faces=mesh_faces,
        )

        pressure_backprop_mode = self._pressure_backprop_mode()
        pressure_field_for_integration = pressure_field if pressure_backprop_mode == "full" else pressure_field.detach()

        pressure_integration_mode = self.cfd_config.get("pressure_integration_mode", "face")
        if pressure_integration_mode == "face":
            int_drag_f = self._integrate_pressure_drag(pressure_field_for_integration, batch, mesh_faces)
        elif pressure_integration_mode == "sdf_vertex_constant_area":
            if pressure_backprop_mode == "geometry_area_only":
                raise ValueError(
                    "pressure_backprop_mode='geometry_area_only' is not supported with "
                    "pressure_integration_mode='sdf_vertex_constant_area' because area is detached there."
                )
            detach_normals = bool(self.cfd_config.get("pressure_integration_detach_sdf_normals", True))
            if pressure_backprop_mode in {"geometry_only", "geometry_normal_only"}:
                detach_normals = False
            int_drag_f = self._integrate_pressure_drag_sdf_normals(
                pressure_field=pressure_field_for_integration,
                mesh_verts=mesh_verts,
                mesh_faces=mesh_faces,
                detach_normals=detach_normals,
            )
        else:
            raise ValueError(
                f"Unknown pressure_integration_mode={pressure_integration_mode!r}. Expected 'face' or 'sdf_vertex_constant_area'."
            )

        shearforce = self.shear_force_calculator.calculate_shear_force(mesh_verts, mesh_faces)
        wandb.log({"cfd/Shearforce:": shearforce}, step=self.epoch_name)
        wandb.log({"cfd/int_drag_f": int_drag_f}, step=self.epoch_name)
        return int_drag_f + shearforce

    def _reformat_data(self, mesh_verts: torch.Tensor, mesh_faces: torch.Tensor) -> Batch:
        optim_verts = mesh_verts.to(self.device)
        mesh_faces = mesh_faces.to(self.device)

        v0 = optim_verts[mesh_faces[:, 0]]
        v1 = optim_verts[mesh_faces[:, 1]]
        v2 = optim_verts[mesh_faces[:, 2]]
        face_centers = (v0 + v1 + v2) / 3

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=-1)

        with torch.no_grad():
            sdf_values = self.sdf_model(face_centers + face_normals * 50).squeeze(-1)
        normals_flipped = sdf_values < 0
        self._normals_flipped_count = int(normals_flipped.sum().item())
        wandb.log({"cfd/unhealthy normals:": self._normals_flipped_count}, step=self.epoch_name)

        num_verts = optim_verts.shape[0]
        all_idx = mesh_faces.reshape(-1)
        all_normals = face_normals.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)

        vertex_normals = torch.zeros(num_verts, 3, device=self.device, dtype=optim_verts.dtype)
        vertex_normals = vertex_normals.index_add(0, all_idx, all_normals)
        vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=-1)

        raw_vertex_normals = vertex_normals.clone()

        model_vertex_normals = vertex_normals.clone()
        model_vertex_normals[:, 0] *= -1.0

        model_verts_m = optim_verts.clone() * 1e-3
        model_verts_m[:, 0] *= -1.0

        data = Data(
            pos=model_verts_m,
            face=mesh_faces.T,
            c=model_vertex_normals,
            c_raw=raw_vertex_normals,
        )

        if self._og_config["dataset"]["use_rescale_new"]:
            data = RescalePositionNew(lims=(-1.0, 1.0), phy_domain=self._phy_domain)(data)
        else:
            data = RescalePosition(lims=(-1.0, 1.0))(data)

        return Batch.from_data_list([data]).to(self.device)

    def _check_watertightness(self, verts: torch.Tensor, faces: torch.Tensor) -> None:
        # Check number of vertecies and faces
        v = verts.shape[0]
        f = faces.shape[0]
        d = v - f / 2
        wandb.log({"cfd/Mesh Quality (Target:2)": d}, step=self.epoch_name)
