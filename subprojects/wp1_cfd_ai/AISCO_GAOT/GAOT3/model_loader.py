from __future__ import annotations

import ast
import copy
import logging
import os
import tempfile
import warnings
from dataclasses import replace as _replace
from typing import TYPE_CHECKING

import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader

import wandb
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.shear_force import ShearForce
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.dataloader.metadata import DATASET_METADATA
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.dataloader.pyg_transforms import RescalePosition, RescalePositionNew
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.model.gaot_3d import GAOT3D
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.trainer.utils.plot import plot_3d_comparison_matplotlib
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.utils.pressure_integration import cp_field_to_drag
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.utils.scale import rescale, rescale_new
from subprojects.wp4_aeroshell_optimizer import MeshSDFGrads, Visualiser
from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor, load_mesh_extraction_config

if TYPE_CHECKING:
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager import CentralVisManager

logger = logging.getLogger(__name__)


class ModelLoader:
    optimization_state: dict[str, object]
    vis_manager: CentralVisManager | None
    _DEFAULT_MAIN_CONFIG_PATH = "subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/config/main_config.yaml"

    def __init__(self, cfd_config: dict, device: str) -> None:
        self.cfd_config = cfd_config
        self.device = device
        self._init_viz_consistency_settings()
        self.shear_force_calculator = ShearForce(unit="mm^2")

        self.optimization_state = {
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
            bbox=[[-3e3, 3e3], [-3e3, 3e3], [-3e3, 3e3]],
            isovalue=0.0,
            target_num_vertices=int(cfd_config["mesh_resolution"]),
            vertex_count_tolerance=float(cfd_config.get("mesh_resolution_tolerance", 0.1)),
        )
        self.extractor = MeshSDFExtractor(
            config=mesh_cfg,
            gradient_max=float(cfd_config["gradient_max"]),
        )

        self.visualiser = Visualiser(device=self.device)
        self.gradlogger = MeshSDFGrads(self.visualiser, intensity=10)

    def _get_dataset_viz_config(self) -> dict:
        dataset_cfg = self.cfd_config.get("dataset", {}) if isinstance(self.cfd_config.get("dataset", {}), dict) else None
        if isinstance(dataset_cfg, dict) and dataset_cfg:
            return dataset_cfg

        main_config_path = self.cfd_config.get("main_config_path", self._DEFAULT_MAIN_CONFIG_PATH)
        if not os.path.isabs(main_config_path):
            main_config_path = os.path.join(os.getcwd(), main_config_path)
        if not os.path.exists(main_config_path):
            raise FileNotFoundError(
                "Could not resolve dataset visualization settings. "
                f"Expected cfd_config['dataset'] or main config at {main_config_path}."
            )

        with open(main_config_path) as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise TypeError(f"Main config at {main_config_path} must be a dictionary.")

        loaded_dataset_cfg = cfg.get("dataset", {})
        if not isinstance(loaded_dataset_cfg, dict):
            raise TypeError(f"Expected 'dataset' section in {main_config_path} to be a dictionary.")
        return loaded_dataset_cfg

    def _init_viz_consistency_settings(self):
        """Build fixed axis and color ranges exactly like GAOT3 trainer/stat.py."""
        dataset_cfg = self._get_dataset_viz_config()

        axis_cfg = dataset_cfg.get("viz_axis_limits")
        if axis_cfg is not None:
            parsed_limits: dict[str, tuple[float, float]] = {}
            for axis_name in ("x", "y", "z"):
                axis_range = axis_cfg.get(axis_name) if isinstance(axis_cfg, dict) else None
                if axis_range is None or len(axis_range) != 2:
                    raise ValueError(f"dataset.viz_axis_limits.{axis_name} must be a 2-element range [min, max].")
                axis_min = float(axis_range[0])
                axis_max = float(axis_range[1])
                if axis_max <= axis_min:
                    raise ValueError(f"Invalid dataset.viz_axis_limits.{axis_name}: min={axis_min} must be < max={axis_max}.")
                parsed_limits[axis_name] = (axis_min, axis_max)
            self._viz_fixed_axis_limits = parsed_limits
        else:
            self._viz_fixed_axis_limits = {
                "x": (-3.0, 3.0),
                "y": (-3.0, 3.0),
                "z": (-3.0, 3.0),
            }

        vmin = dataset_cfg.get("viz_field_vmin")
        vmax = dataset_cfg.get("viz_field_vmax")
        diff_vmax = dataset_cfg.get("viz_abs_diff_vmax")

        if vmin is None or vmax is None:
            raise ValueError("dataset.viz_field_vmin and dataset.viz_field_vmax are required for visualization.")
        if diff_vmax is None:
            raise ValueError("dataset.viz_abs_diff_vmax is required for visualization.")

        vmin_f = float(vmin)
        vmax_f = float(vmax)
        diff_vmax_f = float(diff_vmax)
        if vmax_f <= vmin_f:
            raise ValueError(f"Invalid viz field range: viz_field_vmin={vmin_f} must be < viz_field_vmax={vmax_f}.")
        if diff_vmax_f <= 0.0:
            raise ValueError(f"Invalid viz_abs_diff_vmax={diff_vmax_f}; it must be > 0.")

        self._viz_fixed_clim_common = (vmin_f, vmax_f)
        self._viz_fixed_clim_diff = (0.0, diff_vmax_f)

    @staticmethod
    def _to_wandb_step(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _get_checkpoint_path(self):
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

    def _load_model(self) -> None:  # Noqa: PLR0912, PLR0915
        config_path = self.model_path.replace("model.pt", "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config not found at {config_path}. Expected config.yaml next to model.pt.")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise TypeError("Loaded config must be a dictionary.")

        self._og_config = copy.deepcopy(config)

        if "model" not in config or "args" not in config["model"]:
            raise KeyError("Expected config['model']['args'] in saved config.")

        model_args = config["model"]["args"]

        if "magno" not in model_args:
            raise KeyError("Expected 'magno' section in model args.")
        if "transformer" not in model_args:
            raise KeyError("Expected 'transformer' section in model args.")

        model_args["magno"]["precompute_edges"] = False

        norm_eps = model_args["transformer"].get("norm_eps")
        if norm_eps is None:
            raise KeyError("Expected transformer.norm_eps in model args.")
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

    def forward(self, sdf_model, epoch_name):
        self.epoch_name = epoch_name
        self.sdf_model = sdf_model
        verts, faces, loss, surrogate_loss, dL_dvertices = self.extractor.extract_and_compute_loss(
            sdf_model=sdf_model, loss_fn=self.loss_fn
        )

        diag = getattr(self, "_last_diagnostics", {})
        diag["mesh_n_vertices"] = int(verts.shape[0]) if verts is not None and verts.numel() > 0 else 0
        diag["mesh_n_faces"] = int(faces.shape[0]) if faces is not None and faces.numel() > 0 else 0
        diag["raw_loss"] = loss.item()
        diag["surrogate_loss"] = surrogate_loss.item()
        diag["mesh_empty"] = verts is None or verts.numel() == 0
        if dL_dvertices is not None and dL_dvertices.numel() > 0:
            diag["dL_dvertices_norm"] = dL_dvertices.norm().item()
            diag["dL_dvertices_max"] = dL_dvertices.abs().max().item()
        self._last_diagnostics = diag

        self.optimization_state["vertice_grads"] = dL_dvertices

        if self.cfd_config["visualisations"]:
            self.gradlogger.log_single_image_to_wandb(self.optimization_state, epoch_name=self.epoch_name)

        cuppled_loss = loss.detach() + surrogate_loss - surrogate_loss.detach()
        return cuppled_loss

    def _visualize_pressure_field(
        self,
        pressure_field: torch.Tensor,
        mesh_verts: torch.Tensor,
        mesh_faces: torch.Tensor,
        reference_field: torch.Tensor | None = None,
    ) -> None:
        if not self.cfd_config["visualisations"]:
            return
        if pressure_field is None:
            return

        # CentralVisManager path: write PLY instead of wandb PNG.
        _vm = getattr(self, "vis_manager", None)
        if _vm is not None:
            if pressure_field.dim() == 2 and pressure_field.size(-1) == 1:
                pressure_field = pressure_field.squeeze(-1)
            field_pred = pressure_field.detach().cpu().numpy().squeeze()
            epoch_int = self._to_wandb_step(self.epoch_name) or 0
            _vm.save_ply(
                "aero/cp_field",
                epoch_int,
                mesh_verts.detach().cpu().numpy(),
                mesh_faces.detach().cpu().numpy(),
                scalar_values=field_pred,
            )
            return

        sample = Data(
            pos=mesh_verts.detach() * 1e-3,
            x=None,
            face=mesh_faces.detach().T,
        )

        loader = PyGDataLoader([sample], batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(self.device)

        coords_tensor = batch.pos_raw if hasattr(batch, "pos_raw") else batch.pos
        coords = coords_tensor.detach().cpu().numpy()
        fname = getattr(self, "epoch_name", "mesh")

        if pressure_field.dim() == 2 and pressure_field.size(-1) == 1:
            pressure_field = pressure_field.squeeze(-1)
        if reference_field is not None and reference_field.dim() == 2 and reference_field.size(-1) == 1:
            reference_field = reference_field.squeeze(-1)

        field_pred = pressure_field.detach().cpu().numpy().squeeze()
        field_ref = None if reference_field is None else reference_field.detach().cpu().numpy().squeeze()

        # we don't have a numeric epoch here, only a name
        title_epoch = getattr(self, "epoch_name", "opt_step")
        with tempfile.NamedTemporaryFile(suffix=".png", prefix="pressure_field_", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        step_val = self._to_wandb_step(self.epoch_name)
        try:
            plot_3d_comparison_matplotlib(
                coords=coords,
                u_gtr=field_ref,
                u_prd=field_pred,
                save_path=tmp_path,
                variable_name="Pressure",
                point_size=2.0,
                cmap="jet",
                dpi=150,
                view_angle=(20, -120),
                hide_grid=False,
                equal_aspect=False,
                show_abs_diff=field_ref is not None,
                show_input_shape=False,
                fixed_clim_common=self._viz_fixed_clim_common,
                fixed_clim_diff=self._viz_fixed_clim_diff,
                fixed_axis_limits=self._viz_fixed_axis_limits,
            )
            wandb_payload = {
                "Pressure-Field visualisation": wandb.Image(
                    tmp_path,
                    caption=f"opt {fname} {title_epoch}",
                )
            }
            if step_val is None:
                wandb.log(wandb_payload)
            else:
                wandb.log(wandb_payload, step=step_val)
        finally:
            try:  # noqa: SIM105
                os.remove(tmp_path)
            except OSError:
                pass

    def _log_front_back_pressure_drag(self, pressure_field: torch.Tensor, batch: Batch, mesh_faces: torch.Tensor) -> None:
        with torch.no_grad():
            p = pressure_field.squeeze(-1) if pressure_field.dim() > 1 else pressure_field
            face = mesh_faces.T
            pos = batch.pos_raw
            normals = batch.c

            v0 = pos[face[0]]
            v1 = pos[face[1]]
            v2 = pos[face[2]]
            area_normals = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1)

            outward_n = (normals[face[0]] + normals[face[1]] + normals[face[2]]) / 3.0
            dot_prod = (area_normals * outward_n).sum(dim=-1, keepdim=True)
            area_normals = torch.where(dot_prod < 0, -area_normals, area_normals)

            q_inf = 0.5 * self.cfd_config["rho"] * self.cfd_config["v_inf"] ** 2
            p_face = q_inf * (p[face[0]] + p[face[1]] + p[face[2]]) / 3.0
            n_dir_area = area_normals[:, 0]
            face_drag = -p_face * n_dir_area

            front_faces = n_dir_area < 0
            back_faces = n_dir_area > 0
            wandb.log(
                {
                    "cfd/int_drag_front_f": face_drag[front_faces].sum().item(),
                    "cfd/int_drag_back_f": face_drag[back_faces].sum().item(),
                    "cfd/front_face_count": int(front_faces.sum().item()),
                    "cfd/back_face_count": int(back_faces.sum().item()),
                },
                step=self.epoch_name,
            )

    def loss_fn(self, mesh_verts, mesh_faces):
        if mesh_verts is None or mesh_verts.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)

        self._check_watertightness(mesh_verts, mesh_faces)

        step_val = self._to_wandb_step(self.epoch_name)
        if step_val is None:
            wandb.log({"cfd/Aero-resolution": mesh_verts.shape[0]})
        else:
            wandb.log({"cfd/Aero-resolution": mesh_verts.shape[0]}, step=step_val)

        batch = self._reformat_data(mesh_verts, mesh_faces)

        # Fix 5: use reconstructed latent tokens from training context
        tokens_pos = self._latent_tokens

        with torch.autograd.set_detect_anomaly(True):
            gaot_loss = self.gaot.forward(batch=batch, tokens_pos=tokens_pos)

        self.optimization_state = {"vertices": mesh_verts, "faces": mesh_faces, "vertice_grads": None}

        pressure_field = gaot_loss["field"]

        reference_field = None
        if isinstance(gaot_loss, dict):
            for key in ("field_target", "field_gt", "target_field", "gt_field"):
                if key in gaot_loss and gaot_loss[key] is not None:
                    reference_field = gaot_loss[key]
                    break

        self._visualize_pressure_field(
            pressure_field=pressure_field,
            mesh_verts=mesh_verts,
            mesh_faces=mesh_faces,
            reference_field=reference_field,
        )

        int_drag_f = cp_field_to_drag(
            cp_field=pressure_field,
            normals=batch.c,
            face=mesh_faces.T,
            pos=batch.pos_raw,
            v_inf=self.cfd_config["v_inf"],
            rho=self.cfd_config["rho"],
        )
        self._log_front_back_pressure_drag(pressure_field=pressure_field, batch=batch, mesh_faces=mesh_faces)

        with torch.no_grad():
            self._last_diagnostics = {
                "drag_value": int_drag_f.item(),
                "drag_is_nan": bool(torch.isnan(int_drag_f).any()),
                "drag_is_inf": bool(torch.isinf(int_drag_f).any()),
                "pressure_field_min": pressure_field.min().item(),
                "pressure_field_max": pressure_field.max().item(),
                "pressure_field_mean": pressure_field.mean().item(),
                "pressure_field_std": pressure_field.std().item(),
                "pressure_field_nan_count": int(torch.isnan(pressure_field).sum().item()),
                "pressure_field_inf_count": int(torch.isinf(pressure_field).sum().item()),
                "n_input_vertices": int(mesh_verts.shape[0]),
                "n_input_faces": int(mesh_faces.shape[0]),
                "normals_flipped_count": getattr(self, "_normals_flipped_count", 0),
            }
            if hasattr(batch, "scale_factors") and batch.scale_factors is not None:
                self._last_diagnostics["scale_factors"] = batch.scale_factors.detach().cpu().flatten().tolist()
            if hasattr(batch, "bbox_center") and batch.bbox_center is not None:
                self._last_diagnostics["bbox_center"] = batch.bbox_center.detach().cpu().flatten().tolist()

        shearforce = self.shear_force_calculator.calculate_shear_force(mesh_verts, mesh_faces)
        wandb.log({"cfd/Shearforce:": shearforce}, step=self.epoch_name)
        wandb.log({"cfd/int_drag_f": int_drag_f}, step=self.epoch_name)
        return int_drag_f + shearforce

    def _reformat_data(self, mesh_verts, mesh_faces):
        optim_verts = mesh_verts.to(self.device)
        mesh_faces = mesh_faces.to(self.device)

        # Compute normals in the original/native SDF frame
        v0 = optim_verts[mesh_faces[:, 0]]
        v1 = optim_verts[mesh_faces[:, 1]]
        v2 = optim_verts[mesh_faces[:, 2]]
        face_centers = (v0 + v1 + v2) / 3

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=-1)

        sdf_values = self.sdf_model(face_centers + face_normals * 50).squeeze(-1)
        normals_flipped = sdf_values < 0
        face_normals[normals_flipped] *= -1
        self._normals_flipped_count = int(normals_flipped.sum().item())

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

    def _check_watertightness(self, verts, faces) -> None:
        # Check number of vertecies and faces
        v = verts.shape[0]
        f = faces.shape[0]
        d = v - f / 2

        if d == 2:
            print("CFD-AI Check: Mesh is likely watertight (Euler characteristic check passed).")
        elif d > 2:
            if d > 12 and self.cfd_config.get("strict_watertightness_check", False):
                print(
                    "You can disable the strict watertightness check in the config to avoid this error and only get a warning instead. But if you do so, CFD-AI does not guarantee any reasonable behavior on non-watertight meshes. Proceed with caution and consider improving the mesh quality."
                )
                raise ValueError(f"CFD-AI Check: Mesh is likely not watertight (Euler characteristic check failed with d={d}).")
            else:
                warnings.warn(
                    f"CFD-AI Check: Mesh may not be watertight (Euler characteristic check failed with d={d}).", stacklevel=2
                )
        elif d < -8 and self.cfd_config.get("strict_watertightness_check", False):
            print(
                "You can disable the strict watertightness check in the config to avoid this error and only get a warning instead. But if you do so, CFD-AI does not guarantee any reasonable behavior on non-watertight meshes. Proceed with caution and consider improving the mesh quality."
            )
            raise ValueError(f"CFD-AI Check: Mesh is likely not watertight (Euler characteristic check failed with d={d}).")
        else:
            warnings.warn(
                f"CFD-AI Check: Mesh may not be watertight (Euler characteristic check failed with d={d}).", stacklevel=2
            )
        wandb.log({"cfd/Mesh Quality (Target:2)": d}, step=self.epoch_name)
