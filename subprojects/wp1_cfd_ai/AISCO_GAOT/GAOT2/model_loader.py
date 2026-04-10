import ast
import copy
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader

import wandb
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.src.dataloader.metadata import DATASET_METADATA
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.src.dataloader.pyg_transforms import RescalePosition, RescalePositionNew
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.src.model.gaot_3d import GAOT3D
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.src.utils.pressure_integration import cp_field_to_drag
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.src.utils.scale import rescale, rescale_new
from subprojects.wp3_solar_estimator.mesh_extractors.mesh_sdf.mesh_sdf_extractor import MeshSDFExtractor
from subprojects.wp4_aeroshell_optimizer import MeshSDFGrads, Visualiser

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, cfd_config: dict, device: str) -> None:
        self.cfd_config = cfd_config
        self.device = device

        self.model_path = self._get_checkpoint_path()
        self._load_model()

        self.extractor = MeshSDFExtractor(
            resolution=128,
            bbox=[-3e3, 3e3, -3e3, 3e3, -3e3, 3e3],
            device=self.device,
            isovalue=0.0,
            target_nr_vertices=int(cfd_config["mesh_resolution"]),
            max_iters=1000,
            gradient_max=cfd_config["gradient_max"],
        )

        self.visualiser = Visualiser(device=self.device)
        self.gradlogger = MeshSDFGrads(self.visualiser, intensity=10)

    def _get_checkpoint_path(self):
        df = pd.read_csv("/cluster/work/fuge/AISCO/wp1_cfd_ai/results/metrics.csv")
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

        # Fix 1: keep objective sign consistent with Variant A
        cuppled_loss = loss.detach() + surrogate_loss - surrogate_loss.detach()
        assert cuppled_loss == loss.item() + surrogate_loss - surrogate_loss.item()
        return cuppled_loss

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

        sample = Data(
            pos=mesh_verts.detach() * 1e-3,
            x=None,
            face=mesh_faces.detach().T,
        )

        loader = PyGDataLoader([sample], batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(self.device)

        coords = batch.pos.detach().cpu().numpy()
        fname = self.epoch_name

        if pressure_field.dim() == 2 and pressure_field.size(-1) == 1:
            pressure_field = pressure_field.squeeze(-1)

        field_pred = pressure_field.detach().cpu().numpy().squeeze()
        field_gt = None

        if field_gt is not None:
            u_mean = getattr(self, "u_mean", torch.tensor(0.0, device=self.device))
            u_std = getattr(self, "u_std", torch.tensor(1.0, device=self.device))
            if isinstance(u_mean, torch.Tensor):
                u_mean = u_mean.detach().cpu().numpy().item() if u_mean.numel() == 1 else 0.0
            if isinstance(u_std, torch.Tensor):
                u_std = u_std.detach().cpu().numpy().item() if u_std.numel() == 1 else 1.0

            field_pred_dn = field_pred * u_std + u_mean
            field_gt_dn = field_gt * u_std + u_mean
            field_diff = np.abs(field_gt_dn - field_pred_dn)

            common_min = float(np.min(field_gt_dn))
            common_max = float(np.max(field_gt_dn))
        else:
            field_pred_dn = field_pred
            field_gt_dn = None
            field_diff = None
            common_min = float(np.min(field_pred_dn))
            common_max = float(np.max(field_pred_dn))

        panels = []
        if field_gt_dn is not None:
            panels.append((field_gt_dn, "Ground Truth", "jet", common_min, common_max))
        panels.append((field_pred_dn, "Prediction", "jet", common_min, common_max))
        if field_diff is not None:
            panels.append((field_diff, "Absolute Difference", "Reds", None, None))

        fig = plt.figure(figsize=(20, 6))
        for panel_idx, (data, title, cmap_name, vmin, vmax) in enumerate(panels):
            ax = fig.add_subplot(1, len(panels), panel_idx + 1, projection="3d")
            sc = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=data,
                cmap=cmap_name,
                s=2,
                vmin=vmin,
                vmax=vmax,
                depthshade=False,
            )
            ax.set_title(title)
            fig.colorbar(sc, ax=ax, shrink=0.5)
            ax.view_init(elev=20, azim=-120)

        fig.suptitle(f"Pressure Field [opt] {fname} — {self.epoch_name}", fontsize=13)
        plt.tight_layout()

        wandb.log(
            {
                "Pressure-Field visualisation": wandb.Image(
                    fig,
                    caption=f"opt {fname} {self.epoch_name}",
                )
            },
            step=int(self.epoch_name),
        )
        plt.close(fig)

    def loss_fn(self, mesh_verts, mesh_faces):
        wandb.log({"Aero-resolution": mesh_verts.shape[0]}, step=self.epoch_name)
        if mesh_verts is None or mesh_verts.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)

        batch = self._reformat_data(mesh_verts, mesh_faces)

        # Fix 5: use reconstructed latent tokens from training context
        tokens_pos = self._latent_tokens

        with torch.autograd.set_detect_anomaly(True):
            gaot_loss = self.gaot.forward(batch=batch, tokens_pos=tokens_pos)

        self.optimization_state = {"vertices": mesh_verts, "faces": mesh_faces, "vertice_grads": []}

        pressure_field = gaot_loss["field"]

        self._visualize_pressure_field(pressure_field=pressure_field, mesh_verts=mesh_verts, mesh_faces=mesh_faces)

        # Fix 4: use physical/raw normals for drag integration
        integration_normals = batch.c_raw if hasattr(batch, "c_raw") and batch.c_raw is not None else batch.c

        int_drag_f = cp_field_to_drag(
            cp_field=pressure_field,
            normals=integration_normals,
            face=mesh_faces.T,
            pos=mesh_verts * 1e-3,
            v_inf=self.cfd_config["v_inf"],
            rho=self.cfd_config["rho"],
        )

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

        return int_drag_f

    def _reformat_data(self, mesh_verts, mesh_faces):
        # Fixes 2, 3, 4:
        # - keep physical/original frame for normal computation and SDF probing
        # - do not introduce the *1e3 bug in SDF probing
        # - preserve raw normals for drag integration
        # - only flip x for model input frame

        optim_verts = mesh_verts.to(self.device)
        mesh_faces = mesh_faces.to(self.device)

        # Compute normals in the original/native SDF frame
        v0 = optim_verts[mesh_faces[:, 0]]
        v1 = optim_verts[mesh_faces[:, 1]]
        v2 = optim_verts[mesh_faces[:, 2]]
        face_centers = (v0 + v1 + v2) / 3

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=-1)

        # Fix 3: no erroneous *1e3 here; SDF probe stays in native mesh/SDF units
        sdf_values = self.sdf_model(face_centers + face_normals * 50)
        normals_flipped = sdf_values < 0
        face_normals[normals_flipped] *= -1
        self._normals_flipped_count = int(normals_flipped.sum().item())

        num_verts = optim_verts.shape[0]
        all_idx = mesh_faces.reshape(-1)
        all_normals = face_normals.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)

        vertex_normals = torch.zeros(num_verts, 3, device=self.device, dtype=optim_verts.dtype)
        vertex_normals = vertex_normals.index_add(0, all_idx, all_normals)
        vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=-1)

        # Fix 4: preserve physical normals for force integration
        raw_vertex_normals = vertex_normals.clone()

        # Fix 2: convert to model/training frame only for network input
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
            data = RescalePositionNew(
                lims=(-1.0, 1.0),
                phy_domain=self._phy_domain,
            )(data)
        else:
            data = RescalePosition()(data)

        return Batch.from_data_list([data]).to(self.device)
