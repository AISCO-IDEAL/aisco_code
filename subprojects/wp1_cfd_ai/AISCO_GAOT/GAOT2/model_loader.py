import ast
import copy
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader

import wandb
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.src.dataloader.pyg_transforms import RescalePosition, RescalePositionNew
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.src.model.gaot_3d import GAOT3D
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.src.utils.pressure_integration import cp_field_to_drag
from subprojects.wp3_solar_estimator.mesh_extractors.mesh_sdf.mesh_sdf_extractor import MeshSDFExtractor
from subprojects.wp4_aeroshell_optimizer import MeshSDFGrads, Visualiser

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, cfd_config: dict, device: str) -> None:
        model_id = cfd_config["model_id"]
        self.rendering = cfd_config["visualisations"]
        self.v_inf = float(cfd_config.get("v_inf", 25.0))
        self.rho = float(cfd_config.get("rho", 1.225))
        self.model_path = self._get_checkpoint_path(model_id)
        self.device = device
        self._load_model()
        print("initialising Extractor")
        self.extractor = MeshSDFExtractor(
            resolution=128,
            bbox=[-3e3, 3e3, -3e3, 3e3, -3e3, 3e3],
            device=self.device,
            isovalue=0.0,
            target_nr_vertices=int(cfd_config["mesh_resolution"]),
            max_iters=1000,
            gradient_max=cfd_config["gradient_max"],
        )
        # keep visualiser on self so we can reuse it
        self.visualiser = Visualiser(device=self.device)
        self.gradlogger = MeshSDFGrads(self.visualiser, intensity=10)

    def _get_checkpoint_path(self, model_id):
        df = pd.read_csv("/cluster/work/fuge/AISCO/wp1_cfd_ai/results/metrics.csv")
        for _, row in df.iterrows():
            setup_dict = ast.literal_eval(row["setup"])
            if str(setup_dict.get("id")) == str(model_id):
                path_dict = ast.literal_eval(row["path"])
                return path_dict.get("ckpt_path")
        raise ValueError("Wrong ID for GAOT-Model")

    def _load_model(self) -> None:
        # ── load config YAML saved alongside the checkpoint ─────────────
        config_path = self.model_path.replace("model.pt", "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Model config not found at {config_path}. Expected a _config.yaml next to the .pt checkpoint."
            )

        with open(config_path) as f:
            config = yaml.safe_load(f)

        print(config)
        self._og_config = copy.deepcopy(config)

        if "model" in config and "args" in config["model"]:
            config = config["model"]["args"]

        if "magno" in config:
            config["magno"]["precompute_edges"] = False

        if type(config["transformer"]["norm_eps"]) is str:
            config["transformer"]["norm_eps"] = float(config["transformer"]["norm_eps"])

        # ── build model from config dict ─────────────────────────────────
        self.gaot = GAOT3D(config=config).to(self.device)
        self.gaot.eval()

        # ── load weights ─────────────────────────────────────────────────
        state = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if "model" in state:
            sd = state["model"]
        elif "state_dict" in state:
            sd = state["state_dict"]
        else:
            sd = state

        # Strip "module." prefix from DDP/DataParallel checkpoints
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        missing, unexpected = self.gaot.load_state_dict(sd, strict=False)
        if unexpected:
            logger.warning(
                "Ignoring %d unexpected key(s) in checkpoint: %s",
                len(unexpected),
                unexpected,
            )
        if missing:
            logger.warning(
                "Missing %d key(s) in checkpoint (using init values): %s",
                len(missing),
                missing,
            )

    def forward(self, sdf_model, epoch_name):
        self.epoch_name = epoch_name
        self.sdf_model = sdf_model
        verts, faces, loss, surrogate_loss, dL_dvertices = self.extractor.extract_and_compute_loss(
            sdf_model=sdf_model, loss_fn=self.loss_fn
        )

        # Store diagnostics for DebugManager
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

        if self.rendering:
            self.gradlogger.log_single_image_to_wandb(self.optimization_state, epoch_name=self.epoch_name)

        cuppled_loss = loss.detach() + surrogate_loss - surrogate_loss.detach()
        assert cuppled_loss == loss.item() + surrogate_loss - surrogate_loss.item()
        return cuppled_loss

    def _visualize_pressure_field(
        self,
        pressure_field: torch.Tensor,
        mesh_verts: torch.Tensor,
        mesh_faces: torch.Tensor,
    ) -> None:
        """
        Visualise the predicted pressure field in the same style as
        StaticTrainer3D._run_viz_for_sample (field part).

        Uses a temporary PyG Data object + PyGDataLoader and logs a 3-panel
        3D scatter figure (GT, Pred, |Diff|) to self.tracker.
        """
        if not self.rendering:
            return
        if pressure_field is None:
            return

        # ---- build a synthetic "sample" compatible with trainer logic ----
        # vertices are in mm from SDF pipeline; convert to m as in GAOT training
        sample = Data(
            pos=mesh_verts.detach() * 1e-3,  # mm -> m
            x=None,  # no GT field on this mesh
            face=mesh_faces.detach().T,
        )

        loader = PyGDataLoader([sample], batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(self.device)

        # coordinates for scatter plot  (DETACH before numpy)
        coords = batch.pos.detach().cpu().numpy()
        fname = getattr(self, "epoch_name", "mesh")

        # predicted field is passed in as pressure_field (node-wise GAOT output)
        if pressure_field.dim() == 2 and pressure_field.size(-1) == 1:
            pressure_field = pressure_field.squeeze(-1)

        field_pred = pressure_field.detach().cpu().numpy().squeeze()
        field_gt = None  # no GT available on optimization mesh

        # if you *do* have GT on this mesh in future, set batch.x and use it here
        if field_gt is not None:
            # optional: same denorm as StaticTrainer3D if you store stats on ModelLoader
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
            # no GT: just plot the prediction with its own range
            field_pred_dn = field_pred
            field_gt_dn = None
            field_diff = None
            common_min = float(np.min(field_pred_dn))
            common_max = float(np.max(field_pred_dn))

        # ---- build matplotlib figure in same style as trainer ------------

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

        # we don't have a numeric epoch here, only a name
        title_epoch = getattr(self, "epoch_name", "opt_step")
        fig.suptitle(f"Pressure Field [opt] {fname} — {title_epoch}", fontsize=13)
        plt.tight_layout()

        step_val = int(self.epoch_name)

        wandb.log(
            {
                "Pressure-Field visualisation": wandb.Image(
                    fig,
                    caption=f"opt {fname} {title_epoch}",
                )
            },
            step=step_val,
        )
        plt.close(fig)

    def loss_fn(self, mesh_verts, mesh_faces):
        wandb.log({"Aero-resolution": mesh_verts.shape[0]}, step=self.epoch_name)
        if mesh_verts is None or mesh_verts.shape[0] == 0:
            print("Fail")
            return torch.tensor(0.0, device=self.device)
        batch = self._reformat_data(mesh_verts, mesh_faces)

        # ── latent token positions (registered buffer inside the model) ──
        tokens_pos = self.gaot.latent_tokens

        # ── forward through GAOT3D ───────────────────────────────────────
        with torch.autograd.set_detect_anomaly(True):
            gaot_loss = self.gaot.forward(batch=batch, tokens_pos=tokens_pos)

        self.optimization_state = {"vertices": mesh_verts, "faces": mesh_faces, "vertice_grads": []}

        # predicted pressure field from GAOT (key name "field" as in your code)
        pressure_field = gaot_loss["field"]

        # visualize pressure on mesh every call
        self._visualize_pressure_field(pressure_field=pressure_field, mesh_verts=mesh_verts, mesh_faces=mesh_faces)

        int_drag_f = cp_field_to_drag(
            cp_field=pressure_field,
            normals=batch.c,
            face=mesh_faces.T,
            pos=mesh_verts * 1e-3,
            v_inf=self.v_inf,
            rho=self.rho,
        )

        # Store diagnostics
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

        print(f"Force-Value:{int_drag_f.item()}")
        return int_drag_f

    def _reformat_data(self, mesh_verts, mesh_faces):
        optim_verts = mesh_verts.to(self.device)
        mesh_faces = mesh_faces.to(self.device)

        # ── flip geometry in x ───────────────────────────────────────────
        optim_verts = optim_verts.clone()
        optim_verts[:, 0] *= -1.0

        # ── convert vertices from mm to meters (SDF model outputs mm, GAOT expects m) ──
        optim_verts_m = optim_verts * 1e-3
        # ── per-vertex normals (fully differentiable, no in-place ops) ──
        v0 = optim_verts[mesh_faces[:, 0]]
        v1 = optim_verts[mesh_faces[:, 1]]
        v2 = optim_verts[mesh_faces[:, 2]]
        face_centers = (v0 + v1 + v2) / 3

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=-1)

        sdf_values = self.sdf_model(face_centers * 1e3 + face_normals * 50)
        normals_flipped = sdf_values < 0
        face_normals[normals_flipped] *= -1
        self._normals_flipped_count = int(normals_flipped.sum().item())
        print("dumdidum", normals_flipped)

        num_verts = optim_verts.shape[0]
        all_idx = mesh_faces.reshape(-1)
        all_normals = face_normals.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)

        vertex_normals = torch.zeros(num_verts, 3, device=self.device, dtype=optim_verts.dtype)
        vertex_normals = vertex_normals.index_add(0, all_idx, all_normals)
        vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=-1)

        # optional: also flip normals' x-component to be consistent
        vertex_normals[:, 0] *= -1.0

        # ── build PyG batch ──────────────────────────────────────────────
        data = Data(pos=optim_verts_m, face=mesh_faces.T, c=vertex_normals)

        # ── rescale positions to [-1, 1] using physical domain in meters ──
        if self._og_config["dataset"]["use_rescale_new"]:
            rescale_transform = RescalePositionNew(lims=(-1.0, 1.0), phy_domain=([-5.0, -1.3, -1.2], [4.7, 1.7, 1.6]))
        else:
            rescale_transform = RescalePosition()
        data = rescale_transform(data)

        return Batch.from_data_list([data]).to(self.device)
