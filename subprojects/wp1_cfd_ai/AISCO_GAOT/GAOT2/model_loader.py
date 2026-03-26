import ast
import logging
import os

import pandas as pd
import torch
import torch.nn as nn  # noqa: PLR0402
import yaml
from torch_geometric.data import Batch, Data

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.src.dataloader.pyg_transforms import RescalePositionNew
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.src.model.gaot_3d import GAOT3D
from subprojects.wp3_solar_estimator.mesh_extractors.mesh_sdf.mesh_sdf_extractor import MeshSDFExtractor
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.detached_modes.mesh_sdf_grads import MeshSDFGrads
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser import Visualiser
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.src.utils.pressure_integration import pressure_to_drag

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, cfd_config: dict, device: str) -> None:
        model_id = cfd_config["model_id"]
        self.model_path = self._get_checkpoint_path(model_id)
        self.device = device
        self._load_model()
        self.extractor = MeshSDFExtractor(
            resolution=256, bbox=[-3e3, 3e3, -3e3, 3e3, -3e3, 3e3], device=self.device, isovalue=0.0
        )
        visualiser = Visualiser(device=self.device)
        self.gradlogger = MeshSDFGrads(visualiser, intensity=10)

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
        config_path = self.model_path.replace(".pt", "_config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Model config not found at {config_path}. Expected a _config.yaml next to the .pt checkpoint."
            )

        with open(config_path) as f:
            config = yaml.safe_load(f)

        if "model" in config and "args" in config["model"]:
            config = config["model"]["args"]

        if "magno" in config:
            config["magno"]["precompute_edges"] = False

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
        _, _, loss, surrogate_loss, dL_dvertices = self.extractor.extract_and_compute_loss(
            sdf_model=sdf_model, loss_fn=self.loss_fn
        )
        
        self.optimization_state["vertice_grads"] = dL_dvertices
        self.gradlogger.log_single_image_to_wandb(self.optimization_state, epoch_name=self.epoch_name)

        return loss.item() + surrogate_loss - surrogate_loss.item()

    def loss_fn(self, mesh_verts, mesh_faces):
        print(f"Number of Mesh Verts:{mesh_faces.shape[0]}")
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

        pressure_field = gaot_loss["field"]

        int_drag_f = pressure_to_drag(pred_pressure=pressure_field, normals=batch.c, face=mesh_faces.T, pos=mesh_verts*1e-3)
        print(f"Force-Value:{int_drag_f.item()}")
        return int_drag_f

    def _reformat_data(self, mesh_verts, mesh_faces):
        optim_verts = mesh_verts.to(self.device)
        mesh_faces = mesh_faces.to(self.device)

        # ── convert vertices from mm to meters (SDF model outputs mm, GAOT expects m) ──
        optim_verts_m = optim_verts * 1e-3
        # ── per-vertex normals (fully differentiable, no in-place ops) ──
        v0 = optim_verts[mesh_faces[:, 0]]
        v1 = optim_verts[mesh_faces[:, 1]]
        v2 = optim_verts[mesh_faces[:, 2]]
        face_centers = (v0 + v1 + v2) / 3

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=-1)

        sdf_values = self.sdf_model(face_centers*1e3+face_normals*50)
        face_normals[sdf_values < 0] *= -1
        print("dumdidum",sdf_values < 0)

        num_verts = optim_verts.shape[0]
        all_idx = mesh_faces.reshape(-1)
        all_normals = face_normals.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)

        vertex_normals = torch.zeros(num_verts, 3, device=self.device, dtype=optim_verts.dtype)
        vertex_normals = vertex_normals.index_add(0, all_idx, all_normals)
        vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=-1)

        # ── build PyG batch ──────────────────────────────────────────────
        data = Data(pos=optim_verts_m, face=mesh_faces.T, c=vertex_normals)

        # ── rescale positions to [-1, 1] using physical domain in meters ──
        rescale_transform = RescalePositionNew(lims=(-1.0, 1.0), phy_domain=([-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]))
        data = rescale_transform(data)

        return Batch.from_data_list([data]).to(self.device)
