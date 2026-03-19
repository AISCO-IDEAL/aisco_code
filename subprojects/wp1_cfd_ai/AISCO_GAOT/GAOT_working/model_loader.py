import ast
import logging
import os

import pandas as pd
import torch
import torch.nn as nn  # noqa: PLR0402
import yaml
from torch_geometric.data import Batch, Data

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT_working.src.dataloader.pyg_transforms import RescalePositionNew
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT_working.src.model.gaot_3d import GAOT3D
from subprojects.wp3_solar_estimator.mesh_extractors.mesh_sdf.mesh_sdf_extractor import MeshSDFExtractor

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, cfd_config: dict, device: str) -> None:
        model_id = cfd_config["model_id"]
        self.model_path = self.get_checkpoint_path(model_id)
        self.device = device
        self._load_model()
        self.extractor = MeshSDFExtractor(
            resolution=128, bbox=[-5e3, 5e3, -5e3, 5e3, -5e3, 5e3], device=self.device, isovalue=0.0
        )

    def get_checkpoint_path(self, model_id):
        df = pd.read_csv("/cluster/work/fuge/AISCO/wp1_cfd_ai/results/metrics.csv")

        for _, row in df.iterrows():
            try:
                setup_dict = ast.literal_eval(row["setup"])
                if str(setup_dict.get("id")) == str(model_id):
                    path_dict = ast.literal_eval(row["path"])
                    return path_dict.get("ckpt_path")
            except Exception:
                continue

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

        self.normal_model = AutoNormDenorm(
            core_model=self.gaot,
            device=self.device,
            per_head_stats={
                "force": {
                    "mean": 95.75044250488281,
                    "std": 105.13729858398438,
                }
            },
        )

    def forward(self, sdf_model, epoch_name):
        self.epoch_name = epoch_name
        _, _, loss, surrogate_loss, _ = self.extractor.extract_and_compute_loss(sdf_model=sdf_model, loss_fn=self.loss_fn)
        return loss.item() + surrogate_loss - surrogate_loss.item()

    def loss_fn(self, mesh_verts, mesh_faces):
        if mesh_verts is None or mesh_verts.shape[0] == 0:
            print("Fail")
            return torch.tensor(0.0, device=self.device)

        batch = self._reformat_data(mesh_verts, mesh_faces)

        # ── latent token positions (registered buffer inside the model) ──
        tokens_pos = self.gaot.latent_tokens  # already on self.device

        # ── forward through GAOT3D ───────────────────────────────────────
        with torch.autograd.set_detect_anomaly(True):
            gaot_loss = self.normal_model.forward(data=batch, tokens_pos=tokens_pos)
        return gaot_loss

    def _reformat_data(self, mesh_verts, mesh_faces):
        optim_verts = mesh_verts.to(self.device)
        mesh_faces = mesh_faces.to(self.device)

        # ── convert vertices from mm to meters (SDF model outputs mm, GAOT expects m) ──
        MM_TO_M = 1e-3
        optim_verts_m = optim_verts * MM_TO_M
        # ── per-vertex normals (fully differentiable, no in-place ops) ──
        v0 = optim_verts[mesh_faces[:, 0]]
        v1 = optim_verts[mesh_faces[:, 1]]
        v2 = optim_verts[mesh_faces[:, 2]]

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=-1)

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


EPSILON = 1e-12


class AutoNormDenorm(nn.Module):
    """
    Matches your NormalizeFeatures transform:
      - input: normalizes data.c (optional)
      - output: de-normalizes data.y OR (data.force, data.cd) using per_head_stats
    Keeps autograd graph w.r.t. inputs (no detaching of runtime tensors).
    """

    def __init__(
        self,
        core_model: nn.Module,
        device,
        per_head_stats,  # : dict = None,  # {"force": {"mean":..., "std":...}, "cd": {...}}
    ):
        super().__init__()
        self.core = core_model
        self.device = device

        y_mean = torch.tensor(95.75044250488281, device=self.device)
        y_std = torch.tensor(105.13729858398438, device=self.device)
        c_mean = torch.tensor([-0.09271910786628723, 0.0027784164994955063, 0.04134528338909149], device=self.device)
        c_std = torch.tensor([0.4258987605571747, 0.6058956980705261, 0.6641716361045837], device=self.device)

        def buf(name, t, persistent=True):
            if t is None:
                self.register_buffer(name, None, persistent=False)
            else:
                self.register_buffer(name, torch.as_tensor(t).clone().detach(), persistent=persistent)

        buf("y_mean", y_mean)
        buf("y_std", y_std)
        buf("c_mean", c_mean)
        buf("c_std", c_std)

        # per-head stats as buffers too
        self.per_head = {}
        per_head_stats = per_head_stats or {}
        for head in ("force", "cd"):
            if head in per_head_stats:
                buf(f"{head}_mean", per_head_stats[head]["mean"])
                buf(f"{head}_std", per_head_stats[head]["std"])
                self.per_head[head] = True
            else:
                buf(f"{head}_mean", None, persistent=False)
                buf(f"{head}_std", None, persistent=False)
                self.per_head[head] = False

    def _norm(self, t, mean, std):
        return (t - mean) / (std + EPSILON)

    def _denorm(self, t, mean, std):
        return t * (std + EPSILON) + mean

    def forward(self, data, tokens_pos=None):
        # ---- normalize inputs (only what you normalized during training) ----
        if getattr(data, "c", None) is not None and self.c_mean is not None and self.c_std is not None:
            data.c = self._norm(data.c, self.c_mean, self.c_std)

        # ---- run model ----
        out = self.core(data, tokens_pos=tokens_pos) if tokens_pos is not None else self.core(data)

        out["force"] = self._denorm(out["force"], self.force_mean, self.force_std)

        return out["force"]
