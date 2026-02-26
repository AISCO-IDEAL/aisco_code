import logging
import os

import torch
import yaml
from torch_geometric.data import Batch, Data

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT_working.src.model.gaot_3d import GAOT3D
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT_working.src.utils.kaolin_utils import get_diff_mesh

logger = logging.getLogger(__name__)

# Each entry maps to (checkpoint_path, norm_stats_path).
# norm_stats_path is the *_norm_stats.pt produced during training for the
# dataset the checkpoint was trained on.  It contains per_head_stats with
# mean / std for "force" and "cd" so we can denormalize predictions.
_EULER_STATS = "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler_pressure/euler_pressure_field_norm_stats.pt"
_DRIVAER_STATS = "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_drivaer/drivaernet_drag_norm_stats.pt"
_COMBINED_STATS = "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_combined_multi/drivaernet_drag_norm_stats.pt"

model_dict = {
    1: ("/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_drag_results/ckpt/drag_model_58232331.pt", _DRIVAER_STATS),
    2: ("/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_drag_results/ckpt/drag_model_58245986.pt", _DRIVAER_STATS),
    3: ("/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_drag_results/ckpt/drag_model_58310187.pt", _DRIVAER_STATS),
    4: ("/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_drag_results/ckpt/overfit_model_58316029.pt", _DRIVAER_STATS),
    "Herbert": ("/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_drag_results/ckpt/multi_model_Testing.pt", _COMBINED_STATS),
    "Neural Operatinator": ("/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_drag_results/ckpt/euler_field_58577926.pt", _EULER_STATS),
}


class ModelLoader:
    def __init__(self, model_id: int, device: str) -> None:
        if model_id not in model_dict:
            raise ValueError(f"Unknown model_id={model_id}. Valid: {list(model_dict.keys())}")
        self.model_path, self.stats_path = model_dict[model_id]
        self.device = device
        self._load_model()
        self._load_norm_stats()

    def _load_model(self) -> None:
        # ── load config YAML saved alongside the checkpoint ─────────────
        config_path = self.model_path.replace(".pt", "_config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Model config not found at {config_path}. Expected a _config.yaml next to the .pt checkpoint."
            )

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # The saved YAML may be either:
        #   (a) flat: top-level keys are input_size, magno, transformer, ...
        #   (b) nested under model.args from the training arg structure
        # Support both layouts transparently.
        if "model" in config and "args" in config["model"]:
            config = config["model"]["args"]

        # ── Force online edge computation for inference ──────────────────
        # The training config may have precompute_edges=True, but at inference
        # time we build Data objects on-the-fly without precomputed edges.
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

    def _load_norm_stats(self) -> None:
        """Load per-head normalization statistics so we can denormalize predictions."""
        if not os.path.exists(self.stats_path):
            logger.warning(
                "Norm stats not found at %s — predictions will stay in normalized space.",
                self.stats_path,
            )
            self.per_head_stats = {}
            return

        stats = torch.load(self.stats_path, map_location=self.device, weights_only=False)
        self.per_head_stats = stats.get("per_head_stats", {})
        for head_name, hs in self.per_head_stats.items():
            hs["mean"] = hs["mean"].to(self.device)
            hs["std"] = hs["std"].to(self.device)
            logger.info(
                "Loaded norm stats for '%s': mean=%.4f, std=%.4f",
                head_name, hs["mean"].item(), hs["std"].item(),
            )

    def _denormalize(self, head_name: str, value: torch.Tensor) -> torch.Tensor:
        """Convert a normalized prediction back to physical units."""
        if head_name in self.per_head_stats:
            h_mean = self.per_head_stats[head_name]["mean"]
            h_std = self.per_head_stats[head_name]["std"]
            return value * h_std + h_mean
        return value

    def forward(self, sdf_model) -> torch.Tensor:
        mesh_verts, mesh_faces = get_diff_mesh(model=sdf_model)

        if mesh_verts is None or mesh_verts.shape[0] == 0:
            raise ValueError("Empty mesh generated by SDF model.")

        optim_verts = mesh_verts.detach().clone().to(self.device).requires_grad_(True)
        mesh_faces = mesh_faces.to(self.device)

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
        data = Data(pos=optim_verts, face=mesh_faces.T, c=vertex_normals)
        batch = Batch.from_data_list([data]).to(self.device)

        # ── latent token positions (registered buffer inside the model) ──
        tokens_pos = self.gaot.latent_tokens  # already on self.device

        # ── forward through GAOT3D ───────────────────────────────────────
        with torch.autograd.set_detect_anomaly(True):
            gaot_output = self.gaot(batch=batch, tokens_pos=tokens_pos)

            # gaot_output is a dict: {"force": Tensor, "cd": Tensor, ...}
            # Denormalize to get the physical drag force value
            gaot_loss = self._denormalize("force", gaot_output["force"])
            loss_to_vert_grad = torch.autograd.grad(gaot_loss, optim_verts, retain_graph=True)[0]

        # ── surrogate coupling to SDF model ──────────────────────────────
        sdf_at_verts = sdf_model(optim_verts).view(-1)
        vert_to_sdf_grad = torch.autograd.grad(
            sdf_at_verts,
            optim_verts,
            grad_outputs=torch.ones_like(sdf_at_verts),
            create_graph=True,
            retain_graph=True,
        )[0]

        dot = (loss_to_vert_grad * vert_to_sdf_grad).sum(dim=-1)
        surrogate = (dot.detach() * sdf_at_verts).sum()
        return gaot_loss + (surrogate - surrogate.detach())
