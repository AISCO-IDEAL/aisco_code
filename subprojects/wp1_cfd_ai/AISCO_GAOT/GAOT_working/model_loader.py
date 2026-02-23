import torch

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT_working.src.model.gaot_3d import (
    GAOT3D,  # Assume GAOT produces a scalar loss with gradients w.r.t. verts/faces
)
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT_working.src.utils.kaolin_utils import get_diff_mesh


class ModelLoader:
    def __init__(self, model_path: str, device: str) -> None:
        self.model_path = model_path
        self.device = device
        self._load_model()

    def _load_model(self) -> None:
        self.gaot = GAOT3D().to(self.device)
        if self.model_path:
            state = torch.load(self.model_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                self.gaot.load_state_dict(state["state_dict"])
            elif isinstance(state, dict):
                self.gaot.load_state_dict(state)

    def forward(self, sdf_model) -> torch.Tensor:
        mesh_verts, mesh_faces = get_diff_mesh(model=sdf_model)

        optim_verts = mesh_verts.detach().clone().to(self.device).requires_grad_(True)

        mesh_faces = mesh_faces.to(self.device)

        gaot_loss = self.gaot.forward(batch=(optim_verts, mesh_faces))

        loss_to_vert_grad = torch.autograd.grad(gaot_loss, optim_verts, retain_graph=True)[0]

        # Push gradients into sdf_model weights via surrogate coupling
        sdf_at_verts = sdf_model(optim_verts).view(-1)  # SDF per vertex
        vert_to_sdf_grad = torch.autograd.grad(
            sdf_at_verts, optim_verts, grad_outputs=torch.ones_like(sdf_at_verts), create_graph=True, retain_graph=True
        )[0]

        # Surrogate term to transfer GAOT vertex gradients onto sdf_model parameters
        loss_dot_vert_to_sdf_grad = (loss_to_vert_grad * vert_to_sdf_grad).sum(dim=-1)
        surrogate_coupling = (loss_dot_vert_to_sdf_grad.detach() * sdf_at_verts).sum()

        # Compose final loss that keeps GAOT loss value and provides gradients to sdf_model weights
        coupled_loss = gaot_loss + (surrogate_coupling - surrogate_coupling.detach())
        return coupled_loss


# test the model loader

if __name__ == "__main__":
    from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT_working.src.model.sdf_model import SDFModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_loader = ModelLoader(
        model_path="/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_drag_results/ckpt/drag_model.pt", device="cuda"
    )
    sdf_model = SDFModel().to(device)

    loss = model_loader.forward(sdf_model)
    print("Coupled Loss:", loss.item())
