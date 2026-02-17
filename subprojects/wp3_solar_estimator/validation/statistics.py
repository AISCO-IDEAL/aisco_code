import torch
import wandb


class Statistics:
    """
    This class should log relevant mesh data to the wandb project to help with solar/misc. debugging
    """

    def __init__(self, device: str = "cpu", use_wandb: bool = True):
        self.device = device
        self.use_wandb = use_wandb  # Set to True if you want to log to wandb, make sure to initialize wandb in your main file and set up a project

    def log_basic_mesh_stats(self, vertices: torch.Tensor, faces: torch.Tensor, epoch: int) -> None:
        """
        Logs basic mesh statistics such as number of vertices, number of faces, average edge length, etc.
        """
        if not self.use_wandb:
            return
        num_vertices = vertices.shape[0]
        num_faces = faces.shape[0]

        wandb.log({"num_vertices": num_vertices, "num_faces": num_faces}, step=epoch)
