# just the loss class which calls other losses here


import importlib.util
import sys
from pathlib import Path

import torch
import trimesh

import wandb

# NOTE: You can remove all hardcoded loss imports (like BbLoss1, SolarLoss, etc.)
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.kaolin_conversions import get_diff_mesh
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.visualizer import CADVisualizer


class TotalLoss:
    def __init__(self, dev: str = "cuda") -> None:
        # set up configs
        self.__general_config = load_config()
        self.losses_config = load_config(
            v=self.__general_config["general_info"]["version"],
            cfg_name=self.__general_config["active_configs"]["losses"],
        )
        self.weights = self.losses_config.get("losses", {})  # get nested losses dict
        self.__active_losses = {}

        project_root = Path.cwd()  # Correctly defines project root

        track_run = self.__general_config["wandb_general"].get("use_wandb", False)
        if track_run:
            wandb.init(
                # Set the wandb log directory
                dir="wp4_aeroshell_optimizer/data/wandb_logs",
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="aisco",
                # Set the wandb project where this run will be logged.
                project="Aeroshell Optimiser",
                # Track hyperparameters and run metadata.
                config=self.__general_config,
            )

        # category: aero, ... | losses: frontal_area_v0, ...
        for category, losses in self.weights.items():
            if category == "total":
                continue

            # loss_name: frontal_area_v0, ... | compute: bool | weight: float | path: str
            for loss_name, (compute, _weight, path) in losses.items():
                if compute:
                    # Dynamically load the loss class
                    # 1. Correctly construct the absolute path from the relative path in config
                    absolute_path = project_root / Path(path)

                    class_name = Path(path).stem.title().replace("_", "")

                    # 2. Call load_class_from_path with the absolute path string
                    loss_class = self.load_class_from_path(str(absolute_path), class_name)
                    loss_instance = loss_class(
                        device=dev,
                        loss_name=loss_name,
                        losses_config=self.losses_config,
                    )
                    self.__active_losses[loss_name] = loss_instance

        self.__rendering_frequency = self.__general_config.get("wandb_general", {}).get("rendering_frequency")

    def load_class_from_path(self, path: str, class_name: str):
        """Dynamically load a class from a file path."""
        spec = importlib.util.spec_from_file_location("module.name", path)

        # 3. Add explicit check for None to prevent AttributeError
        if spec is None or spec.loader is None:
            raise FileNotFoundError(  # noqa: TRY003
                f"Cannot find module file for dynamic loading. "
                f"Path provided: {path}. "
                f"Please verify the file exists and the path in the config (must use '/' separators)."
            )

        module = importlib.util.module_from_spec(spec)
        sys.modules["module.name"] = module
        spec.loader.exec_module(module)
        return getattr(module, class_name)

    def get_total_loss(self, model, epoch_name: int = -1) -> torch.Tensor:
        # Calculate the individual Losses
        individual_losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(model.parameters())).device)
        loss_tensors = []
        # TODO: add config for individual losses (e.g. flexicubes resolution)

        # 1. Calculate and sum up the active losses dynamically
        for loss_name, loss_instance in self.__active_losses.items():
            # Get the loss value (which should be a weighted value if done inside the loss)
            # NOTE: This assumes the loss instances have a .get_loss(model) method.
            current_loss: torch.Tensor = loss_instance.get_loss(model)
            loss_tensors.append(current_loss)

            # Store for logging
            individual_losses[loss_name] = current_loss.item()

        # Sum all loss tensors in a differentiable way
        total_loss = torch.sum(torch.stack(loss_tensors))

        # Add the total loss to the logging dictionary
        individual_losses["total_loss"] = total_loss.item()

        # # 2. Logging them on WandB (not working as of now)
        if self.__general_config.get("wandb_general", {}).get("use_wandb") is True:
            wandb.log(individual_losses, step=epoch_name)

            if epoch_name % self.__rendering_frequency == 0:
                verts, faces = get_diff_mesh(model, grid_res=64, device=next(iter(model.parameters())).device)

                # Convert to numpy if tensors
                if torch.is_tensor(verts):
                    verts = verts.detach().cpu().numpy()
                if torch.is_tensor(faces):
                    faces = faces.detach().cpu().numpy()

                # Create trimesh object and export to STL
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                stl_path = Path(f"subprojects/wp4_aeroshell_optimizer/v1/logs/mesh_epoch_{epoch_name}.stl")
                stl_path.parent.mkdir(parents=True, exist_ok=True)
                mesh.export(str(stl_path))

                # Visualize the mesh using CADVisualizer
                image_path = Path(f"subprojects/wp4_aeroshell_optimizer/v1/logs/render_epoch_{epoch_name}.png")
                visualizer = CADVisualizer(stl_path=str(stl_path), out_image_path=str(image_path))
                visualizer.render_until_blue_detected()

                # Delete local files after logging to WandB
                if stl_path.exists():
                    wandb.log({"mesh_3d": wandb.Object3D(str(stl_path))}, step=epoch_name)
                    stl_path.unlink()
                if image_path.exists():
                    wandb.log({"render_image": wandb.Image(str(image_path))}, step=epoch_name)
                    image_path.unlink()

        return total_loss

        return total_loss
