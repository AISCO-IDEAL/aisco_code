# just the loss class which calls other losses here


import importlib.util
import sys
from pathlib import Path

import torch
import trimesh
import wandb

from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel

# NOTE: You can remove all hardcoded loss imports (like BbLoss1, SolarLoss, etc.)
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.kaolin_conversions import get_diff_mesh
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.visualizer import CADVisualizer
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser_utils import render_image


class TotalLoss:
    def __init__(self, dev: str = "cuda") -> None:
        # set up configs
        self.__general_config = load_config()
        self._use_wandb = self.__general_config.get("wandb_general", {}).get("use_wandb", False)
        self._use_neural_renderer = self.__general_config.get("wandb_general", {}).get("use_neural_renderer", False)
        self.losses_config = load_config(
            v=self.__general_config["general_info"]["version"],
            cfg_name=self.__general_config["active_configs"]["losses"],
        )
        self.weights = self.losses_config.get("losses", {})  # get nested losses dict
        self.__active_losses = {}
        self._renderer_initialized = False

        project_root = Path.cwd()  # Correctly defines project root
        self.losses = self.losses_config.get("losses", {})  # get nested losses dict
        self.__active_losses = {}
        self.__rendering_frequency = self.__general_config.get("wandb_general", {}).get("rendering_frequency")

        # ALM parameters init
        if self.losses_config["general_info"].get("ALM", False):
            self.lamda: dict = {}
            self.mu: dict = {}
            self.nu: dict = {}
            self.alpha: dict = {}

            # category: aero, ... | lossnames: frontal_area_v0, ...
            for category, lossnames in self.losses.items():
                if category == "total":
                    continue

                # Initialize inner dictionaries for each category
                # Using dicts {loss_name: value} is more efficient than tuples for updates
                self.lamda[category] = {}
                self.mu[category] = {}
                self.nu[category] = {}
                self.alpha[category] = {}

                # loss_name: frontal_area_v0, ... | compute: bool | weight: float | path: str
                for loss_name, params in lossnames.items():
                    # params is [compute, weight, path]
                    if params[0]:  # if compute is True
                        # Initialize ALM parameters
                        # lambda (Lagrange multiplier) usually starts at 0
                        self.lamda[category][loss_name] = 0.0
                        # mu/nu (Penalty parameters) usually start > 0
                        self.mu[category][loss_name] = 1.0
                        self.nu[category][loss_name] = 1.0
                        # alpha parameter usually start at 1
                        self.alpha[category][loss_name] = 1.0
        # wandb init
        if self.__general_config["wandb_general"].get("use_wandb", False):
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

        # Dynamically load and initialise active losses
        # category: aero, ... | lossnames: frontal_area_v0, ...
        for category, lossnames in self.losses.items():
            if category == "total":
                continue

            # loss_name: frontal_area_v0, ... | compute: bool | weight: float | path: str
            for loss_name, (compute, _weight, path) in lossnames.items():
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

    def get_total_loss(self, model: BaseModel, epoch_name: int = -1) -> torch.Tensor:
        # Calculate the individual Losses
        individual_losses = {}  # For logging purposes
        device = next(iter(model.parameters())).device
        total_loss = torch.tensor(0.0, device=device)
        loss_tensors = []  # To store individual loss tensors for further processing

        # TODO: add config for individual losses (e.g. flexicubes resolution)
        # wandb sweeping config:
        # 1. Calculate and sum up the active losses dynamically
        for loss_name, loss_instance in self.__active_losses.items():
            # Get the loss value (which should be a weighted value if done inside the loss)
            # NOTE: This assumes the loss instances have a .get_loss(model) method.
            current_loss: torch.Tensor = loss_instance.get_loss(model)
            loss_tensors.append(current_loss)

            # Store for logging
            individual_losses[loss_name] = current_loss.item()
        if self.losses_config["general_info"]["ALM"] == "True":
            # ALM Version: L_A​(x,λ,ρ)= f(x) +λ ⋅ g(x) rho/2 ⋅ g(x)^2
            # f(x): primary loss (solar and drag)
            # g(x): constraint loss (bb, eikonal, smoothness)
            f = loss_tensors[0] + loss_tensors[4]  # solar + drag
            g = loss_tensors[1] + loss_tensors[2] + loss_tensors[3]  # bb + eikonal + smoothness
            lambda_ = 10.0
            rho = wandb.config.lagrangian_params.rho_bb
            total_loss = f + lambda_ * g + (rho / 2) * (g**2)
            # Sum all loss tensors in a differentiable way
        else:
            loss_tensors_weighted = loss_tensors * self.__weight[1]
            total_loss = torch.sum(torch.stack(loss_tensors_weighted))

        # Add the total loss to the logging dictionary
        individual_losses["total_loss"] = total_loss.item()

        # # 2. Logging them on WandB (not working as of now)
        if self._use_wandb is True:
            wandb.log(individual_losses, step=epoch_name)
            if self._use_neural_renderer and not self._renderer_initialized:
                model.init_neural_renderer(
                    image_size=self.__general_config.get("neural_renderer", {}).get("image_size", 256),
                    background_color=[0, 0, 0], #black background
                    anti_aliasing=True,
                    camera_mode='look_at',
                    viewing_angle=30,
                    device=device,
                    verbose=False
                )
                self._renderer_initialized = True

            if self.__rendering_frequency is not None and epoch_name % self.__rendering_frequency == 0:
                verts, faces = get_diff_mesh(model, grid_res=64, device=device)

                if self._use_neural_renderer:
                    # Use Alex' faster Neural Renderer for rendering and logging (no assembly though)
                    # Needs to be initialised first and converted into a gif after training
                    render_image(
                        visualiser=model.visualiser,
                        model=model,
                        position=[4000, 1000, 1000],
                        look_at=[0, 0, 0],
                        save_frame=True
                    )

                else:
                    # Use Clemens' CADVisualizer for rendering and logging
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
