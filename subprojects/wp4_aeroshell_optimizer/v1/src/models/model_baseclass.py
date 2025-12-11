# just the models class which calls other models here
import torch
from torch import nn

from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser import Visualiser


class BaseModel(nn.Module):
    def __init__(self, config: dict, overwrite_base_init: bool = True, in_dim: int = 3, hidden: int = 64, layers: int = 3):
        super().__init__()  # gets all the stuff from nn.Module

        self.config = config
        self.in_dim = in_dim
        self.hidden = hidden
        self.layers = layers

        self.visual_config: dict = {}

        if overwrite_base_init:
            return  # Skip base init if overwritten in subclass, avoids creating redundant layers

        # Define a simple MLP with Relu and Tanh activations
        self.model_layers = nn.ModuleList()
        self.model_layers.append(nn.Linear(in_dim, hidden))

        for i in range(layers - 1):
            self.model_layers.append(nn.Linear(hidden, hidden))
            # Alternate activations
            if i % 2:
                self.model_layers.append(nn.Tanh())
            else:
                self.model_layers.append(nn.ReLU())

        self.model_layers.append(nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 1000.0  # Scale input to meters
        for layer in self.model_layers:
            x = layer(x)
        output = x.squeeze(-1)
        return output * 1000.0

    def optimise(self):
        super().train()  # Set training attribute of inherited model to true, ensures things like e.g. dropout work properly

        # For textbook example of how our optimization works, look at basic_MLP_v1.py
        raise NotImplementedError("Train method not implemented yet in BaseModel. Please implement in the subclass inheriting.")

    def validation(self):
        super().eval()  # Set training attribute of inherited model to false, ensures things like e.g. dropout are disabled
        # Validation Logic TODO

    def get_model_info_dict(self):
        # Return model info as a dictionary
        model_info = {
            "in_dim": self.in_dim,
            "hidden": self.hidden,
            "layers": self.layers,
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }
        return model_info

    def debug_info(self):
        raise ValueError("No Debug Info implemented")  # noqa: TRY003

    def init_neural_renderer(
        self,
        renderer_config: dict | None = None,
        image_size: int = 256,
        background_color: list[float] = [0, 0, 0],
        anti_aliasing: bool = True,
        camera_mode: str = "look_at",
        viewing_angle: float = 30,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = False,
    ):
        """
        Initialises a Neural Renderer via the Visualiser class and stores it as an attribute.

        Parameters:
            image_size (int): Size of the rendered image.
            background_color (list[float]): Background color of the rendered image (RGB values in range [0, 1]).
            anti_aliasing (bool): Whether to use anti-aliasing.
            camera_mode (str): Camera mode for rendering.
            viewing_angle (float): Viewing angle for the camera. (Will be overwritten if you give position in functions later)
            device (str): Device to run the renderer on ("cuda" or "cpu").

            If you need more control, just initialise the Visualiser class manually as an attribute.
        """
        if renderer_config is None:
            renderer_config = {}

        self.visualiser = Visualiser(
            image_size=image_size,
            background_color=background_color,
            anti_aliasing=anti_aliasing,
            camera_mode=camera_mode,
            viewing_angle=viewing_angle,
            device=device,
        )
        if verbose:
            print(f"Neural Renderer for model initialized with image size {image_size} on device {device}.")
