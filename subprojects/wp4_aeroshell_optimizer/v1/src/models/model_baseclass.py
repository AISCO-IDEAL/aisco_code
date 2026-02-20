# just the models class which calls other models here

from __future__ import annotations

from typing import TYPE_CHECKING  # this prevents from circular imports

import torch
from torch import nn

if TYPE_CHECKING:
    from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visual_wrapper import VisualisationManager


class BaseModel(nn.Module):
    def __init__(
        self, config: dict | None = None, overwrite_base_init: bool = True, in_dim: int = 3, hidden: int = 64, layers: int = 3
    ):
        super().__init__()  # gets all the stuff from nn.Module

        self.config = config if config is not None else {}
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

    def optimise(self, renderer_manager: VisualisationManager | None = None):  # noqa: ARG002
        super().train()  # Set training attribute of inherited model to true, ensures things like e.g. dropout work properly

        # For textbook example of how our optimization works, look at basic_MLP_v1.py
        # Subclasses should implement the optimization loop and, if a wrapper is provided,
        # call renderer_wrapper.update(optimization_state=...) at the desired frequency.
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
        raise ValueError("No Debug Info implemented")

    def init_neural_renderer(self, *_, **kwargs):
        """Deprecated: The renderer is now managed by the shared Manager.

        This method is kept for backwards compatibility but no longer creates
        a per-model Visualiser. Use a shared `Manager` and call its `update()`
        during optimization instead.
        """
        if kwargs.get("verbose", False):
            print("init_neural_renderer is deprecated. Use Manager + update() calls instead.")
