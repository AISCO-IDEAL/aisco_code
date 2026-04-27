# Basic MLP Model

import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_baseclass import BaseModel


class BasicMLP(BaseModel):
    def __init__(
        self, config: dict | None = None, overwrite_base_init: bool = False, in_dim: int = 3, hidden: int = 128, layers: int = 6
    ):
        if config is None:
            config = {}
        super().__init__(config, overwrite_base_init, in_dim, hidden, layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def get_model_info_dict(self):
        return {
            "Model_name": "BasicMLP",
            "in_dim": self.in_dim,
            "hidden": self.hidden,
            "layers": self.layers,
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }

    def debug_info(self):
        print("===================================")
        print("Model Debug Info:")
        print(self)
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters())}")
        print("Model Layers:")
        for i, layer in enumerate(self.model_layers):
            print(f" Layer {i}: {layer}")
        print("Device/Gradient Info:")
        device = next(self.parameters()).device
        print(f" Model is on device: {device}")
        print(f"Do Model Parameters require Gradients? (Y/N): {any(p.requires_grad for p in self.parameters())}")
        print("End of Model Debug Info")
        print("===================================")
