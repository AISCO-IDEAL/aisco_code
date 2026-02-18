import Torch

class ModelLoader:
    def __init__(self, model_path:str) -> None:
        self.model_path = model_path
        self.gaot = None

    def forward(self, sdf_model) -> Torch.tensor:
        pass