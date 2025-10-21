import torch
from sdf import SDFNetwork
import typing as T
# Define various loss functions for training the SDFNetwork
# These losses include eikonal loss, smoothness loss, bounding box constraints,
# watertightness loss, and performance-related losses.

class SDFLosses:
    def __init__(self, weights: T.List[float] = [1.0, 1.0, 1.0, 1.0, 1.0]):
        assert len(weights) == 5, "Weights must match the number of losses: [eikonal, smoothness, bounding_box, watertightness, performance]"
        self.weights = weights
  
    def eikonal_loss(self, model: SDFNetwork, points: torch.Tensor) -> torch.Tensor: 
        sdf, grad_sdf = model(points) ######## DO NOT TOUCH #######
        return self.weights[0] * torch.abs(torch.norm(grad_sdf, p=2, dim=-1) - 1.0).mean() 

    def smoothness_loss(self, model: SDFNetwork, points: torch.Tensor) -> torch.Tensor:
        sdf, grad_sdf, laplacian_sdf = model(points, return_laplacian=True) ######## DO NOT TOUCH #######
        return self.weights[1] * (laplacian_sdf**2).mean() 

    def bounding_boxes_loss(self, model: SDFNetwork, points: torch.Tensor) -> torch.Tensor:
        sdf, _ = model(points)
        return self.weights[2] * (sdf**2).mean()

    def watertightness_loss(self, model: SDFNetwork, points: torch.Tensor) -> torch.Tensor:
        sdf, _ = model(points)
        return self.weights[3] * (sdf**2).mean()

    def performance_loss(self, model: SDFNetwork, points: torch.Tensor) -> torch.Tensor:
        # TODO: Replace with actual solar/drag performance loss
        sdf, _ = model(points)
        return self.weights[4] * (sdf**2).mean()

    def get_total_loss(self, model: SDFNetwork, points: torch.Tensor) -> torch.Tensor:
        return (
            self.eikonal_loss(model, points) +
            self.smoothness_loss(model, points) +
            self.bounding_boxes_loss(model, points) +
            self.watertightness_loss(model, points) +
            self.performance_loss(model, points)
        )