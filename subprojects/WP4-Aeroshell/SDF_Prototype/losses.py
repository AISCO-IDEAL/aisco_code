import typing as T

import cadquery as cq
import torch
from CadQuery_to_SDF import BackupAssembly, SDFfromCadQuery
from sdf import SDFNetwork

# Define various loss functions for training the SDFNetwork
# These losses include eikonal loss, smoothness loss, bounding box constraints,
# watertightness loss, and performance-related losses.


class SDFLosses:
    def __init__(self, weights: T.List[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]):
        assert len(weights) == 6, "Weights must match the number of losses: [eikonal, smoothness, bounding_box, watertightness, fronatl_area, solar_area]"
        self.weights = weights
  
    def eikonal_loss(self, model: SDFNetwork, points: torch.Tensor) -> torch.Tensor: 
        sdf, grad_sdf = model(points) ######## DO NOT TOUCH #######
        return self.weights[0] * torch.abs(torch.norm(grad_sdf, p=2, dim=-1) - 1.0).mean() 

    def smoothness_loss(self, model: SDFNetwork, points: torch.Tensor) -> torch.Tensor:
        sdf, grad_sdf, laplacian_sdf = model(points, return_laplacian=True) ######## DO NOT TOUCH #######
        return self.weights[1] * (laplacian_sdf**2).mean() 

    def bounding_boxes_loss(self, model:SDFNetwork, inner_box:cq.Shape=None, outer_box:cq.Shape=None, pointgrid:list[list[float]]=None) -> torch.Tensor:  # noqa: RUF013
        """
        Arguments:
            model: The SDF network                                              NO Default!!
            inner_box: CadQuery Shape representing the inner bounding box       Defaults to BackupAssembly boxes if None
            outer_box: CadQuery Shape representing the outer bounding box       Defaults to BackupAssembly boxes if None
            pointgrid: List of points at which compatibility is enforced        Defaults to a 21x21x21 grid from -2 to 2 in each axis if None
        
        Returns:
            Bounding boxes loss (to be minimized)
            
        """
        total_loss = 0.0

        # Provide default boxes and pointgrid if not supplied:
        if inner_box is None: 
            backup_assembly = BackupAssembly()
            inner_box = backup_assembly.inner_solid()
        if outer_box is None:
            backup_assembly = BackupAssembly()
            outer_box = backup_assembly.outer_solid()
        if pointgrid is None:
            pointgrid = [[x/5, y/5, z/5] for x in range(-10, 11) for y in range(-10, 11) for z in range(-10, 11)]


        inner = SDFfromCadQuery(inner_box, pointgrid)
        outer = SDFfromCadQuery(outer_box, pointgrid)
        sdf_inner = inner.sdf_at_pointgrid()
        sdf_outer = outer.sdf_at_pointgrid()

        for p in pointgrid:
            sdf_value, _ = model(torch.tensor(p).unsqueeze(0).float())
            sdf_value = sdf_value.item()
            idx = pointgrid.index(p)
            total_loss += torch.relu(sdf_value - sdf_inner[idx]).item()
            total_loss += torch.relu(-sdf_value + sdf_outer[idx]).item()

        
        bounding_boxes_loss = torch.tensor(total_loss / len(pointgrid)).float()  
        return self.weights[2]* bounding_boxes_loss

    def watertightness_loss(self, model: SDFNetwork, infinity_box:list[list[float]]=None) -> torch.Tensor:  # noqa: RUF013
        total_loss = 0.0
        if infinity_box is None: # samples points on a large box around the object
            infinity_box = [[x, y, z] for x in [-10.0, 10.0] for y in [-10.0, 10.0] for z in [-10.0, 10.0, 19.9]]
            infinity_box += [[x, y, z] for x in [-10.0, 10.0] for y in [-10.0, 10.0, 19.9] for z in [-10.0, 10.0]]
            infinity_box += [[x, y, z] for x in [-10.0, 10.0, 19.9] for y in [-10.0, 10.0] for z in [-10.0, 10.0]]
            
            
        for p in infinity_box:
            sdf_value, _ = model(torch.tensor(p).unsqueeze(0).float())
            sdf_value = sdf_value.item()
            total_loss += torch.relu(-sdf_value).item()
        return self.weights[3] * torch.tensor(total_loss / len(infinity_box)).float()

    def frontal_area_loss(self, model: SDFNetwork,resolution: int=128,height: float = 4.0, width: float = 6.0) -> torch.Tensor:
        """
        Compute frontal area loss (minimize drag by reducing frontal area).
        
        Args:
            model: The SDF network
            resolution: Resolution for area computation
        
        Returns:
            Frontal area loss (to be minimized)
        """
        # Create a grid in the Y-Z plane (frontal projection)
        y_coords = torch.linspace(-width/2, width/2, resolution)
        z_coords = torch.linspace(-1.0, height, resolution)
        # z coords from -1 to height to still evaluate at least some area under the ground for now

        y_grid, z_grid = torch.meshgrid(y_coords, z_coords, indexing='ij')
        
        # Fix x at the front of the vehicle (x = 0 for frontal area)
        x_coords = torch.zeros_like(y_grid)
        
        # Create points for evaluation
        #As of yet: takes primitive area at x=0 (middle of car in theory)
        points = torch.stack([x_coords.flatten(), y_grid.flatten(), z_grid.flatten()], dim=1)
        points.requires_grad = True
        outputs = model(points)
        # Ensure only the scalar field is extracted
        if isinstance(outputs, tuple):
            sdf_values = outputs[0]  # first element = scalar SDF field
        else:
            sdf_values = outputs
        sdf_values.reshape(resolution, resolution)
        
        # Approximate area using the Heaviside function
        # Use a smooth approximation of the Heaviside function
        epsilon = 0.02
        heaviside_approx = torch.sigmoid(-sdf_values / epsilon)
        
        # Integrate over the grid to get area
        grid_spacing = (width / resolution) * ((1+height) / resolution)  # dy * dz
        frontal_area = torch.sum(heaviside_approx) * grid_spacing
        
        return self.weights[4] * frontal_area
        
    def solar_area_loss(self, model: SDFNetwork, resolution: int = 128,length: float = 8.0, width: float = 6.0) -> torch.Tensor:
        """
        Compute top area loss (maximize solar collection area on top surface).
        
        Args:
            model: The SDF network
            resolution: Resolution for area computation
        
        Returns:
            NOTE: As of now only calculates area of a given slice thorugh the car at arbitrary height!
            Negative top area (for maximization, more area = better)
        """
        # Create a grid in the X-Y plane (top projection)
        x_coords = torch.linspace(-length/2, length/2, resolution)
        y_coords = torch.linspace(-width/2, width/2, resolution)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Fix z at the top of the vehicle (z = 1.2 for top area)
        # again arbitrary height, just to get an area on top, need to be fixed later
        z_coords = torch.full_like(x_grid, 1.2)
        
        # Create points for evaluation
        points = torch.stack([x_grid.flatten(), y_grid.flatten(), z_coords.flatten()], dim=1)
        points.requires_grad = True
        outputs = model(points)
        # Ensure only the scalar field is extracted
        if isinstance(outputs, tuple):
            sdf_values = outputs[0]  # first element = scalar SDF field
        else:
            sdf_values = outputs
        sdf_values.reshape(resolution, resolution)
        # Approximate area using the Heaviside function
        epsilon = 0.02
        heaviside_approx = torch.sigmoid(-sdf_values / epsilon)
        
        # Integrate over the grid to get area
        grid_spacing = (length / resolution) * (width / resolution)  # dx * dy
        top_area = torch.sum(heaviside_approx) * grid_spacing
        
        return - self.weights[5] * top_area  # Negative for maximization
    

    def get_total_loss(self, model: SDFNetwork, points: torch.Tensor) -> torch.Tensor:
        return (
            self.eikonal_loss(model, points) +
            self.smoothness_loss(model, points) +
            self.bounding_boxes_loss(model, points) +
            self.watertightness_loss(model, points) +
            self.frontal_area_loss(model) +
            self.solar_area_loss(model)
        )