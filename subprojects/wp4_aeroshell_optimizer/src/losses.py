
import cadquery as cq
import torch
from CadQuery_to_SDF import BackupAssembly, SDFfromCadQuery
from sdf import SDFNetwork

# Define various loss functions for training the SDFNetwork
# These losses include eikonal loss, smoothness loss, bounding box constraints,
# watertightness loss, and performance-related losses.


class SDFLosses:
    def __init__(self, weights: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]):
        assert len(weights) == 6, "Weights must match the number of losses: [eikonal, smoothness, bounding_box, watertightness, fronatl_area, solar_area]"
        self.weights = weights


    ### DO NOT TOUCH WORK FROM EMANUEL ###
    def eikonal_loss(self, model: SDFNetwork, points: torch.Tensor) -> torch.Tensor:
        sdf, grad_sdf = model(points)
        return self.weights[0] * torch.abs(torch.norm(grad_sdf, p=2, dim=-1) - 1.0).mean()
    ################################
    ### DO NOT TOUCH WORK FROM EMANUEL ###
    def smoothness_loss(self, model: SDFNetwork, points: torch.Tensor) -> torch.Tensor:
        sdf, grad_sdf, laplacian_sdf = model(points, return_laplacian=True) 
        return self.weights[1] * (laplacian_sdf**2).mean()
    ################################
    
    def bounding_boxes_loss(
        self,
        model: SDFNetwork,
        inner_box: cq.Shape = None,
        outer_box: cq.Shape = None,
        pointgrid: list[list[float]] = None,
        ) -> torch.Tensor:
        """
        Arguments:
            model: The SDF network                                              NO Default!!
            inner_box: CadQuery Shape representing the inner bounding box       Defaults to BackupAssembly boxes if None
            outer_box: CadQuery Shape representing the outer bounding box       Defaults to BackupAssembly boxes if None
            pointgrid: List of points at which compatibility is enforced        Defaults to a 21x21x21 grid from -2 to 2 in each axis if None
        Returns:
            Bounding boxes loss (to be minimized)

        Constrains model SDF to lie between outer and inner SDFs:
            sdf_outer(p) <= sdf_model(p) <= sdf_inner(p)
        Loss = mean( relu(sdf_model - sdf_inner) + relu(sdf_outer - sdf_model) )
        """
        # Defaults (build backup assembly once)
        if inner_box is None or outer_box is None:
            backup_assembly = BackupAssembly()
            if inner_box is None:
                inner_box = backup_assembly.inner_solid()
            if outer_box is None:
                outer_box = backup_assembly.outer_solid()

        if pointgrid is None:
            # 21^3 grid from -2..2
            coords = [i / 5 for i in range(-10, 11)]
            pointgrid = [[x, y, z] for x in coords for y in coords for z in coords]

        # Evaluate reference SDFs (assumed to return arrays aligned with pointgrid)
        inner = SDFfromCadQuery(inner_box, pointgrid)
        outer = SDFfromCadQuery(outer_box, pointgrid)
        sdf_inner = inner.sdf_at_pointgrid()  # shape [N]
        sdf_outer = outer.sdf_at_pointgrid()  # shape [N]

        # Device & tensors
        device = next(model.parameters()).device
        P = torch.tensor(pointgrid, dtype=torch.float32, device=device)  # [N,3]
        sdf_inner_t = torch.as_tensor(sdf_inner, dtype=torch.float32, device=device)  # [N]
        sdf_outer_t = torch.as_tensor(sdf_outer, dtype=torch.float32, device=device)  # [N]

        # Batch forward
        with torch.enable_grad():
            sdf_pred, _ = model(P)                       # expect shape [N,1] or [N]
        sdf_pred = sdf_pred.squeeze(-1)              # [N]

        # Box constraints: clamp violations with ReLU
        upper_violation = torch.relu(sdf_pred - sdf_inner_t)     # penalize > inner
        lower_violation = torch.relu(sdf_outer_t - sdf_pred)     # penalize < outer

        loss = (upper_violation + lower_violation).mean()

        return self.weights[2] * loss


    def watertightness_loss(self, model: SDFNetwork, infinity_box:list[list[float]]=None) -> torch.Tensor:  # noqa: RUF013
        return torch.Tensor(0.0) #Avoid crash
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
            self.bounding_boxes_loss(model) +
            #self.watertightness_loss(model) +
            self.frontal_area_loss(model) +
            self.solar_area_loss(model)
        )
