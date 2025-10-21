import torch
from sdf import SDFNetwork
import typing as T

# The following is to handle cad-Query-files
import cadquery as cq
from OCP.gp import gp_Pnt
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.TopAbs import TopAbs_IN, TopAbs_ON
# Define various loss functions for training the SDFNetwork
# These losses include eikonal loss, smoothness loss, bounding box constraints,
# watertightness loss, and performance-related losses.


class ShapeWrapper:
    def __init__(self, size=(5, 5, 5), shape: cq.Shape | None = None):
        # If a shape is provided, use it; otherwise create a box.
        if shape is not None:
            self.shape = shape
        else:
            self.shape = cq.Workplane("XY").box(*size, centered=True)

    def _make_point_vertex(self, point) -> cq.Vertex:
        x, y, z = point
        return cq.Vertex.makeVertex(x, y, z)

    def is_point_inside(self, point, tol: float = 1e-9) -> bool:
        """
        True if `point` is inside or on the boundary of ANY solid contained in the shape.
        For non-solids (faces, wires, etc.) returns False (3D 'inside' is undefined).
        """
        shp = self.shape._wrapped
        pnt = gp_Pnt(*point)

        solids = list(shp.Solids()) if hasattr(shp, "Solids") else []
        if solids:  # Compound or multi-solid
            for s in solids:
                cls = BRepClass3d_SolidClassifier(s.wrapped, pnt, tol)
                state = cls.State()
                if state in (TopAbs_IN, TopAbs_ON):
                    return True
            return False

        if shp.isSolid():  # Single-solid
            cls = BRepClass3d_SolidClassifier(shp.wrapped, pnt, tol)
            state = cls.State()
            return state in (TopAbs_IN, TopAbs_ON)

        return False  # Not a solid

    def distance_to_shape(self, point) -> float:
        """
        Shortest Euclidean distance from `point` to the shape.
        Returns 0.0 if the point is on/inside a solid.
        """
        shp = self.shape._wrapped
        vtx = self._make_point_vertex(point)
        dss = BRepExtrema_DistShapeShape(vtx.wrapped, shp.wrapped)
        dss.Perform()
        return float(dss.Value())
    






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

    def bounding_boxes_loss(self, model, inner_box=None, outer_box=None) -> torch.Tensor:
        #Compute Bounding Boxes loss here
        inner_box = ShapeWrapper(size=(2,2,2)) if inner_box is None else ShapeWrapper(shape=inner_box)
        outer_box = ShapeWrapper(shape=outer_box) 
        total_loss = 0

        k = 50
        p_outside, p_inside = [], []
        for x in range(-k, k):
            for y in range(-k, k):
                for z in range(-k, k):
                    p = torch.tensor([x/10,y/10,z/10, 0], dtype=torch.float32)
                    if not outer_box.is_point_inside(p[0:3].numpy()):
                        p[3] = outer_box.distance_to_shape(p)
                        p_outside.append(p)
                    elif inner_box.is_point_inside(p[0:3].numpy()):
                        p[3] = inner_box.distance_to_shape(p)
                        p_inside.append(p)

        for p in p_outside:
            sdf_value = model.forward(p[0:3].unsqueeze(0))
            loss = torch.relu(-sdf_value + p[3])  # SDF should be greater than distance to outer box
            total_loss += loss
        
        for p in p_inside:
            sdf_value = model.forward(p[0:3].unsqueeze(0))
            loss = torch.relu(sdf_value + p[3])  # SDF should be less than negative distance to inner box
            total_loss += loss

        #Ensure compatibility
        bounding_boxes_loss = torch.tensor(total_loss)  
        return self.weights[2]* bounding_boxes_loss

    def watertightness_loss(self, model: SDFNetwork, points: torch.Tensor) -> torch.Tensor:
        sdf, _ = model(points)
        return self.weights[3] * (sdf**2).mean()

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
        sdf_values = model(points).reshape(resolution, resolution)
        
        # Approximate area using the Heaviside function
        # Use a smooth approximation of the Heaviside function
        epsilon = 0.02
        heaviside_approx = torch.sigmoid(-sdf_values / epsilon)
        
        # Integrate over the grid to get area
        grid_spacing = (width / resolution) * ((1+height) / resolution)  # dy * dz
        frontal_area = torch.sum(heaviside_approx) * grid_spacing
        
        return self.weights[4] * frontal_area
        
    def solar_area_loss(self, model: SDFNetwork) -> torch.Tensor:
        # Placeholder for solar area loss computation
        # This should be implemented based on specific requirements
        return self.weights[4]* torch.tensor(0.0)
    

    def get_total_loss(self, model: SDFNetwork, points: torch.Tensor) -> torch.Tensor:
        return (
            self.eikonal_loss(model, points) +
            self.smoothness_loss(model, points) +
            self.bounding_boxes_loss(model, points) +
            self.watertightness_loss(model, points) +
            self.frontal_area_loss(model) +
            self.solar_area_loss(model)
        )