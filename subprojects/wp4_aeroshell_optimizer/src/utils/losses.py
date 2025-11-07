import cadquery as cq
import torch
import wandb
from wp1_cfd_ai.crude_estimator.crude_estimator_diff import (
    calculateCdProxyDifferentiable,
)

from ....wp2_assembler.version1.components.nbz import innerNBZ, outerNBZ
from ..models.sdf import SDFNetwork
from ..utils.CadQuery_to_SDF import BackupAssembly, SDFfromCadQuery
from ..utils.tracking_loss_for_vis import LossTracker
from .kaolin_utils import get_diff_mesh

# Losses being defined here:
# - Eikonal Loss [ES]
# - Smoothness Loss [ES]
# - Bounding Boxes Loss [CH]
# - Watertightness Loss [CH] (currently disabled)
# - Frontal Area Loss [AA] (TODO: archive, as currently replaced by drag loss)
# - Drag Loss [ES] (to be implemented)
# - Solar Area Loss [AA]
# - Total Loss (weighted sum of all losses)


class SDFLosses:
    def __init__(
        self, weights: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dev: str = "cuda"
    ):
        assert len(weights) == 6, (
            "Weights must match the number of losses: [eikonal, smoothness, bounding_box, watertightness, fronatl_area, solar_area]"
        )
        self.weights = weights
        self.device = dev
        self.loss_tracker = LossTracker(
            ["eikonal", "smoothness", "bounding_box", "frontal_area", "solar_area"]
        )

    def eikonal_loss(self, model: SDFNetwork, points: torch.Tensor) -> torch.Tensor:
        """
        This Loss ensures that the gradient of our SDF
        has a unit norm (magnit ude of 1) everywhere in space.
        Args:
            self: SDFLosses Class
            model: The SDF Network
            points: a Batch of 3D points to evaluate the loss at
        Returns:
            loss: The computed Eikonal loss for the given points.
        """
        # sdf: the sdf function values for each point
        # grad_sdf: the gradients of the sdf function at each point
        # both are torch.Tensors
        sdf, grad_sdf = model(points)
        # loss: Mean absolute difference between the norm of the gradients and 1
        # loss = 1/n * sum from 1 to n of  ( ||grad_sdf||_2 - 1 )^2
        # source: https://arxiv.org/pdf/2002.10099, Implicit Geometric Regularization for Learning Shapes, Page 2, 2. Method l(theta)
        loss = (
            self.weights[0] * torch.abs(torch.norm(grad_sdf, p=2, dim=-1) - 1.0).mean()
        )
        return loss

    def smoothness_loss(self, model: SDFNetwork, points: torch.Tensor) -> torch.Tensor:
        """
        This Loss encourages the SDF to be smooth by penalizing large values of the Laplacian.
        The Laplacian of the SDF measures local curvature; minimizing its magnitude reduces high-frequency variations.
        Args:
            self: SDFLosses Class
            model: The SDF Network
            points: a Batch of 3D points to evaluate the loss at
        Returns:
            loss: The computed smoothness loss for the given points.
        """
        sdf, grad_sdf, laplacian_sdf = model(points, return_laplacian=True)
        return self.weights[1] * (laplacian_sdf**2).mean()

    def bounding_boxes_loss_1(
        self,
        model: SDFNetwork,
        inner_box: cq.Solid = innerNBZ(
            random_magnitude_chassis=0.0,
            random_magnitude_seat=0.0,
            random_magnitude_wheels=0.0,
        ),
        outer_box: cq.Solid = outerNBZ(),
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

        # Evaluate reference SDFs (assumed to return arrays aligned with pointgrid)
        inner = SDFfromCadQuery(inner_box, in_mms=False)
        outer = SDFfromCadQuery(outer_box, in_mms=False)
        sdf_inner = inner.sdf_at_pointgrid()  # shape [N]
        sdf_outer = outer.sdf_at_pointgrid()  # shape [N]

        # Device & tensors
        device = next(model.parameters()).device
        P = torch.tensor(pointgrid, dtype=torch.float32, device=self.device)  # [N,3]
        sdf_inner_t = torch.as_tensor(
            sdf_inner, dtype=torch.float32, device=self.device
        )  # [N]
        sdf_outer_t = torch.as_tensor(
            sdf_outer, dtype=torch.float32, device=self.device
        )  # [N]

        # Batch forward
        with torch.enable_grad():
            sdf_pred, _ = model(P)  # expect shape [N,1] or [N]
        sdf_pred = sdf_pred.squeeze(-1)  # [N]

        # Box constraints: clamp violations with ReLU
        upper_violation = torch.relu(sdf_pred - sdf_inner_t)
        lower_violation = torch.relu(sdf_outer_t - sdf_pred)

        loss = (upper_violation + lower_violation).mean()

        return self.weights[2] * loss

    def bounding_box_loss_2(self, model: SDFNetwork) -> torch.Tensor:
        """
        Calculates a Bounding Box Loss based on an Idea of Alex
        """

        # Create SDFs for inner and outer boxes
        inner_box: cq.Solid = innerNBZ(
            random_magnitude_chassis=0.0,
            random_magnitude_seat=0.0,
            random_magnitude_wheels=0.0,
        )
        inner_SDF: SDFfromCadQuery = SDFfromCadQuery(inner_box, in_mms=False)
        outer_box: cq.Solid = outerNBZ()
        outer_SDF: SDFfromCadQuery = SDFfromCadQuery(outer_box, in_mms=False)

        # get pointgrid and sdf values of both boxes
        pointgrid: list[list[float]] = inner_SDF.pointgrid()
        inner_sdf_values: list[float] = inner_SDF.sdf_at_pointgrid()
        outer_sdf_values: list[float] = outer_SDF.sdf_at_pointgrid()

        # get all points where sdf is negative (inside the inner box)
        inner_bb_points = [
            point
            for point, sdf_value in zip(pointgrid, inner_sdf_values, strict=False)
            if sdf_value < 0
        ]

        # get all points where sdf is positive (outside the outer box)
        outer_bb_points = [
            point
            for point, sdf_value in zip(pointgrid, outer_sdf_values, strict=False)
            if sdf_value > 0
        ]

        # get the SDF model values at these points
        inner_bb_points_model = model(
            torch.tensor(inner_bb_points, dtype=torch.float32, device=self.device)
        )
        outer_bb_points_model = model(
            torch.tensor(outer_bb_points, dtype=torch.float32, device=self.device)
        )

        # compute the loss
        inner_loss = torch.relu(inner_bb_points_model[0]).mean()
        outer_loss = torch.relu(-outer_bb_points_model[0]).mean()
        total_loss = inner_loss + outer_loss
        return self.weights[3] * total_loss

    def frontal_area_loss(
        self,
        model: SDFNetwork,
        resolution: int = 128,
        height: float = 4.0,
        width: float = 6.0,
    ) -> torch.Tensor:
        """
        Compute frontal area loss (minimize drag by reducing frontal area).
        Args:
            model: The SDF network
            resolution: Resolution for area computation
        Returns:
            Frontal area loss (to be minimized)
        """
        # Create a grid in the Y-Z plane (frontal projection)
        y_coords = torch.linspace(-width / 2, width / 2, resolution, device=self.device)
        z_coords = torch.linspace(-1.0, height, resolution, device=self.device)
        # z coords from -1 to height to still evaluate at least some area under the ground for now

        y_grid, z_grid = torch.meshgrid(y_coords, z_coords, indexing="ij")
        # Fix x at the front of the vehicle (x = 0 for frontal area)
        x_coords = torch.zeros_like(y_grid, device=self.device)
        # Create points for evaluation
        # As of yet: takes primitive area at x=0 (middle of car in theory)
        points = torch.stack(
            [x_coords.flatten(), y_grid.flatten(), z_grid.flatten()], dim=1
        )
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
        grid_spacing = (width / resolution) * ((1 + height) / resolution)  # dy * dz
        frontal_area = torch.sum(heaviside_approx) * grid_spacing
        return self.weights[4] * frontal_area

    def drag_loss(
        self,
        model: SDFNetwork,
    ) -> torch.Tensor:
        """
        Placeholder for drag loss computation.
        Args:
            model: The SDF network
            resolution: Resolution for area computation
        """
        vertices, faces = get_diff_mesh(model)
        ########### DEBUG PRINTS ############
        print("--- Mesh Debug ---")
        print("Vertices shape:", vertices.shape)
        print("Faces shape:", faces.shape)
        if vertices.numel() == 0:
            print("WARNING: Empty vertices!")
        if faces.numel() == 0:
            print("WARNING: Empty faces!")
        print(
            "Vertices sample (first 5):",
            vertices[:5] if vertices.numel() > 0 else "Empty",
        )
        print("Faces sample (first 5):", faces[:5] if faces.numel() > 0 else "Empty")
        ######################################
        drag = calculateCdProxyDifferentiable(vertices)

        return drag

    def solar_area_loss(
        self,
        model: SDFNetwork,
        resolution: int = 128,
        length: float = 8.0,
        width: float = 6.0,
    ) -> torch.Tensor:
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
        x_coords = torch.linspace(
            -length / 2, length / 2, resolution, device=self.device
        )
        y_coords = torch.linspace(-width / 2, width / 2, resolution, device=self.device)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")
        # Fix z at the top of the vehicle (z = 1.2 for top area)
        # again arbitrary height, just to get an area on top, need to be fixed later
        z_coords = torch.full_like(x_grid, 1.2, device=self.device)
        # Create points for evaluation
        points = torch.stack(
            [x_grid.flatten(), y_grid.flatten(), z_coords.flatten()], dim=1
        )
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
        return -self.weights[5] * top_area  # Negative for maximization

    def get_total_loss(
        self, model: SDFNetwork, points: torch.Tensor, epoch_name: str = "-1"
    ) -> torch.Tensor:
        l1 = self.eikonal_loss(model, points)
        l2 = self.smoothness_loss(model, points)
        l3 = self.bounding_boxes_loss_1(model)
        l4 = self.bounding_box_loss_2(model)
        l5 = self.frontal_area_loss(model)
        l6 = self.drag_loss(model)
        l7 = self.solar_area_loss(model)

        # The following two lines track and store the individual losses per epoch
        losses = {
            "eikonal": l1.item(),
            "smoothness": l2.item(),
            "bounding_box_1": l3.item(),
            "bounding_box_2": l4.item(),
            "frontal_area": l5.item(),
            "drag_loss": l6.item(),
            "solar_area": l7.item(),
        }
        self.loss_tracker.add_epoch(losses)

        # logging with wandb, added by eschaerer
        wandb.log(
            {
                "losses/eikonal_loss": l1.item(),
                "losses/smoothness_loss": l2.item(),
                "losses/bounding_box_loss": l3.item(),
                "losses/frontal_area_loss": l4.item(),
                "losses/drag_loss": l5.item(),
                "losses/solar_area_loss": l6.item(),
            }
        )

        return (
            l1 / 1.4 + l2 / 2.5 + l3 / 4 + 0 * l4 / 30 + l5 / 1000 + l6 / 40 + l7
        )  # normed by hand for debugging purposes


#### 1. Eikonal loss
#### 2. Smoothness loss
#### 3. Bounding box loss
#### 4. Frontal area loss
#### 5. Drag loss
#### 6. Solar area loss
