# Create the loss functions module

import torch
import torch.nn.functional as F
import numpy as np

class SDF3DLossCalculator:
    """
    Comprehensive loss function calculator for 3D SDF optimization.
    
    This class handles multiple objectives:
    1. Eikonal constraint for valid SDF properties
    2. Size constraints (bounding box limits)
    3. Frontal area minimization (drag reduction)
    4. Top area maximization (solar collection)
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Loss weights - these can be tuned for different priorities
        self.weights = {
            'eikonal': 1.0,          # Ensures valid SDF field
            'size_constraint': 10.0,  # Enforces bounding box limits
            'frontal_area': 5.0,     # Minimizes drag
            'top_area': -3.0,        # Maximizes solar area (negative for maximization)
            'smoothness': 0.5        # Encourages smooth surfaces
        }
    
    def compute_eikonal_loss(self, sdf_net, points):
        """
        Compute Eikonal loss: ||∇SDF|| = 1
        
        Args:
            sdf_net: The SDF network
            points: Sample points in 3D space (N, 3)
        
        Returns:
            Eikonal loss value
        """
        points.requires_grad_(True)
        sdf_values = sdf_net(points)
        
        # Compute gradients of SDF with respect to input coordinates
        grad_outputs = torch.ones_like(sdf_values)
        gradients = torch.autograd.grad(
            outputs=sdf_values,
            inputs=points,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Eikonal loss: penalize deviation from unit gradient norm
        gradient_norm = torch.norm(gradients, dim=1)
        eikonal_loss = torch.mean((gradient_norm - 1.0) ** 2)
        
        return eikonal_loss
    
    def compute_size_constraint_loss(self, sdf_net, outer_bounds, inner_bounds):
        """
        Compute size constraint loss to keep shape within bounding boxes.
        
        Args:
            sdf_net: The SDF network
            outer_bounds: (min_coords, max_coords) for outer bounding box
            inner_bounds: (min_coords, max_coords) for inner bounding box
        
        Returns:
            Size constraint loss
        """
        # Sample points on the bounding box faces
        outer_min, outer_max = outer_bounds
        inner_min, inner_max = inner_bounds
        
        # Outer boundary constraint points
        outer_points = []
        for i in range(3):  # x, y, z dimensions
            for val in [outer_min[i], outer_max[i]]:
                # Create grid on each face of the outer bounding box
                other_dims = [j for j in range(3) if j != i]
                coords = torch.zeros(100, 3, device=self.device)
                coords[:, i] = val
                
                # Random sampling on the face
                for j, dim in enumerate(other_dims):
                    coords[:, dim] = torch.rand(100, device=self.device) * \
                                   (outer_max[dim] - outer_min[dim]) + outer_min[dim]
                outer_points.append(coords)
        
        outer_points = torch.cat(outer_points, dim=0)
        outer_sdf = sdf_net(outer_points)
        
        # Loss: penalize if shape extends beyond outer boundary (SDF should be positive)
        outer_loss = torch.mean(F.relu(-outer_sdf))  # Penalty for negative SDF outside
        
        # Inner boundary constraint points
        inner_points = []
        for i in range(3):  # x, y, z dimensions
            for val in [inner_min[i], inner_max[i]]:
                other_dims = [j for j in range(3) if j != i]
                coords = torch.zeros(100, 3, device=self.device)
                coords[:, i] = val
                
                for j, dim in enumerate(other_dims):
                    coords[:, dim] = torch.rand(100, device=self.device) * \
                                   (inner_max[dim] - inner_min[dim]) + inner_min[dim]
                inner_points.append(coords)
        
        inner_points = torch.cat(inner_points, dim=0)
        inner_sdf = sdf_net(inner_points)
        
        # Loss: penalize if shape doesn't fill inner boundary (SDF should be negative)
        inner_loss = torch.mean(F.relu(inner_sdf))  # Penalty for positive SDF inside
        
        return outer_loss + inner_loss
    
    def compute_frontal_area_loss(self, sdf_net, resolution=64):
        """
        Compute frontal area loss (minimize drag by reducing frontal area).
        
        Args:
            sdf_net: The SDF network
            resolution: Resolution for area computation
        
        Returns:
            Frontal area loss (to be minimized)
        """
        # Create a grid in the Y-Z plane (frontal projection)
        y_coords = torch.linspace(-2.0, 2.0, resolution, device=self.device)
        z_coords = torch.linspace(-1.0, 1.0, resolution, device=self.device)
        y_grid, z_grid = torch.meshgrid(y_coords, z_coords, indexing='ij')
        
        # Fix x at the front of the vehicle (x = 0 for frontal area)
        x_coords = torch.zeros_like(y_grid)
        
        # Create points for evaluation
        points = torch.stack([x_coords.flatten(), y_grid.flatten(), z_grid.flatten()], dim=1)
        sdf_values = sdf_net(points).reshape(resolution, resolution)
        
        # Approximate area using the Heaviside function
        # Use a smooth approximation of the Heaviside function
        epsilon = 0.02
        heaviside_approx = torch.sigmoid(-sdf_values / epsilon)
        
        # Integrate over the grid to get area
        grid_spacing = (4.0 / resolution) * (2.0 / resolution)  # dy * dz
        frontal_area = torch.sum(heaviside_approx) * grid_spacing
        
        return frontal_area
    
    def compute_top_area_loss(self, sdf_net, resolution=64):
        """
        Compute top area loss (maximize solar collection area on top surface).
        
        Args:
            sdf_net: The SDF network
            resolution: Resolution for area computation
        
        Returns:
            Negative top area (for maximization)
        """
        # Create a grid in the X-Y plane (top projection)
        x_coords = torch.linspace(-2.0, 2.0, resolution, device=self.device)
        y_coords = torch.linspace(-2.0, 2.0, resolution, device=self.device)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Fix z at the top of the vehicle (z = 0.8 for top area)
        z_coords = torch.full_like(x_grid, 0.8)
        
        # Create points for evaluation
        points = torch.stack([x_grid.flatten(), y_grid.flatten(), z_coords.flatten()], dim=1)
        sdf_values = sdf_net(points).reshape(resolution, resolution)
        
        # Approximate area using the Heaviside function
        epsilon = 0.02
        heaviside_approx = torch.sigmoid(-sdf_values / epsilon)
        
        # Integrate over the grid to get area
        grid_spacing = (4.0 / resolution) * (4.0 / resolution)  # dx * dy
        top_area = torch.sum(heaviside_approx) * grid_spacing
        
        return -top_area  # Negative for maximization
    
    def compute_smoothness_loss(self, sdf_net, points):
        """
        Compute smoothness loss to encourage smooth surfaces.
        
        Args:
            sdf_net: The SDF network
            points: Sample points in 3D space
        
        Returns:
            Smoothness loss based on Laplacian
        """
        points.requires_grad_(True)
        sdf_values = sdf_net(points)
        
        # Compute first derivatives
        grad_outputs = torch.ones_like(sdf_values)
        first_derivs = torch.autograd.grad(
            outputs=sdf_values,
            inputs=points,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute second derivatives (Laplacian)
        laplacian = 0
        for i in range(3):  # x, y, z dimensions
            second_deriv = torch.autograd.grad(
                outputs=first_derivs[:, i],
                inputs=points,
                grad_outputs=torch.ones_like(first_derivs[:, i]),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0][:, i]
            laplacian += second_deriv
        
        # Penalize high curvature (large Laplacian)
        smoothness_loss = torch.mean(laplacian ** 2)
        
        return smoothness_loss
    
    def compute_total_loss(self, sdf_net, sample_points, outer_bounds, inner_bounds):
        """
        Compute the total weighted loss combining all objectives.
        
        Args:
            sdf_net: The SDF network
            sample_points: Random sample points for evaluation
            outer_bounds: Outer bounding box limits
            inner_bounds: Inner bounding box limits
        
        Returns:
            Dictionary containing individual losses and total loss
        """
        losses = {}
        
        # Compute individual loss components
        losses['eikonal'] = self.compute_eikonal_loss(sdf_net, sample_points)
        losses['size_constraint'] = self.compute_size_constraint_loss(
            sdf_net, outer_bounds, inner_bounds)
        losses['frontal_area'] = self.compute_frontal_area_loss(sdf_net)
        losses['top_area'] = self.compute_top_area_loss(sdf_net)
        losses['smoothness'] = self.compute_smoothness_loss(sdf_net, sample_points)
        
        # Compute weighted losses separately
        weighted_losses = {}
        total_loss = 0
        
        # Calculate weighted losses for original loss components
        for loss_name in ['eikonal', 'size_constraint', 'frontal_area', 'top_area', 'smoothness']:
            if loss_name in losses:
                weighted_loss = self.weights[loss_name] * losses[loss_name]
                weighted_losses[f'{loss_name}_weighted'] = weighted_loss
                total_loss += weighted_loss
        
        # Combine original and weighted losses
        combined_losses = {**losses, **weighted_losses, 'total': total_loss}
        
        return combined_losses
    
    def update_weights(self, new_weights):
        """Update loss weights for different optimization phases"""
        self.weights.update(new_weights)


print("Created loss_functions.py - Comprehensive loss functions for 3D SDF optimization")