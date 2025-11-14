import warnings

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

warnings.filterwarnings("ignore")


class SDF3DVisualizer:
    """
    Comprehensive visualization tools for 3D SDF optimization results.

    This class provides various visualization methods including:
    - 3D surface rendering
    - Cross-sectional views
    - Loss curve plotting
    - Area projections
    """

    def __init__(self, device="cpu"):
        self.device = device

    def extract_mesh_marching_cubes(self, sdf_net, bounds=(-2, 2), resolution=64):
        """
        Extract 3D mesh using marching cubes algorithm.

        Args:
            sdf_net: Trained SDF network
            bounds: (min, max) coordinates for sampling
            resolution: Grid resolution for marching cubes

        Returns:
            vertices, faces, sdf_values grid
        """
        # Create 3D grid
        coords = np.linspace(bounds[0], bounds[1], resolution)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

        # Flatten coordinates for network evaluation
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        points_tensor = torch.tensor(points, dtype=torch.float32, device=self.device)

        # Evaluate SDF on the grid
        with torch.no_grad():
            sdf_values = sdf_net(points_tensor).cpu().numpy()

        # Reshape back to 3D grid
        sdf_grid = sdf_values.reshape(resolution, resolution, resolution)

        # Extract mesh using marching cubes at level=0
        try:
            vertices, faces, _, _ = measure.marching_cubes(sdf_grid, level=0.0)

            # Scale vertices back to world coordinates
            scale = (bounds[1] - bounds[0]) / resolution
            vertices = vertices * scale + bounds[0]

            return vertices, faces, sdf_grid

        except Exception as e:
            print(f"Marching cubes failed: {e}")
            return None, None, sdf_grid

    def plot_3d_surface_matplotlib(
        self,
        sdf_net,
        bounds=(-2, 2),
        resolution=64,
        save_path=None,
        show_wireframe=False,
    ):
        """
        Create 3D surface plot using matplotlib.

        Args:
            sdf_net: Trained SDF network
            bounds: Coordinate bounds for visualization
            resolution: Grid resolution
            save_path: Path to save the plot
            show_wireframe: Whether to show wireframe overlay
        """
        vertices, faces, sdf_grid = self.extract_mesh_marching_cubes(
            sdf_net, bounds, resolution
        )

        if vertices is None:
            print("Could not extract mesh for visualization")
            return

        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Create mesh collection
        mesh = Poly3DCollection(
            vertices[faces], alpha=0.8, edgecolor="k" if show_wireframe else None
        )
        mesh.set_facecolor("lightblue")
        ax.add_collection3d(mesh)

        # Set axis properties
        ax.set_xlabel("X (Forward)")
        ax.set_ylabel("Y (Lateral)")
        ax.set_zlabel("Z (Vertical)")
        ax.set_title("3D SDF Optimized Vehicle Shape")

        # Set equal aspect ratio and bounds
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_zlim(bounds)

        # Add bounding box visualization
        self._add_bounding_boxes(
            ax,
            outer_bounds=((-1.8, -1.8, -0.8), (1.8, 1.8, 0.8)),
            inner_bounds=((-0.5, -0.5, -0.2), (0.5, 0.5, 0.2)),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"3D surface plot saved to {save_path}")

        plt.show()

    def plot_3d_surface_plotly(
        self, sdf_net, bounds=(-2, 2), resolution=64, save_path=None, show_plot=True
    ):
        """
        Create interactive 3D surface plot using plotly.

        Args:
            sdf_net: Trained SDF network
            bounds: Coordinate bounds for visualization
            resolution: Grid resolution
            save_path: Path to save the HTML plot
        """
        vertices, faces, sdf_grid = self.extract_mesh_marching_cubes(
            sdf_net, bounds, resolution
        )

        if vertices is None:
            print("Could not extract mesh for visualization")
            return

        # Create plotly 3D mesh
        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    opacity=0.8,
                    color="lightblue",
                    name="Vehicle Shape",
                )
            ]
        )

        # Add bounding box wireframes
        self._add_plotly_bounding_boxes(
            fig,
            outer_bounds=((-1.8, -1.8, -0.8), (1.8, 1.8, 0.8)),
            inner_bounds=((-0.5, -0.5, -0.2), (0.5, 0.5, 0.2)),
        )

        # Set layout
        fig.update_layout(
            title="Interactive 3D SDF Optimized Vehicle Shape",
            scene=dict(
                xaxis_title="X (Forward)",
                yaxis_title="Y (Lateral)",
                zaxis_title="Z (Vertical)",
                aspectmode="cube",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            width=1000,
            height=800,
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Interactive 3D plot saved to {save_path}")
        if show_plot:
            fig.show()

    def plot_cross_sections(
        self, sdf_net, bounds=(-2, 2), resolution=128, save_path=None, show_plot=False
    ):
        """
        Plot cross-sectional views of the SDF.

        Args:
            sdf_net: Trained SDF network
            bounds: Coordinate bounds
            resolution: Grid resolution for cross-sections
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        coords = np.linspace(bounds[0], bounds[1], resolution)

        # XY cross-section (top view) at z=0
        X, Y = np.meshgrid(coords, coords, indexing="ij")
        Z = np.zeros_like(X)
        points = torch.tensor(
            np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            sdf_values = sdf_net(points).cpu().numpy().reshape(resolution, resolution)

        im1 = axes[0, 0].contour(
            X, Y, sdf_values, levels=[0], colors="red", linewidths=2
        )
        axes[0, 0].contourf(X, Y, sdf_values, levels=20, cmap="RdBu", alpha=0.7)
        axes[0, 0].set_title("Top View (XY at z=0)")
        axes[0, 0].set_xlabel("X (Forward)")
        axes[0, 0].set_ylabel("Y (Lateral)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_aspect("equal")

        # XZ cross-section (side view) at y=0
        X, Z = np.meshgrid(coords, coords, indexing="ij")
        Y = np.zeros_like(X)
        points = torch.tensor(
            np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            sdf_values = sdf_net(points).cpu().numpy().reshape(resolution, resolution)

        im2 = axes[0, 1].contour(
            X, Z, sdf_values, levels=[0], colors="red", linewidths=2
        )
        axes[0, 1].contourf(X, Z, sdf_values, levels=20, cmap="RdBu", alpha=0.7)
        axes[0, 1].set_title("Side View (XZ at y=0)")
        axes[0, 1].set_xlabel("X (Forward)")
        axes[0, 1].set_ylabel("Z (Vertical)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_aspect("equal")

        # YZ cross-section (front view) at x=0
        Y, Z = np.meshgrid(coords, coords, indexing="ij")
        X = np.zeros_like(Y)
        points = torch.tensor(
            np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            sdf_values = sdf_net(points).cpu().numpy().reshape(resolution, resolution)

        im3 = axes[0, 2].contour(
            Y, Z, sdf_values, levels=[0], colors="red", linewidths=2
        )
        axes[0, 2].contourf(Y, Z, sdf_values, levels=20, cmap="RdBu", alpha=0.7)
        axes[0, 2].set_title("Front View (YZ at x=0)")
        axes[0, 2].set_xlabel("Y (Lateral)")
        axes[0, 2].set_ylabel("Z (Vertical)")
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_aspect("equal")

        # Frontal area visualization (YZ plane)
        y_coords = np.linspace(-2.0, 2.0, resolution)
        z_coords = np.linspace(-1.0, 1.0, resolution)
        Y_front, Z_front = np.meshgrid(y_coords, z_coords, indexing="ij")
        X_front = np.zeros_like(Y_front)
        points_front = torch.tensor(
            np.stack([X_front.flatten(), Y_front.flatten(), Z_front.flatten()], axis=1),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            sdf_front = (
                sdf_net(points_front).cpu().numpy().reshape(resolution, resolution)
            )

        # Highlight the frontal area (where SDF <= 0)
        frontal_mask = sdf_front <= 0
        axes[1, 0].imshow(
            frontal_mask,
            extent=[-2, 2, -1, 1],
            origin="lower",
            cmap="Reds",
            alpha=0.8,
            aspect="equal",
        )
        axes[1, 0].contour(
            Y_front, Z_front, sdf_front, levels=[0], colors="black", linewidths=2
        )
        axes[1, 0].set_title("Frontal Area (Drag-relevant)")
        axes[1, 0].set_xlabel("Y (Lateral)")
        axes[1, 0].set_ylabel("Z (Vertical)")
        axes[1, 0].grid(True, alpha=0.3)

        # Top area visualization (XY plane at high z)
        x_coords = np.linspace(-2.0, 2.0, resolution)
        y_coords = np.linspace(-2.0, 2.0, resolution)
        X_top, Y_top = np.meshgrid(x_coords, y_coords, indexing="ij")
        Z_top = np.full_like(X_top, 0.5)  # At a height where solar panels would be
        points_top = torch.tensor(
            np.stack([X_top.flatten(), Y_top.flatten(), Z_top.flatten()], axis=1),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            sdf_top = sdf_net(points_top).cpu().numpy().reshape(resolution, resolution)

        # Highlight the top area (where SDF <= 0)
        top_mask = sdf_top <= 0
        axes[1, 1].imshow(
            top_mask,
            extent=[-2, 2, -2, 2],
            origin="lower",
            cmap="Blues",
            alpha=0.8,
            aspect="equal",
        )
        axes[1, 1].contour(
            X_top, Y_top, sdf_top, levels=[0], colors="black", linewidths=2
        )
        axes[1, 1].set_title("Top Area (Solar Collection)")
        axes[1, 1].set_xlabel("X (Forward)")
        axes[1, 1].set_ylabel("Y (Lateral)")
        axes[1, 1].grid(True, alpha=0.3)

        # 3D visualization of level sets
        axes[1, 2].text(
            0.5,
            0.5,
            "See separate\n3D visualization\nfor complete view",
            transform=axes[1, 2].transAxes,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightgray"),
        )
        axes[1, 2].set_title("3D View (Separate Plot)")
        axes[1, 2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Cross-section plots saved to {save_path}")

        if show_plot:
            plt.show()

    def plot_loss_curves(self, loss_history, save_path=None, show_plot=False):
        """
        Plot training loss curves for all loss components.

        Args:
            loss_history: Dictionary containing loss values over iterations
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        iterations = range(len(loss_history["total"]))

        # Total loss
        axes[0, 0].plot(
            iterations, loss_history["total"], "k-", linewidth=2, label="Total Loss"
        )
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Loss Value")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Eikonal loss
        if "eikonal" in loss_history:
            axes[0, 1].plot(
                iterations, loss_history["eikonal"], "b-", linewidth=2, label="Eikonal"
            )
            axes[0, 1].set_title("Eikonal Loss (SDF Validity)")
            axes[0, 1].set_xlabel("Iteration")
            axes[0, 1].set_ylabel("Loss Value")
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

        # Size constraint loss
        if "size_constraint" in loss_history:
            axes[0, 2].plot(
                iterations,
                loss_history["size_constraint"],
                "r-",
                linewidth=2,
                label="Size Constraint",
            )
            axes[0, 2].set_title("Size Constraint Loss")
            axes[0, 2].set_xlabel("Iteration")
            axes[0, 2].set_ylabel("Loss Value")
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].legend()

        # Frontal area loss
        if "frontal_area" in loss_history:
            axes[1, 0].plot(
                iterations,
                loss_history["frontal_area"],
                "g-",
                linewidth=2,
                label="Frontal Area",
            )
            axes[1, 0].set_title("Frontal Area Loss (Drag)")
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("Area")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

        # Top area loss (note: this is negative for maximization)
        if "top_area" in loss_history:
            # Plot the actual area (negative of the loss)
            actual_top_area = [-x for x in loss_history["top_area"]]
            axes[1, 1].plot(
                iterations, actual_top_area, "m-", linewidth=2, label="Top Area"
            )
            axes[1, 1].set_title("Top Area (Solar Collection)")
            axes[1, 1].set_xlabel("Iteration")
            axes[1, 1].set_ylabel("Area")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()

        # Smoothness loss
        if "smoothness" in loss_history:
            axes[1, 2].plot(
                iterations,
                loss_history["smoothness"],
                "orange",
                linewidth=2,
                label="Smoothness",
            )
            axes[1, 2].set_title("Smoothness Loss")
            axes[1, 2].set_xlabel("Iteration")
            axes[1, 2].set_ylabel("Loss Value")
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Loss curves saved to {save_path}")
        if show_plot:
            plt.show()

    def _add_bounding_boxes(self, ax, outer_bounds, inner_bounds):
        """Add bounding box wireframes to matplotlib 3D plot"""
        # Constants for corner connections
        ALL_ZEROS = 0
        ALL_ONES = 3

        # Unpack bounds
        outer_min, outer_max = outer_bounds
        inner_min, inner_max = inner_bounds

        # Draw outer box edges
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if i + j + k == ALL_ZEROS or i + j + k == ALL_ONES:
                        continue
                    p1 = [
                        outer_min[0] if i == 0 else outer_max[0],
                        outer_min[1] if j == 0 else outer_max[1],
                        outer_min[2] if k == 0 else outer_max[2],
                    ]
                    ax.plot3D([p1[0]], [p1[1]], [p1[2]], "r.", alpha=0.5)

        # Draw inner box edges
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if i + j + k == ALL_ZEROS or i + j + k == ALL_ONES:
                        continue
                    p1 = [
                        inner_min[0] if i == 0 else inner_max[0],
                        inner_min[1] if j == 0 else inner_max[1],
                        inner_min[2] if k == 0 else inner_max[2],
                    ]
                    ax.plot3D([p1[0]], [p1[1]], [p1[2]], "b.", alpha=0.5)

        # Add labels
        ax.text(
            outer_max[0],
            outer_max[1],
            outer_max[2],
            "Outer Bound",
            color="red",
            fontsize=8,
        )
        ax.text(
            inner_max[0],
            inner_max[1],
            inner_max[2],
            "Inner Bound",
            color="blue",
            fontsize=8,
        )

    def _add_plotly_bounding_boxes(self, fig, outer_bounds, inner_bounds):
        """Add bounding box wireframes to plotly 3D plot"""
        # This would add wireframe boxes to the plotly figure
        # Implementation simplified for brevity

    def compute_final_metrics(self, sdf_net):
        """
        Compute final performance metrics of the optimized shape.

        Args:
            sdf_net: Trained SDF network

        Returns:
            Dictionary of performance metrics
        """
        from losses import SDF3DLossCalculator

        loss_calc = SDF3DLossCalculator(device=self.device)

        # Compute areas
        frontal_area = loss_calc.compute_frontal_area_loss(sdf_net)
        top_area = -loss_calc.compute_top_area_loss(
            sdf_net
        )  # Convert back from negative

        # Compute volume (approximate)
        bounds = (-2, 2)
        resolution = 64
        coords = np.linspace(bounds[0], bounds[1], resolution)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        points_tensor = torch.tensor(points, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            sdf_values = sdf_net(points_tensor).cpu().numpy()

        # Volume approximation
        voxel_volume = ((bounds[1] - bounds[0]) / resolution) ** 3
        interior_voxels = np.sum(sdf_values <= 0)
        volume = interior_voxels * voxel_volume

        metrics = {
            "frontal_area": float(frontal_area.cpu()),
            "top_area": float(top_area.cpu()),
            "volume": volume,
            "drag_efficiency": float(top_area.cpu())
            / float(frontal_area.cpu()),  # Solar area / drag area ratio
        }

        print("\n=== Final Shape Metrics ===")
        print(f"Frontal Area (drag-related): {metrics['frontal_area']:.4f}")
        print(f"Top Area (solar collection): {metrics['top_area']:.4f}")
        print(f"Volume: {metrics['volume']:.4f}")
        print(f"Drag Efficiency Ratio: {metrics['drag_efficiency']:.4f}")

        return metrics
