"""
Latent Space Interpolator for SDF Visualization

Generates 2D grid visualizations showing how a neural SDF model's mesh evolves
as you interpolate through the first two latent dimensions (z0, z1).

Uses marching cubes for mesh extraction and PyVista for off-screen rendering.
All parameters are config-driven to avoid hardcoding.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.GINN_visuals.tim_render import (
    render_mesh,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.pyvista_headless import (
    ensure_pyvista_headless,
)
from torch import Tensor

# Import after checking availability to give helpful error message
try:
    pv = ensure_pyvista_headless()
except ImportError:
    raise ImportError("PyVista is required for interpolator. Install with: pip install pyvista")

from subprojects.wp3_solar_estimator.mesh_extractors.non_diff_algos.marching_cubes import (
    MarchingCubesExtractor,
)


class Interpolator:
    """
    Generate 2D interpolation grids through latent space dimensions.

    Creates a grid of mesh visualizations by interpolating through z[0] and z[1],
    showing how the SDF-based geometry evolves. All meshes are extracted using
    marching cubes and rendered with PyVista in off-screen mode.

    Attributes:
        model: Trained neural SDF model with set_latent_vector() method
        config: Dict with keys:
            - grid_resolution: int, points per latent dimension (e.g., 10 for 10x10)
            - latent_range: [min, max], bounds for z[0] and z[1]
            - mesh_resolution: int, marching cubes grid resolution (e.g., 128)
            - bbox_min, bbox_max: lists, bounding box for mesh extraction
            - isovalue: float, SDF level set (default 0.0)
            - camera_view: str, camera angle (iso, front, back, left, right, top, bottom)
            - mesh_color: str, PyVista color (e.g., "lightgrey")
            - window_size: (width, height) tuple for rendered images
            - save_dir: Path for saving intermediate/final images
            - upload_to_wandb: bool, whether to upload grid to WandB
        device: torch device (cuda or cpu)
    """

    def __init__(self, model, config: dict, device: torch.device):
        """
        Initialize interpolator with model and config.

        Args:
            model: Neural SDF model with set_latent_vector() method
            config: Configuration dict (see class docstring for keys)
            device: torch device
        """
        self.model = model
        self.config = config
        self.device = device

        # Initialize marching cubes extractor
        self.mc_extractor = MarchingCubesExtractor()

        # Parse config
        self.grid_resolution = int(config.get("grid_resolution", 10))
        self.latent_range = config.get("latent_range", [-2.0, 2.0])
        self.mesh_resolution = int(config.get("mesh_resolution", 128))
        
        # Extract bbox values as lists
        bbox_min_list = config.get("bbox_min", [-4000, -4000, -4000])
        bbox_max_list = config.get("bbox_max", [4000, 4000, 4000])
        
        # Convert to MarchingCubesExtractor format: [[xmin,xmax], [ymin,ymax], [zmin,zmax]]
        self.bbox = [
            [bbox_min_list[0], bbox_max_list[0]],
            [bbox_min_list[1], bbox_max_list[1]],
            [bbox_min_list[2], bbox_max_list[2]],
        ]
        
        self.isovalue = float(config.get("isovalue", 0.0))
        self.camera_view = config.get("camera_view", "iso")
        self.mesh_color = config.get("mesh_color", "lightgrey")
        self.window_size = tuple(config.get("window_size", [800, 600]))
        self.save_dir = Path(config.get("save_dir", "./interpolation_output"))
        self.upload = config.get("upload_to_wandb", True)

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure the marching cubes grid (call once during init)
        self.mc_extractor.set_grid(bbox=self.bbox, resolution=self.mesh_resolution)

        # Get model's latent dimension (try z_dim first, then check for _active_z attribute)
        try:
            self.z_dim = getattr(model, "z_dim")
        except AttributeError:
            self.z_dim = getattr(model, "_active_z", None).shape[0] if getattr(model, "_active_z", None) is not None else None

        
        if self.z_dim is None:
            raise ValueError("Model must have z_dim attribute (latent vector dimension)")

        # Placeholder for compiled grid image
        self._grid_image = None
        self._has_logged_first_nonempty_render = False

        # Validate rendering backend once up front to avoid late crash mid-grid.
        self._run_render_preflight()

    def _run_render_preflight(self) -> None:
        """Run a tiny off-screen render probe to validate backend before main loop."""
        try:
            probe_plotter = pv.Plotter(off_screen=True, window_size=(128, 128))
            probe_plotter.add_mesh(pv.Sphere(radius=1.0), color="white")
            probe_image = probe_plotter.screenshot(transparent_background=False)
            probe_plotter.close()

            if probe_image is None or getattr(probe_image, "size", 0) == 0:
                raise RuntimeError("Render probe produced no image output")

            print(
                "PyVista render preflight passed "
                f"(DISPLAY={os.environ.get('DISPLAY', '<unset>')})"
            )
        except Exception as exc:
            raise RuntimeError(
                "PyVista headless preflight failed before interpolation. "
                f"DISPLAY={os.environ.get('DISPLAY', '<unset>')} "
                f"PYVISTA_OFF_SCREEN={os.environ.get('PYVISTA_OFF_SCREEN', '<unset>')}"
            ) from exc

    def generate_interpolation_grid(self) -> Image.Image:
        """
        Generate 2D grid of mesh visualizations interpolating through latent space.

        Sweeps through z[0] and z[1] coordinates in a grid pattern, extracting
        and rendering a mesh at each point. Intermediate images saved to disk,
        only final grid image returned.

        Returns:
            PIL Image of final 2D grid layout
        """
        print(
            f"Generating {self.grid_resolution}x{self.grid_resolution} "
            "interpolation grid..."
        )

        # Generate latent coordinate grid
        z0_values = np.linspace(self.latent_range[0], self.latent_range[1], self.grid_resolution)
        z1_values = np.linspace(self.latent_range[0], self.latent_range[1], self.grid_resolution)

        # Store image paths for grid assembly
        grid_images = []

        # Sweep through latent space
        for i, z0 in enumerate(z0_values):
            row_images = []
            for j, z1 in enumerate(z1_values):
                print(f"  Point ({i + 1}/{self.grid_resolution}, {j + 1}/{self.grid_resolution}): " 
                      f"z=[{z0:.3f}, {z1:.3f}]")

                # Extract and render mesh at this latent point
                image = self._extract_and_render_at_latent(z0, z1)
                if image is not None:
                    row_images.append(image)
                else:
                    # Create blank image if extraction fails
                    row_images.append(Image.new("RGB", self.window_size, (200, 200, 200)))

            grid_images.append(row_images)

        # Assemble grid layout
        final_grid = self._assemble_grid(grid_images)
        self._grid_image = final_grid

        print("Grid generation complete!")
        return final_grid

    def _extract_and_render_at_latent(self, z0: float, z1: float) -> Optional[Image.Image]:
        """
        Extract mesh at latent coordinates (z0, z1) and render to image.

        Args:
            z0: First latent coordinate value
            z1: Second latent coordinate value

        Returns:
            PIL Image of rendered mesh, or None if extraction failed
        """
        # Create latent vector: [z0, z1, 0, 0, ..., 0]
        z_vector = torch.zeros(self.z_dim, device=self.device)
        z_vector[0] = z0
        z_vector[1] = z1

        # Set model's latent vector
        with torch.no_grad():
            self.model.set_latent_vector(z_vector)

            # Extract mesh using marching cubes
            # Grid was already configured in __init__, just call extract_mesh
            try:
                verts, faces = self.mc_extractor.extract_mesh(
                    model=self.model,
                    isovalue=self.isovalue,
                    batch_size=8192,
                )
            except Exception as e:
                print(f"    Warning: Mesh extraction failed: {e}")
                return None

            # Check if we got valid mesh data
            if verts is None or faces is None or len(verts) == 0 or len(faces) == 0:
                print("    Warning: Empty mesh returned from extractor")
                return None

            # Render to image using PyVista off-screen renderer
            try:
                image = self._render_mesh_to_image(verts, faces)
                if not self._has_logged_first_nonempty_render:
                    print(
                        "First non-empty mesh rendered successfully: "
                        f"{len(verts)} vertices, {len(faces)} faces"
                    )
                    self._has_logged_first_nonempty_render = True
                return image
            except Exception as e:
                print(f"    Warning: Mesh rendering failed: {e}")
                return None

    def _render_mesh_to_image(self, verts: Tensor, faces: Tensor) -> Image.Image:
        """
        Render mesh to image using PyVista via tim_render helper.

        Args:
            verts: [V, 3] vertex tensor
            faces: [F, 3] face index tensor

        Returns:
            PIL Image of rendered mesh
        """
        plotter = render_mesh(
            verts=verts,
            faces=faces,
            show=False,
            color=self.mesh_color,
            show_edges=False,
            origin=False,
        )
        plotter.camera_position = self.camera_view

        # Render to numpy array
        image_array = plotter.screenshot(
            transparent_background=False,
            return_img=True,
            window_size=self.window_size,
        )

        # Convert to PIL Image
        image = Image.fromarray(image_array, mode="RGB")

        # Cleanup
        plotter.close()
        del plotter

        return image

    def _assemble_grid(self, grid_images: list[list[Image.Image]]) -> Image.Image:
        """
        Assemble 2D grid of images into single image.

        Args:
            grid_images: 2D list of PIL Images (rows × columns)

        Returns:
            Single PIL Image containing all sub-images in grid layout
        """
        if not grid_images or not grid_images[0]:
            raise ValueError("Empty grid_images")

        img_width, img_height = grid_images[0][0].size
        n_rows = len(grid_images)
        n_cols = len(grid_images[0])

        # Create large canvas
        grid_width = n_cols * img_width
        grid_height = n_rows * img_height
        grid_image = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))

        # Paste images into grid
        for i, row in enumerate(grid_images):
            for j, img in enumerate(row):
                x = j * img_width
                y = i * img_height
                grid_image.paste(img, (x, y))

        return grid_image

    def save_interpolation_grid(self, filepath: Optional[Path] = None) -> Path:
        """
        Save interpolation grid to file.

        Args:
            filepath: Output file path (optional, uses save_dir if not provided)

        Returns:
            Path to saved image
        """
        if self._grid_image is None:
            raise ValueError("No grid image generated. Call generate_interpolation_grid() first.")

        if filepath is None:
            filepath = self.save_dir / "interpolation_grid.png"
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._grid_image.save(filepath)
        print(f"Saved interpolation grid to {filepath}")

        return filepath

    def upload_to_wandb(
        self,
        key: str = "latent_interpolation",
        step: Optional[int] = None,
    ) -> None:
        """
        Upload interpolation grid to WandB.

        Args:
            key: WandB log key for the image
            step: Training step/epoch for WandB logging
        """
        if not self.upload:
            print("WandB upload disabled in config")
            return

        try:
            import wandb

            if wandb.run is None:
                print("WandB not initialized, skipping upload")
                return

            # Convert PIL Image to numpy array
            image_array = np.array(self._grid_image)

            # Log to wandb
            wandb.log(
                {key: wandb.Image(image_array, caption="Latent Space Interpolation Grid")},
                step=step,
            )
            print(f"Uploaded interpolation grid to WandB (key={key})")

        except ImportError:
            print("WandB not installed, skipping upload")
        except Exception as e:
            print(f"Warning: WandB upload failed: {e}")

    def cleanup(self) -> None:
        """Clean up resources."""
        self._grid_image = None
        if hasattr(self, "mc_extractor"):
            del self.mc_extractor
