"""
Latent Space Interpolator for SDF Visualization

Generates 2D grid visualizations showing how a neural SDF model's mesh evolves
as you interpolate through the first two latent dimensions (z0, z1).

Uses marching cubes for mesh extraction and PyVista for off-screen rendering.
All parameters are config-driven to avoid hardcoding.
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

import wandb
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.pyvista_headless import (
    ensure_pyvista_headless,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.custom_modes.mesh_assembly_mode_v4 import (
    MeshAssemblyModeV4,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.GINN_visuals.tim_render import (
    render_mesh,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.visualiser import Visualiser

# Import after checking availability to give helpful error message
try:
    pv = ensure_pyvista_headless()
except ImportError as exc:
    raise ImportError("PyVista is required for interpolator. Install with: pip install pyvista") from exc

from subprojects.wp3_solar_estimator.mesh_extractors.non_diff_algos.marching_cubes import (
    MarchingCubesExtractor,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat


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
            - render_backend: "pyvista" (default) or "mesh_assembly_mode_v4"
            - mesh_assembly_mode_config: optional dict passed to MeshAssemblyModeV4
            - mesh_visualiser: optional dict used for Visualiser(...) kwargs
        device: torch device (cuda or cpu)
    """

    def __init__(self, model: SupportsLatentConcat, config: dict, device: torch.device | str):
        """
        Initialize interpolator with model and config.

        Args:
            model: Neural SDF model with set_latent_vector() method
            config: Configuration dict (see class docstring for keys)
            device: torch device
        """
        self.model: SupportsLatentConcat = model
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device

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

        raw_camera_view = config.get("camera_view", "iso")
        if isinstance(raw_camera_view, list):
            # Convert list of lists to tuple of tuples for PyVista camera_position
            # Format as tuple in shape: ( (cam_x, cam_y, cam_z), (focal_x, focal_y, focal_z), (up_x, up_y, up_z) )
            self.camera_view = tuple(tuple(float(v) for v in x) for x in raw_camera_view)
        else:
            self.camera_view = raw_camera_view

        self.mesh_color = config.get("mesh_color", "lightgrey")
        self.window_size = tuple(config.get("window_size", [800, 600]))
        self.save_dir = Path(config.get("save_dir", "./interpolation_output"))
        self.upload = config.get("upload_to_wandb", True)
        self.verbose = config.get("verbose", False)
        self.render_backend = str(config.get("render_backend", "pyvista")).strip().lower()
        self._pyvista_preflight_done = False

        # Optional mesh-assembly mode backend state
        self._mesh_assembly_mode: MeshAssemblyModeV4 | None = None

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Configure the marching cubes grid (call once during init)
        self.mc_extractor.set_grid(bbox=self.bbox, resolution=self.mesh_resolution)

        # Get model's latent dimension (try z_dim first, then check for _active_z attribute)
        self.z_dim: int | None = None
        if hasattr(model, "z_dim"):
            self.z_dim = model.z_dim
        else:
            active_z = getattr(model, "_active_z", None)
            if active_z is not None:
                self.z_dim = int(active_z.shape[0])

        if self.z_dim is None:
            raise ValueError("Model must have z_dim attribute (latent vector dimension)")

        # Placeholder for compiled grid image
        self._grid_image: Image.Image | None = None
        self._has_logged_first_nonempty_render = False

        self._configure_render_backend()

    def _configure_render_backend(self) -> None:
        """Initialize the selected render backend and run required preflight checks."""
        if self.render_backend == "mesh_assembly_mode_v4":
            self._init_mesh_assembly_backend()

        if self.render_backend == "pyvista":
            self._ensure_pyvista_ready()

    def _ensure_pyvista_ready(self) -> None:
        """Run the PyVista preflight once."""
        if not self._pyvista_preflight_done:
            self._run_render_preflight()
            self._pyvista_preflight_done = True

    def _disable_mesh_backend(self, reason: str) -> None:
        """Downgrade to PyVista backend after a mesh-backend failure."""
        print(f"Warning: {reason}. Falling back to PyVista backend.")
        self._mesh_assembly_mode = None
        self.render_backend = "pyvista"
        self._ensure_pyvista_ready()

    def _run_render_preflight(self) -> None:
        """Run a tiny off-screen render probe to validate backend before main loop."""
        probe_failed = False
        try:
            probe_plotter = pv.Plotter(off_screen=True, window_size=(128, 128))
            probe_plotter.add_mesh(pv.Sphere(radius=1.0), color="white")
            probe_image = probe_plotter.screenshot(transparent_background=False)
            probe_plotter.close()

            probe_failed = probe_image is None or getattr(probe_image, "size", 0) == 0
        except Exception as exc:
            raise RuntimeError(
                "PyVista headless preflight failed before interpolation. "
                f"DISPLAY={os.environ.get('DISPLAY', '<unset>')} "
                f"PYVISTA_OFF_SCREEN={os.environ.get('PYVISTA_OFF_SCREEN', '<unset>')}"
            ) from exc
        if probe_failed:
            raise RuntimeError("Render probe produced no image output")
        print(f"PyVista render preflight passed (DISPLAY={os.environ.get('DISPLAY', '<unset>')})")

    def _init_mesh_assembly_backend(self) -> None:
        """Initialize mesh_assembly_mode_v4 rendering backend."""
        try:
            mode_config = dict(self.config.get("mesh_assembly_mode_config", {}))

            # Derive camera defaults from existing interpolator camera config if possible.
            camera_position, look_at = self._resolve_mode_camera_defaults()
            mode_config.setdefault("camera_position", camera_position)
            mode_config.setdefault("look_at", look_at)
            mode_config.setdefault("render_mode", "rgb")
            mode_config.setdefault("render_interval", 1)
            mode_config.setdefault("make_gif", False)

            visualiser_kwargs = dict(self.config.get("mesh_visualiser", {}))
            visualiser_kwargs.setdefault("device", str(self.device))
            visualiser_kwargs.setdefault("image_size", int(min(self.window_size)))

            visualiser = Visualiser(**visualiser_kwargs)
            self._mesh_assembly_mode = MeshAssemblyModeV4(visualiser=visualiser, config=mode_config)
            print("Interpolator mesh assembly backend initialized")
        except Exception as exc:
            self._disable_mesh_backend(f"Failed to initialize mesh assembly backend: {exc}")

    def _resolve_mode_camera_defaults(self) -> tuple[list[float], list[float]]:
        """Resolve camera defaults for mode-based rendering from interpolator config."""
        if (
            isinstance(self.camera_view, tuple)
            and len(self.camera_view) == 3
            and all(isinstance(v, tuple) and len(v) == 3 for v in self.camera_view)
        ):
            return [float(x) for x in self.camera_view[0]], [float(x) for x in self.camera_view[1]]

        return [5000.0, 5000.0, 5000.0], [0.0, 0.0, 0.0]

    def generate_interpolation_grid(self, epoch: int = 0) -> Image.Image:
        """
        Generate 2D grid of mesh visualizations interpolating through latent space.

        Sweeps through z[0] and z[1] coordinates in a grid pattern, extracting
        and rendering a mesh at each point. Intermediate images saved to disk,
        only final grid image returned.

        Args:
            epoch: Current training epoch, used to label frames in mesh_assembly_mode_v4

        Returns:
            PIL Image of final 2D grid layout
        """
        print(f"Generating {self.grid_resolution}x{self.grid_resolution} interpolation grid...")

        # Generate latent coordinate grid
        z0_values = np.linspace(self.latent_range[0], self.latent_range[1], self.grid_resolution)
        z1_values = np.linspace(self.latent_range[0], self.latent_range[1], self.grid_resolution)

        # Store image paths for grid assembly
        grid_images = []

        # Sweep through latent space
        for i, z0 in enumerate(z0_values):
            row_images = []
            for j, z1 in enumerate(z1_values):
                if self.verbose:
                    print(f"  Point ({i + 1}/{self.grid_resolution}, {j + 1}/{self.grid_resolution}): z=[{z0:.3f}, {z1:.3f}]")

                # Extract and render mesh at this latent point
                image = self._extract_and_render_at_latent(z0, z1, epoch)
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

    def clear_image_cache(self) -> None:
        """Clear any cached images to free memory."""
        self._grid_image = None

    def _extract_and_render_at_latent(self, z0: float, z1: float, epoch: int = 0) -> Image.Image | None:
        """
        Extract mesh at latent coordinates (z0, z1) and render to image.

        Args:
            z0: First latent coordinate value
            z1: Second latent coordinate value
            epoch: Current training epoch, forwarded to the render backend for labelling

        Returns:
            PIL Image of rendered mesh, or None if extraction failed
        """
        if self.z_dim is None:
            raise RuntimeError("Interpolator z_dim is not initialized")

        # Create latent vector: [z0, z1, 0, 0, ..., 0]
        z_vector = torch.zeros(self.z_dim, device=self.device)
        z_vector[0] = z0
        z_vector[1] = z1

        mesh_data: tuple[Tensor, Tensor] | None
        with torch.no_grad():
            self.model.set_latent_vector(z_vector)
            mesh_data = self._extract_mesh()

        if mesh_data is None:
            return None

        verts, faces = mesh_data
        image = self._render_mesh_to_image(verts, faces, epoch)
        if image is None:
            return None

        if not self._has_logged_first_nonempty_render:
            print(f"First non-empty mesh rendered successfully: {len(verts)} vertices, {len(faces)} faces")
            self._has_logged_first_nonempty_render = True
        return image

    def _extract_mesh(self) -> tuple[Tensor, Tensor] | None:
        """Extract a mesh from the current model state."""
        try:
            verts, faces = self.mc_extractor.extract_mesh(
                model=self.model,
                isovalue=self.isovalue,
                batch_size=8192,
            )
        except Exception as exc:
            print(f"    Warning: Mesh extraction failed: {exc}")
            return None

        if verts is None or faces is None or len(verts) == 0 or len(faces) == 0:
            print("    Warning: Empty mesh returned from extractor")
            return None

        return verts, faces

    def _render_mesh_to_image(self, verts: Tensor, faces: Tensor, epoch: int = 0) -> Image.Image | None:
        """
        Render mesh to image using selected backend.

        Args:
            verts: [V, 3] vertex tensor
            faces: [F, 3] face index tensor
            epoch: Current training epoch, forwarded to mesh_assembly_mode_v4 for labelling

        Returns:
            PIL Image of rendered mesh, or None if rendering failed
        """
        if self.render_backend == "mesh_assembly_mode_v4" and self._mesh_assembly_mode is not None:
            try:
                return self._resize_to_window(self._render_mesh_to_image_with_mesh_assembly(verts, faces, epoch))
            except Exception as exc:
                self._disable_mesh_backend(f"mesh_assembly_mode_v4 render failed: {exc}")

        try:
            return self._render_mesh_to_image_with_pyvista(verts, faces)
        except Exception as exc:
            print(f"    Warning: PyVista rendering failed: {exc}")
            return None

    def _resize_to_window(self, image: Image.Image) -> Image.Image:
        """Resize image to target window size if needed."""
        if image.size == self.window_size:
            return image

        return image.resize(self.window_size, resample=Image.BILINEAR)

    def _render_mesh_to_image_with_mesh_assembly(self, verts: Tensor, faces: Tensor, epoch: int = 0) -> Image.Image:
        """Render one mesh with MeshAssemblyModeV4 and convert to PIL image."""
        if self._mesh_assembly_mode is None:
            raise RuntimeError("mesh_assembly_mode_v4 backend is not initialized")

        optimization_state = {
            "iteration": epoch,
            "vertices": verts.detach(),
            "faces": faces.detach(),
        }

        self._mesh_assembly_mode.update(optimization_state)

        if not self._mesh_assembly_mode.frame_storage:
            raise RuntimeError("mesh_assembly_mode_v4 did not produce a frame")

        frame = self._mesh_assembly_mode.frame_storage[-1]
        frame_uint8 = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        if frame_uint8.ndim == 3 and frame_uint8.shape[-1] == 1:
            frame_uint8 = np.repeat(frame_uint8, 3, axis=-1)

        # We only need a single frame per interpolation point.
        self._mesh_assembly_mode.clear_frames()
        return Image.fromarray(frame_uint8, mode="RGB")

    def _render_mesh_to_image_with_pyvista(self, verts: Tensor, faces: Tensor) -> Image.Image:
        """Render mesh to image using PyVista via tim_render helper."""
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
            grid_images: 2D list of PIL Images (rows x columns)

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

    def save_interpolation_grid(self, filepath: Path | None = None, clear_cache: bool = False) -> Path:
        """
        Save interpolation grid to file.

        Args:
            filepath: Output file path (optional, uses save_dir if not provided)

        Returns:
            Path to saved image
        """
        if self._grid_image is None:
            raise ValueError("No grid image generated. Call generate_interpolation_grid() first.")

        filepath = self.save_dir / "interpolation_grid.png" if filepath is None else Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._grid_image.save(filepath)
        print(f"Saved interpolation grid to {filepath}")
        if clear_cache:
            self.clear_image_cache()
        return filepath

    def upload_to_wandb(
        self,
        key: str = "latent_interpolation",
        step: int | None = None,
        clear_cache: bool = True,
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
            if wandb.run is None:
                print("WandB not initialized, skipping upload")
                return

            # Convert PIL Image to numpy array
            image_array = np.array(self._grid_image)

            # Log to wandb
            log_payload = {key: wandb.Image(image_array, caption="Latent Space Interpolation Grid")}
            if step is None:
                wandb.log(log_payload)
            else:
                wandb.log(log_payload, step=step)
            print(f"Uploaded interpolation grid to WandB (key={key})")
        except Exception as e:
            print(f"Warning: WandB upload failed: {e}")
        if clear_cache:
            self.clear_image_cache()

    def cleanup(self) -> None:
        """Clean up resources."""
        self._grid_image = None
        if hasattr(self, "mc_extractor"):
            del self.mc_extractor
