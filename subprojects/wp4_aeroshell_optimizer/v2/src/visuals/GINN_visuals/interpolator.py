"""
Latent Space Interpolator for SDF Visualization

Generates 2D grid visualizations showing how a neural SDF model's mesh evolves
as you interpolate through the first two latent dimensions (z0, z1).

Uses marching cubes for mesh extraction and PyVista for off-screen rendering.
All parameters are config-driven to avoid hardcoding.
"""

import os
from dataclasses import replace as _dc_replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

import wandb
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.pyvista_headless import (
    ensure_pyvista_headless,
    is_headless_runtime,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.custom_modes.mesh_assembly_mode_v4 import (
    MeshAssemblyModeV4,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.GINN_visuals.tim_render import (
    render_mesh,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.visualiser import Visualiser

# pyvista is imported lazily inside _ensure_pyvista_ready() to avoid triggering
# VTK's X11 initialisation on headless HPC nodes at import time.
_pv = None  # module-level cache; populated on first use of the pyvista backend

from subprojects.wp2_assembler.version4.utils.reg_imp import get_outer_prism_bounds_mm, scale_outer_prism
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat
from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor, load_mesh_extraction_config


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
            - latent_range: [min, max], bounds for z[0] and z[1] (grid mode only)
            - mesh_resolution: int, marching cubes grid resolution (e.g., 128)
            - bbox_scale_factor: float, scale applied to the WP2 outer prism bbox (default 1.0)
            - isovalue: float, SDF level set (default 0.0)
            - sample_mode: "grid" or "circle"
            - z_radius: float, circle radius matching pretraining z_radius (default 2.0)
            - n_circle_samples: int, number of circle samples (default 12)
            - save_plys: bool, write PLY files for each sample to save_dir/plys/
            - camera_view: str, camera angle (iso, front, back, left, right, top, bottom)
            - mesh_color: str, PyVista color (e.g., "lightgrey")
            - window_size: (width, height) tuple for rendered images
            - save_dir: Path for saving intermediate/final images
            - upload_to_wandb: bool, whether to upload grid to WandB
            - render_backend: "mesh_assembly_mode_v4" (default) or "pyvista" (fallback)
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

        # Parse config
        self.grid_resolution = int(config.get("grid_resolution", 10))
        self.latent_range = config.get("latent_range", [-2.0, 2.0])
        self.mesh_resolution = int(config.get("mesh_resolution", 128))
        self.isovalue = float(config.get("isovalue", 0.0))

        # Build WP5 mesh extractor using outer_prism bounds from WP2 with optional scaling.
        # bbox_scale_factor > 1.0 expands the evaluation domain (useful for geometries that
        # slightly exceed the nominal prism), < 1.0 contracts it for faster extraction.
        _bbox_scale_factor = float(config.get("bbox_scale_factor", 1.0))
        _mesh_config = load_mesh_extraction_config()
        _prism = scale_outer_prism(get_outer_prism_bounds_mm(), _bbox_scale_factor)
        _mesh_config = _dc_replace(_mesh_config, device=str(self.device), isovalue=self.isovalue)
        self.mc_extractor = MeshSDFExtractor(config=_mesh_config)
        self.mc_extractor.set_grid(
            bbox=[
                [_prism["x"][0], _prism["x"][1]],
                [_prism["y"][0], _prism["y"][1]],
                [_prism["z"][0], _prism["z"][1]],
            ],
            resolution=self.mesh_resolution,
        )

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
        self.sample_mode = str(config.get("sample_mode", "grid")).strip().lower()
        self.z_radius = float(config.get("z_radius", 2.0))
        self.n_circle_samples = int(config.get("n_circle_samples", 12))
        self.save_plys = bool(config.get("save_plys", False))
        self._pyvista_preflight_done = False

        # Optional mesh-assembly mode backend state
        self._mesh_assembly_mode: MeshAssemblyModeV4 | None = None

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

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
            # mesh_assembly_mode_v4 is the production backend.  PyVista is only
            # activated below when explicitly requested or after a fallback.

        if self.render_backend == "pyvista":
            # PyVista path: used when render_backend is explicitly set to "pyvista",
            # or after _disable_mesh_backend() downgrades from mesh_assembly_mode_v4.
            self._ensure_pyvista_ready()

    def _ensure_pyvista_ready(self) -> None:
        """Import and configure PyVista for off-screen rendering, or raise if not available."""
        global _pv  # noqa: PLW0603
        if _pv is not None:
            return
        if is_headless_runtime():
            raise RuntimeError(
                "PyVista rendering is not available on this headless node (no X server / no OSMesa VTK). "
                "Use render_backend: mesh_assembly_mode_v4 or disable rendering."
            )
        try:
            _pv = ensure_pyvista_headless()
        except ImportError as exc:
            raise ImportError("PyVista is required for pyvista backend. Install with: pip install pyvista") from exc
        if not self._pyvista_preflight_done:
            self._run_render_preflight()
            self._pyvista_preflight_done = True

    def _disable_mesh_backend(self, reason: str) -> None:
        """Downgrade to PyVista backend after a mesh-backend failure, or disable rendering on headless nodes."""
        if is_headless_runtime():
            # Headless compute nodes have no X server / OSMesa VTK — pyvista will abort.
            # Disable rendering entirely rather than crashing.
            print(f"Warning: {reason}. Headless node detected — rendering disabled for this run.")
            self._mesh_assembly_mode = None
            self.render_backend = "none"
            return
        print(f"Warning: {reason}. Falling back to PyVista backend.")
        self._mesh_assembly_mode = None
        self.render_backend = "pyvista"
        self._ensure_pyvista_ready()

    def _run_render_preflight(self) -> None:
        """Run a tiny off-screen render probe to validate PyVista before the main loop."""
        global _pv
        if _pv is None:
            raise RuntimeError("_run_render_preflight called before PyVista was imported via _ensure_pyvista_ready.")
        probe_failed = False
        try:
            probe_plotter = _pv.Plotter(off_screen=True, window_size=(128, 128))
            probe_plotter.add_mesh(_pv.Sphere(radius=1.0), color="white")
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
        if self.sample_mode == "circle":
            n = self.n_circle_samples
            print(f"Generating circle strip ({n} points, r={self.z_radius})...")
            angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            strip: list[Image.Image] = []
            for idx, angle in enumerate(angles):
                z0 = float(self.z_radius * np.cos(angle))
                z1 = float(self.z_radius * np.sin(angle))
                if self.verbose:
                    print(f"  [{idx + 1}/{n}] angle={np.degrees(angle):.1f}° z=[{z0:.3f}, {z1:.3f}]")
                ply_label = f"circle_{idx:02d}" if self.save_plys else None
                image = self._extract_and_render_at_latent(z0, z1, epoch, ply_label=ply_label)
                strip.append(image or Image.new("RGB", self.window_size, (200, 200, 200)))
            final_grid = self._assemble_grid([strip])
        else:
            print(f"Generating {self.grid_resolution}x{self.grid_resolution} interpolation grid...")
            z0_values = np.linspace(self.latent_range[0], self.latent_range[1], self.grid_resolution)
            z1_values = np.linspace(self.latent_range[0], self.latent_range[1], self.grid_resolution)
            grid_images = []
            for i, z0 in enumerate(z0_values):
                row_images = []
                for j, z1 in enumerate(z1_values):
                    if self.verbose:
                        print(f"  Point ({i + 1}/{self.grid_resolution}, {j + 1}/{self.grid_resolution}): z=[{z0:.3f}, {z1:.3f}]")
                    ply_label = f"grid_r{i:02d}_c{j:02d}" if self.save_plys else None
                    image = self._extract_and_render_at_latent(z0, z1, epoch, ply_label=ply_label)
                    row_images.append(image or Image.new("RGB", self.window_size, (200, 200, 200)))
                grid_images.append(row_images)
            final_grid = self._assemble_grid(grid_images)

        self._grid_image = final_grid
        print("Grid generation complete!")
        return final_grid

    def clear_image_cache(self) -> None:
        """Clear any cached images to free memory."""
        self._grid_image = None

    def _save_ply(self, verts: Tensor, faces: Tensor, label: str) -> None:
        """Save mesh as PLY to save_dir/plys/{label}.ply."""
        import trimesh  # noqa: PLC0415

        ply_dir = self.save_dir / "plys"
        ply_dir.mkdir(parents=True, exist_ok=True)
        mesh = trimesh.Trimesh(
            vertices=verts.detach().cpu().numpy(),
            faces=faces.detach().cpu().numpy(),
        )
        out_path = ply_dir / f"{label}.ply"
        mesh.export(str(out_path))
        if self.verbose:
            print(f"    Saved PLY: {out_path}")

    def _extract_and_render_z_vector(self, z_vector: Tensor, epoch: int = 0, ply_label: str | None = None) -> Image.Image | None:
        """Set full z_vector on model, extract mesh, optionally save PLY, render to image."""
        with torch.no_grad():
            self.model.set_latent_vector(z_vector)
            mesh_data = self._extract_mesh()

        if mesh_data is None:
            return None

        verts, faces = mesh_data
        if ply_label is not None and self.save_plys:
            self._save_ply(verts, faces, ply_label)

        image = self._render_mesh_to_image(verts, faces, epoch)
        if image is None:
            return None

        if not self._has_logged_first_nonempty_render:
            print(f"First non-empty mesh rendered successfully: {len(verts)} vertices, {len(faces)} faces")
            self._has_logged_first_nonempty_render = True
        return image

    def _extract_and_render_at_latent(
        self, z0: float, z1: float, epoch: int = 0, ply_label: str | None = None
    ) -> Image.Image | None:
        """Build z_vector from z0/z1 (other dims zero) and delegate to _extract_and_render_z_vector."""
        if self.z_dim is None:
            raise RuntimeError("Interpolator z_dim is not initialized")
        z_vector = torch.zeros(self.z_dim, device=self.device)
        z_vector[0] = z0
        z_vector[1] = z1
        return self._extract_and_render_z_vector(z_vector, epoch, ply_label)

    def _extract_mesh(self) -> tuple[Tensor, Tensor] | None:
        """Extract a mesh from the current model state."""
        try:
            verts, faces = self.mc_extractor.extract_mesh(sdf_model=self.model)
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
            PIL Image of rendered mesh, or None if rendering failed or disabled
        """
        if self.render_backend == "none":
            return None

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
        global _pv
        if _pv is None:
            self._ensure_pyvista_ready()
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


if __name__ == "__main__":
    import argparse

    from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
    from subprojects.wp4_aeroshell_optimizer.v2.src.utils.dynamic_loading import smart_model_init

    def _parse_cli() -> argparse.Namespace:
        p = argparse.ArgumentParser(
            description="Standalone latent-space visualizer: render PLY/PNG from a trained checkpoint.",
        )
        p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint (must contain model_state_dict).")
        p.add_argument("--model-name", required=True, dest="model_name", help="Model name as registered in models.json.")
        p.add_argument(
            "--config",
            default="postprocessing_interpolator_config.yaml",
            help="Interpolator config filename (basename inside v2 configs dir).",
        )
        p.add_argument("--out-dir", default=None, dest="out_dir", help="Output directory override.")
        p.add_argument("--device", default=None, help="Compute device (cpu/cuda). Default: auto-detect.")
        p.add_argument("--epoch", type=int, default=0, help="Epoch label forwarded to render backend.")
        mode = p.add_mutually_exclusive_group()
        mode.add_argument(
            "--z-from-checkpoint",
            action="store_true",
            dest="z_from_checkpoint",
            help="Render each pretrained slot from ckpt['z_pretrained'] as a PLY (uses full z vectors).",
        )
        mode.add_argument(
            "--z-vectors",
            nargs="+",
            dest="z_vectors",
            metavar="Z",
            help="Explicit z vectors as comma-separated floats, e.g. '0.5,1.5' '-1.0,0.0'. "
            "Values are zero-padded to model z_dim.",
        )
        return p.parse_args()

    _args = _parse_cli()
    _device = _args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    _model, _, _, _ = smart_model_init(_args.model_name)
    _ckpt = torch.load(_args.checkpoint, map_location=_device, weights_only=True)
    if not isinstance(_ckpt, dict) or "model_state_dict" not in _ckpt:
        raise ValueError("Checkpoint must contain 'model_state_dict'. Use a latent pretraining checkpoint.")
    _model.load_state_dict(_ckpt["model_state_dict"])
    _model.eval()
    _model.to(_device)

    _loaded_cfg = load_config(v="v2", cfg_name=_args.config)
    _interp_cfg: dict = dict(_loaded_cfg.get("interpolator", {}))
    if _args.out_dir:
        _interp_cfg["save_dir"] = _args.out_dir
    _interp_cfg.setdefault("upload_to_wandb", False)
    _interp_cfg["save_plys"] = True  # always save PLYs in standalone mode

    _interp = Interpolator(model=_model, config=_interp_cfg, device=_device)

    if _interp.z_dim is None:
        raise RuntimeError("Model has no z_dim — ensure you are using a latent model (e.g. ConcatWIREV2).")

    if _args.z_from_checkpoint:
        if "z_pretrained" not in _ckpt:
            raise ValueError(
                "Checkpoint does not contain 'z_pretrained'. Use a latent pretraining checkpoint or pass --z-vectors instead."
            )
        _Z = _ckpt["z_pretrained"].to(_device)
        for _slot_idx in range(len(_Z)):
            _interp._extract_and_render_z_vector(_Z[_slot_idx], epoch=_args.epoch, ply_label=f"slot_{_slot_idx:02d}")
        print(f"[standalone] Exported {len(_Z)} PLYs to {_interp.save_dir / 'plys'}")

    elif _args.z_vectors:
        for _idx, _zv_str in enumerate(_args.z_vectors):
            _parts = [float(x) for x in _zv_str.split(",")]
            _z_vec = torch.zeros(_interp.z_dim, device=_device)
            for _d, _val in enumerate(_parts[: _interp.z_dim]):
                _z_vec[_d] = _val
            _interp._extract_and_render_z_vector(_z_vec, epoch=_args.epoch, ply_label=f"z{_idx:02d}")
        print(f"[standalone] Exported {len(_args.z_vectors)} PLYs to {_interp.save_dir / 'plys'}")

    else:
        _grid = _interp.generate_interpolation_grid(epoch=_args.epoch)
        _out_path = _interp.save_dir / "standalone_grid.png"
        _grid.save(_out_path)
        print(f"[standalone] Grid saved to {_out_path}")
