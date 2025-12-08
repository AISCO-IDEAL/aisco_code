import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import imageio
import numpy as np
import torch

from subprojects.fixed_renderer.renderer import Renderer as NeuralRenderer


class Visualiser:
    """
    A wrapper class for fixed_renderer to easily visualize 3D meshes.

    Features:
    - Render meshes with custom textures
    - Shade meshes based on vertex coordinates
    - Customize camera positions
    - Generate rotating GIF animations
    """

    def __init__(
        self,
        image_size: int = 256,
        background_color: list[float] = [0, 0, 0],
        anti_aliasing: bool = True,
        camera_mode: str = 'look_at',
        viewing_angle: float = 30,
        device: str = 'cuda'
    ):
        """
        Initialize the Visualizer.

        Args:
            image_size: Output image resolution (width and height)
            background_color: RGB background color [R, G, B] in range [0, 1]
            anti_aliasing: Enable anti-aliasing for smoother edges
            camera_mode: 'look_at', 'look', or 'projection'
            viewing_angle: Field of view angle in degrees
            device: 'cuda' or 'cpu'
        """
        self.image_size = image_size
        self.device = device
        self.camera_mode = camera_mode

        self.renderer = NeuralRenderer(
            image_size=image_size,
            anti_aliasing=anti_aliasing,
            background_color=background_color,
            camera_mode=camera_mode,
            viewing_angle=viewing_angle,
            orig_size=image_size
        )
        #Ensure whole car is inside clipping planes
        self.renderer.near = 0.1
        self.renderer.far = 20000.0
    def render_mesh(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        textures: torch.Tensor | None = None,
        mode: str = 'rgb',
        camera_position: list[float] | None = None,
        camera_direction: list[float] | None = None
    ) -> np.ndarray:
        """
        Render a single view of a mesh.

        Args:
            vertices: Vertex positions, shape (batch_size, num_vertices, 3)
            faces: Face indices, shape (batch_size, num_faces, 3)
            textures: Texture per face, shape (batch_size, num_faces, texture_size, texture_size, texture_size, 3)
                     If None, uses default white texture
            mode: 'rgb', 'silhouettes', or 'depth'
            camera_position: Camera eye position [x, y, z]
            camera_direction: Camera look direction [x, y, z]

        Returns:
            Rendered image as numpy array, shape (batch_size, height, width, 3) for RGB
        """
        # Ensure tensors are on the correct device
        vertices = self._to_device(vertices)
        faces = self._to_device(faces)

        # Set camera if provided
        if camera_position is not None:
            self.renderer.eye = camera_position
        if camera_direction is not None:
            self.renderer.camera_direction = camera_direction

        # Create default texture if not provided
        if textures is None and mode == 'rgb':
            textures = self._create_default_texture(faces.shape[0], faces.shape[1])
        else:
            textures = self._to_device(textures)

        # Render
        with torch.no_grad():
            if mode == 'rgb':
                output = self.renderer.render_rgb(vertices, faces, textures)
            elif mode == 'silhouettes':
                output = self.renderer.render_silhouettes(vertices, faces)
            elif mode == 'depth':
                output = self.renderer.render_depth(vertices, faces)
            else:
                raise ValueError(f"Unknown mode: {mode}")

        # Convert to numpy
        return self._to_numpy(output)

    def render_with_vertex_colors(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        vertex_colors: torch.Tensor,
        camera_position: list[float] | None = None
    ) -> np.ndarray:
        """
        Render mesh with per-vertex colors.

        Args:
            vertices: Vertex positions, shape (batch_size, num_vertices, 3)
            faces: Face indices, shape (batch_size, num_faces, 3)
            vertex_colors: RGB colors per vertex, shape (batch_size, num_vertices, 3), range [0, 1]
            camera_position: Camera eye position [x, y, z]

        Returns:
            Rendered RGB image as numpy array
        """
        textures = self.vertex_colors_to_textures(vertex_colors, faces)
        return self.render_mesh(vertices, faces, textures, mode='rgb', camera_position=camera_position)

    def render_with_coordinate_shading(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        coordinate: str = 'z',
        colormap: str = 'viridis',
        camera_position: list[float] | None = None
    ) -> np.ndarray:
        """
        Render mesh with colors based on vertex coordinates.

        Args:
            vertices: Vertex positions, shape (batch_size, num_vertices, 3)
            faces: Face indices, shape (batch_size, num_faces, 3)
            coordinate: 'x', 'y', or 'z' - which coordinate to use for coloring
            colormap: Color mapping strategy ('viridis', 'plasma', 'coolwarm', or 'grayscale')
            camera_position: Camera eye position [x, y, z]

        Returns:
            Rendered RGB image as numpy array
        """
        vertex_colors = self._coordinate_to_colors(vertices, coordinate, colormap)
        return self.render_with_vertex_colors(vertices, faces, vertex_colors, camera_position)

    def create_rotating_gif(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        textures: torch.Tensor | None = None,
        output_path: str | Path = 'rotation.gif',
        num_frames: int = 36,
        elevation: float = 30,
        distance: float = 2.5,
        fps: int = 10,
        axis: str = 'y'
    ) -> None:
        """
        Create a rotating GIF animation of the mesh.

        Args:
            vertices: Vertex positions, shape (batch_size, num_vertices, 3)
            faces: Face indices, shape (batch_size, num_faces, 3)
            textures: Optional textures for the mesh
            output_path: Path to save the GIF file
            num_frames: Number of frames in the animation
            elevation: Elevation angle in degrees
            distance: Distance from origin
            fps: Frames per second
            axis: Rotation axis ('x', 'y', or 'z')
        """
        frames = []

        for i in range(num_frames):
            # Calculate rotation angle
            angle = 2 * math.pi * i / num_frames

            # Calculate camera position
            if axis == 'y':
                x = distance * math.cos(angle) * math.cos(math.radians(elevation))
                y = distance * math.sin(math.radians(elevation))
                z = distance * math.sin(angle) * math.cos(math.radians(elevation))
            elif axis == 'x':
                x = distance * math.sin(math.radians(elevation))
                y = distance * math.cos(angle) * math.cos(math.radians(elevation))
                z = distance * math.sin(angle) * math.cos(math.radians(elevation))
            elif axis == 'z':
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)
                z = distance * math.sin(math.radians(elevation))
            else:
                raise ValueError(f"Unknown axis: {axis}")

            camera_position = [x, y, z]

            # Render frame
            frame = self.render_mesh(
                vertices, faces, textures,
                mode='rgb',
                camera_position=camera_position,
                camera_direction=[0, 0, 0]  # Look at origin
            )

            # Convert to uint8 and remove batch dimension
            frame = (frame[0] * 255).astype(np.uint8)
            frames.append(frame)

        # Save as GIF
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Saved rotating GIF to {output_path}")

    def create_multi_view_image(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        textures: torch.Tensor | None = None,
        views: list[tuple[float, float, float]] = None,
        distance: float = 2000.0,
        labels: list[str] | None = None
    ) -> np.ndarray:
        """
        Create a grid of multiple views of the same mesh.
        ### Not tested yet ###
        Args:
            vertices: Vertex positions
            faces: Face indices
            textures: Optional textures
            views: List of camera positions [(x, y, z), ...]
            labels: Optional labels for each view

        Returns:
            Grid image as numpy array
        """
        if views is None:
            # Default views: front, side, top, perspective
            d = distance
            views = [
                (0, 0, d),      # Front
                (d, 0, 0),      # Side
                (0, d, 0),      # Top
                (d, d, d)       # Perspective
            ]

        rendered_views = []
        for view_pos in views:
            img = self.render_mesh(
                vertices, faces, textures,
                camera_position=list(view_pos)
            )
            rendered_views.append(img[0])  # Remove batch dimension

        # Arrange in grid (2x2 for 4 views, adjust as needed)
        n_views = len(rendered_views)
        grid_size = int(np.ceil(np.sqrt(n_views)))

        # Pad with blank images if needed
        while len(rendered_views) < grid_size * grid_size:
            rendered_views.append(np.zeros_like(rendered_views[0]))

        # Create grid
        rows = []
        for i in range(grid_size):
            row = np.concatenate(rendered_views[i*grid_size:(i+1)*grid_size], axis=1)
            rows.append(row)
        grid = np.concatenate(rows, axis=0)

        return grid

    # ========== Helper Methods ==========

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the correct device."""
        if self.device == 'cuda' and not tensor.is_cuda:
            return tensor.cuda()
        elif self.device == 'cpu' and tensor.is_cuda:
            return tensor.cpu()
        return tensor

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array and transpose to (batch, H, W, C) format."""
        if tensor.is_cuda:
            tensor = tensor.cpu()
        arr = tensor.detach().numpy()
        # If shape is (batch, C, H, W), transpose to (batch, H, W, C)
        if len(arr.shape) == 4 and arr.shape[1] in [1, 3, 4]:
            arr = arr.transpose(0, 2, 3, 1)
        return arr

    def _create_default_texture(self, batch_size: int, num_faces: int, texture_size: int = 2) -> torch.Tensor:
        """Create a default white texture."""
        textures = torch.ones(
            batch_size, num_faces, texture_size, texture_size, texture_size, 3,
            dtype=torch.float32
        )
        return self._to_device(textures)

    def vertex_colors_to_textures(
        self,
        vertex_colors: torch.Tensor,
        faces: torch.Tensor,
        texture_size: int = 2
    ) -> torch.Tensor:
        """
        Convert per-vertex colors to per-face textures.

        Args:
            vertex_colors: (batch_size, num_vertices, 3)
            faces: (batch_size, num_faces, 3)
            texture_size: Texture resolution per face

        Returns:
            textures: (batch_size, num_faces, texture_size, texture_size, texture_size, 3)
        """
        batch_size, num_faces, _ = faces.shape

        # Get colors for each face vertex
        face_colors = torch.zeros(batch_size, num_faces, 3, 3, device=vertex_colors.device)
        for i in range(3):
            face_colors[:, :, i, :] = torch.gather(
                vertex_colors, 1,
                faces[:, :, i:i+1].expand(-1, -1, 3)
            )

        # Average the three vertex colors for each face
        mean_color = face_colors.mean(dim=2)  # (batch_size, num_faces, 3)

        # Expand to texture size
        textures = mean_color.view(batch_size, num_faces, 1, 1, 1, 3).expand(
            -1, -1, texture_size, texture_size, texture_size, -1
        ).contiguous()

        return textures

    def _coordinate_to_colors(
        self,
        vertices: torch.Tensor,
        coordinate: str = 'z',
        colormap: str = 'viridis'
    ) -> torch.Tensor:
        """
        Convert vertex coordinates to colors using a colormap.

        Args:
            vertices: (batch_size, num_vertices, 3)
            coordinate: 'x', 'y', or 'z'
            colormap: 'viridis', 'plasma', 'coolwarm', or 'grayscale'

        Returns:
            colors: (batch_size, num_vertices, 3)
        """
        coord_idx = {'x': 0, 'y': 1, 'z': 2}[coordinate]
        values = vertices[:, :, coord_idx]

        # Normalize to [0, 1]
        values_min = values.min(dim=1, keepdim=True)[0]
        values_max = values.max(dim=1, keepdim=True)[0]
        values_norm = (values - values_min) / (values_max - values_min + 1e-8)

        # Apply colormap
        if colormap == 'grayscale':
            colors = values_norm.unsqueeze(-1).expand(-1, -1, 3)
        elif colormap == 'viridis':
            colors = self._apply_viridis(values_norm)
        elif colormap == 'plasma':
            colors = self._apply_plasma(values_norm)
        elif colormap == 'coolwarm':
            colors = self._apply_coolwarm(values_norm)
        else:
            raise ValueError(f"Unknown colormap: {colormap}")

        return colors

    def _apply_viridis(self, values: torch.Tensor) -> torch.Tensor:
        """Apply viridis colormap (purple -> green -> yellow)."""
        r = 0.267 + 0.004 * values + 2.096 * values**2 - 0.347 * values**3
        g = 0.005 + 0.998 * values + 0.093 * values**2 - 0.096 * values**3
        b = 0.329 + 1.196 * values - 2.045 * values**2 + 0.520 * values**3
        return torch.stack([r, g, b], dim=-1).clamp(0, 1)

    def _apply_plasma(self, values: torch.Tensor) -> torch.Tensor:
        """Apply plasma colormap (purple -> red -> yellow)."""
        r = 0.050 + 2.903 * values - 1.853 * values**2
        g = 0.030 + 0.375 * values + 2.150 * values**2 - 2.555 * values**3
        b = 0.527 + 1.084 * values - 1.611 * values**2
        return torch.stack([r, g, b], dim=-1).clamp(0, 1)

    def _apply_coolwarm(self, values: torch.Tensor) -> torch.Tensor:
        """Apply coolwarm colormap (blue -> white -> red)."""
        r = values
        g = 1.0 - 2.0 * torch.abs(values - 0.5)
        b = 1.0 - values
        return torch.stack([r, g, b], dim=-1).clamp(0, 1)

    def __call__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        textures: torch.Tensor | None = None,
        **kwargs
    ) -> np.ndarray:
        """
        Convenience method to render a mesh.
        Equivalent to calling render_mesh().
        """
        return self.render_mesh(vertices, faces, textures, **kwargs)
