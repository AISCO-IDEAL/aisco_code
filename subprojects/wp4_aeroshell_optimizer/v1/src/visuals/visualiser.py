from pathlib import Path

import imageio
import numpy as np
import torch

from subprojects.fixed_renderer.renderer import Renderer as NeuralRenderer


class Visualiser:
    """
    Minimal wrapper around `fixed_renderer` for rendering and frame export.
    Loads of manual work still required to get everything working smoothly.
    """

    def __init__(
        self,
        image_size: int = 256,
        background_color: list[float] | None = None,
        anti_aliasing: bool = True,
        camera_mode: str = "look_at",
        viewing_angle: float = 30.0,
        device: str = "cuda",
        up: list[float] | None = None,
    ) -> None:
        self.image_size = image_size
        self.device = device
        self.camera_mode = camera_mode
        self.stored_frames: list[np.ndarray] = []

        bg = background_color if background_color is not None else [0.0, 0.0, 0.0]
        up_vec = up if up is not None else [0.0, 0.0, 1.0]  # Default to Z-up
        self.renderer = NeuralRenderer(
            image_size=image_size,
            anti_aliasing=anti_aliasing,
            background_color=bg,
            camera_mode=camera_mode,
            viewing_angle=viewing_angle,
            orig_size=image_size,
            up=up_vec,
        )
        # Default clipping planes
        self.renderer.near = 0.1
        self.renderer.far = 50_000.0


    def render(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        textures: torch.Tensor | None = None,
        mode: str = "rgb",
        save_frame: bool = False,
        camera_position: list[float] | None = None,
        camera_direction: list[float] | None = None,
        k_mat: torch.Tensor | None = None,
        r_mat: torch.Tensor | None = None,
        t_vec: torch.Tensor | None = None,
        dist_coeffs: torch.Tensor | None = None,
        orig_size: int | None = None,
    ) -> np.ndarray:
        """Render a mesh in the requested mode and optionally store the frame.

        Modes: "rgb", "silhouettes", "depth", or None (full render tuple from renderer.render).
        Camera parameters can be overridden per-call. If no textures are provided for RGB,
        a light gray texture is generated.
        """
        vertices, faces, textures = self._ensure_batched(vertices, faces, textures)
        vertices = self._to_device(vertices)
        faces = self._to_device(faces)
        if textures is not None:
            textures = self._to_device(textures)

        if camera_position is not None:
            self.renderer.eye = camera_position
        if camera_direction is not None:
            self.renderer.camera_direction = camera_direction

        if textures is None and mode == "rgb":
            textures = self.create_solid_texture(
                batch_size=faces.shape[0],
                num_faces=faces.shape[1],
                color=(0.7, 0.7, 0.7),
            )

        with torch.no_grad():
            if mode == "rgb":
                output = self.renderer.render_rgb(vertices, faces, textures, K=k_mat, R=r_mat, t=t_vec, dist_coeffs=dist_coeffs, orig_size=orig_size)
            elif mode == "silhouettes":
                output = self.renderer.render_silhouettes(vertices, faces, K=k_mat, R=r_mat, t=t_vec, dist_coeffs=dist_coeffs, orig_size=orig_size)
            elif mode == "depth":
                output = self.renderer.render_depth(vertices, faces, K=k_mat, R=r_mat, t=t_vec, dist_coeffs=dist_coeffs, orig_size=orig_size)
            else:
                output = self.renderer.render(vertices, faces, textures, K=k_mat, R=r_mat, t=t_vec, dist_coeffs=dist_coeffs, orig_size=orig_size)

        np_out = self._to_numpy(output)
        if save_frame:
            # store first frame in batch
            self.stored_frames.append(np_out[0])
        return np_out

    def _ensure_batched(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        textures: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Ensure inputs include a batch dimension.

        The renderer expects shapes (B, V, 3) for vertices and (B, F, 3) for faces.
        If inputs are unbatched (V, 3) or (F, 3), this method adds a leading batch dim.
        """

        if vertices.dim() == 2:
            vertices = vertices.unsqueeze(0)
        if faces.dim() == 2:
            faces = faces.unsqueeze(0)

        # Basic validation on faces shape
        if faces.dim() != 3 or faces.shape[-1] != 3:
            raise ValueError("Faces must have shape (B, F, 3) after batching")

        if textures is not None:
            # textures expected shape: (B, F, ts, ts, ts, 3)
            if textures.dim() == 5:
                textures = textures.unsqueeze(0)
            elif textures.dim() != 6:
                raise ValueError("Textures must have shape (B, F, ts, ts, ts, 3)")

        return vertices, faces, textures

    def clear_frames(self) -> None:
        """Clear any stored frames."""
        self.stored_frames.clear()

    def frames_to_gif(
        self,
        file_path: str | Path | None = None,
        fps: int = 15,
        return_uint8: bool = False,
    ) -> np.ndarray:
        """Convert stored frames to uint8, optionally save a GIF, and return stacked frames.

        Args:
            file_path: Optional path to save GIF file
            fps: Frames per second for the GIF
            return_uint8: If True, return uint8 [0,255]; if False, return float32 [0,1] for wandb

        Returns an array of shape (num_frames, H, W, C).
        """
        if not self.stored_frames:
            raise ValueError("No stored frames to create GIF.")

        print(f"Converting stored frames {len(self.stored_frames)} to GIF format...")
        print(f"First frame shape: {self.stored_frames[0].shape}, dtype: {self.stored_frames[0].dtype}")
        print(f"First frame min/max: {self.stored_frames[0].min():.4f} / {self.stored_frames[0].max():.4f}")

        # Stack frames first
        frames_stacked = np.stack(self.stored_frames, axis=0)
        print(f"Stacked frames shape: {frames_stacked.shape}, dtype: {frames_stacked.dtype}")

        # For local GIF saving, convert to uint8
        if file_path is not None:
            frames_uint8 = []
            for frame in self.stored_frames:
                fr = (frame * 255).astype(np.uint8)
                # If single-channel shape (H, W, 1), squeeze to (H, W) for Pillow
                if fr.ndim == 3 and fr.shape[-1] == 1:
                    fr = fr.squeeze(-1)
                frames_uint8.append(fr)
            imageio.mimsave(file_path, frames_uint8, fps=fps)

        # Return format depends on use case
        if return_uint8:
            frames_uint8 = []
            for frame in self.stored_frames:
                fr = (frame * 255).astype(np.uint8)
                if fr.ndim == 3 and fr.shape[-1] == 1:
                    fr = fr.squeeze(-1)
                frames_uint8.append(fr)
            return np.stack(frames_uint8, axis=0)
        else:
            # wandb.Video expects float32 in [0, 1] range
            # Ensure frames are float32
            return frames_stacked.astype(np.float32)

    def create_solid_texture(
        self,
        batch_size: int,
        num_faces: int,
        color: tuple[float, float, float] = (0.7, 0.7, 0.7),
        texture_size: int = 2,
    ) -> torch.Tensor:
        """Create a solid-color texture tensor on the configured device."""
        textures = torch.ones(
            batch_size,
            num_faces,
            texture_size,
            texture_size,
            texture_size,
            3,
            dtype=torch.float32,
            device=torch.device(self.device),
        )
        textures = textures * torch.tensor(color, dtype=torch.float32, device=textures.device)
        return textures

    def vertex_colors_to_textures(
        self,
        vertex_colors: torch.Tensor,
        faces: torch.Tensor,
        texture_size: int = 2,
    ) -> torch.Tensor:
        """Convert per-vertex colors to per-face textures."""
        batch_size, num_faces, _ = faces.shape
        face_colors = torch.zeros(batch_size, num_faces, 3, 3, device=vertex_colors.device)
        for i in range(3):
            face_colors[:, :, i, :] = torch.gather(
                vertex_colors,
                1,
                faces[:, :, i:i+1].expand(-1, -1, 3),
            )

        mean_color = face_colors.mean(dim=2)
        textures = mean_color.view(batch_size, num_faces, 1, 1, 1, 3).expand(
            -1, -1, texture_size, texture_size, texture_size, -1
        ).contiguous()
        return textures


    def save_image(self, image: np.ndarray, file_path: str) -> None:
        """Save a single image to disk."""
        img = (image * 255).astype(np.uint8)
        # If grayscale with explicit channel, squeeze to (H, W)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        imageio.imwrite(file_path, img)
        print(f"Saved image to {file_path}")
    # ========== Helper Methods ==========

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the correct device."""
        if tensor is None:
            return None
        target_device = torch.device(self.device)
        if tensor.device != target_device:
            return tensor.to(target_device)
        return tensor

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array and transpose to (batch, H, W, C) format."""
        if tensor.is_cuda:
            tensor = tensor.cpu()
        arr = tensor.detach().numpy()
        # If shape is (batch, C, H, W), transpose to (batch, H, W, C)
        if len(arr.shape) == 4 and arr.shape[1] in [1, 3, 4]:
            arr = arr.transpose(0, 2, 3, 1)
        # If shape is (batch, H, W) [e.g., silhouettes], add channel dimension
        elif len(arr.shape) == 3:
            arr = arr[:, :, :, np.newaxis]  # (batch, H, W, 1)
        return arr

    def _create_default_texture(self, batch_size: int, num_faces: int, texture_size: int = 2) -> torch.Tensor:
        """Create a default white texture on the correct device."""
        textures = torch.ones(
            batch_size, num_faces, texture_size, texture_size, texture_size, 3,
            dtype=torch.float32,
            device=torch.device(self.device)
        )
        return textures

    def __call__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        textures: torch.Tensor | None = None,
        **kwargs
    ) -> np.ndarray:
        """
        Convenience method to render a mesh.
        Equivalent to calling render().
        """
        return self.render(vertices, faces, textures, **kwargs)
