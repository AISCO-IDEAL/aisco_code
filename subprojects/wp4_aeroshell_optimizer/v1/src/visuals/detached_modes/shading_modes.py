"""Detached rendering modes for shading visualization."""

from pathlib import Path

import torch

from subprojects.wp4_aeroshell_optimizer.v1.logging_lib.image_uploads import upload_neural_renderer_gif
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base_class import RenderingMode


# ignore the ruff ARG002 error for this file
# ruff: noqa: ARG002
class _ShadingBase(RenderingMode):
    """Base class for per-face shading scalar rendering."""

    _config_key = ""
    _value_key = ""
    _wandb_key = ""

    def __init__(self, visualiser):
        config = load_config(v="v1", cfg_name="neural_renderer_v0.yaml")["detached_modes"][self._config_key]["config"]
        super().__init__(visualiser, config)
        self.camera_position = config["camera_position"]
        self.look_at = config["look_at"]
        self.fps = config["fps"]

    def update(self, optimization_state: dict) -> None:
        vertices = optimization_state["vertices"].detach()
        faces = optimization_state["faces"].detach()
        values = optimization_state[self._value_key].detach()

        weights = values.to(dtype=torch.float32)
        normalized_weights = self._normalize_weights(weights, vertices, faces, optimization_state)

        colors = self._colors_from_weights(normalized_weights, device=faces.device)
        textures = self.visualiser.face_colors_to_textures(colors)

        image = self._render(
            vertices=vertices,
            faces=faces,
            textures=textures,
            mode="rgb",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )

        if hasattr(image, "__len__") and len(image) > 0:
            self.frame_storage.append(image[0])
        else:
            self.frame_storage.append(image)

    def finalize(self, output_dir: Path, file_name: str = "rendering.gif", delete: bool = True) -> None:
        if not self.frame_storage:
            print(f"  {self.__class__.__name__}: No frames to save")
            return

        gif_path = output_dir / file_name
        self._frames_to_gif(gif_path, fps=self.fps)
        print(f"    Saved GIF: {gif_path}")

        try:
            upload_neural_renderer_gif(gif_path, key=f"animations/{self._wandb_key}", step=1)
        except Exception as e:
            print(f"    Warning: Failed to upload shading GIF to wandb: {e}")

        first_frame_path = output_dir / "first_frame.png"
        self._save_image(self.frame_storage[0], first_frame_path)

        last_frame_path = output_dir / "final_frame.png"
        self._save_image(self.frame_storage[-1], last_frame_path)

        if delete:
            self.clear_frames()
        print(f"    Saved sample frames to {output_dir}")

    def _colors_from_weights(self, normalized_weights: torch.Tensor, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    def _normalize_weights(
        self,
        weights: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        optimization_state: dict,
    ) -> torch.Tensor:
        min_weight = weights.min()
        max_weight = weights.max()
        weight_range = (max_weight - min_weight).clamp_min(1e-8)
        return (weights - min_weight) / weight_range


class ShadingVisibility(_ShadingBase):
    """Render per-face visibility values."""

    _config_key = "shading_visibility"
    _value_key = "shading_visibility"
    _wandb_key = "ShadingVisibility"

    def _normalize_weights(
        self,
        weights: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        optimization_state: dict,
    ) -> torch.Tensor:
        return weights.clamp(0.0, 1.0)

    def _colors_from_weights(self, normalized_weights: torch.Tensor, device: torch.device) -> torch.Tensor:
        min_color = torch.tensor([0.0, 0.0, 0.25], dtype=torch.float32, device=device)
        max_color = torch.tensor([0.9, 0.95, 1.0], dtype=torch.float32, device=device)
        return min_color.unsqueeze(0) + normalized_weights.unsqueeze(-1) * (max_color - min_color).unsqueeze(0)


class ShadingPenalty(_ShadingBase):
    """Render per-face shading penalty values."""

    _config_key = "shading_penalty"
    _value_key = "shading_penalty"
    _wandb_key = "ShadingPenalty"

    def _normalize_weights(
        self,
        weights: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        optimization_state: dict,
    ) -> torch.Tensor:
        intensity = optimization_state.get("shading_penalty_intensity", weights)
        return intensity.to(dtype=torch.float32).clamp(0.0, 1.0)

    def _colors_from_weights(self, normalized_weights: torch.Tensor, device: torch.device) -> torch.Tensor:
        min_color = torch.tensor([0.15, 0.0, 0.0], dtype=torch.float32, device=device)
        max_color = torch.tensor([1.0, 0.9, 0.2], dtype=torch.float32, device=device)
        return min_color.unsqueeze(0) + normalized_weights.unsqueeze(-1) * (max_color - min_color).unsqueeze(0)
