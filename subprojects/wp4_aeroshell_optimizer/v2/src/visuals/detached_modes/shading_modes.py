"""Detached rendering modes for shading visualization."""

from pathlib import Path

import torch

from subprojects.wp4_aeroshell_optimizer.v2.logging_lib.image_uploads import upload_neural_renderer_gif
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.rendering_mode_base_class import RenderingMode


# ignore the ruff ARG002 error for this file
# ruff: noqa: ARG002
class SolarBase(RenderingMode):
    """Base class for per-face shading scalar rendering."""

    _config_key = ""
    _value_key = ""
    _wandb_key = ""

    def __init__(self, visualiser, config: dict | None = None):
        if config is None:
            renderer_cfg = load_config(v="v1", cfg_name="neural_renderer_v0.yaml")
            config = renderer_cfg.get("solar_info_modes", {}).get(self._config_key, {}).get("config")
            if config is None:
                config = renderer_cfg["detached_modes"][self._config_key]["config"]
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
            upload_neural_renderer_gif(gif_path, key=f"animations/{self.__class__.__name__}", step=0)
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


class ShadingVisibility(SolarBase):
    """Render per-face visibility values."""

    _config_key = "shading_visibility"
    _value_key = "shading_visibility"
    _wandb_key = "ShadingVisibility"

    def _colors_from_weights(self, normalized_weights: torch.Tensor, device: torch.device) -> torch.Tensor:
        min_color = torch.tensor([0.0, 0.0, 0.3], dtype=torch.float32, device=device)
        max_color = torch.tensor([0.9, 0.95, 1.0], dtype=torch.float32, device=device)
        return min_color.unsqueeze(0) + normalized_weights.unsqueeze(-1) * (max_color - min_color).unsqueeze(0)

    def finalize(self, output_dir: Path, file_name: str = "shading_visibility.gif", delete: bool = True) -> None:
        super().finalize(output_dir, file_name, delete)


class ShadingPenalty(SolarBase):
    """Render per-face shading penalty values."""

    _config_key = "shading_penalty"
    _value_key = "shading_penalty_intensity"
    _wandb_key = "ShadingPenalty"

    def _colors_from_weights(self, normalized_weights: torch.Tensor, device: torch.device) -> torch.Tensor:
        min_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        max_color = torch.tensor([0.95, 0.0, 0.0], dtype=torch.float32, device=device)
        return min_color.unsqueeze(0) + normalized_weights.unsqueeze(-1) * (max_color - min_color).unsqueeze(0)

    def finalize(self, output_dir: Path, file_name: str = "shading_penalty.gif", delete: bool = True) -> None:
        super().finalize(output_dir, file_name, delete)


class ShadingFactor(SolarBase):
    _config_key = "shading_factor"
    _value_key = "shading_factor"
    _wandb_key = "ShadingFactor"

    def _colors_from_weights(self, normalized_weights: torch.Tensor, device: torch.device) -> torch.Tensor:
        # Reversed inferno-style anchors: low values are bright yellow, high values are dark purple.
        palette = torch.tensor(
            [
                [0.988, 0.998, 0.645],
                [0.987, 0.554, 0.147],
                [0.736, 0.216, 0.330],
                [0.341, 0.063, 0.429],
                [0.078, 0.042, 0.206],
            ],
            dtype=torch.float32,
            device=device,
        )

        weights = normalized_weights.clamp(0.0, 1.0)
        scaled = weights * (palette.shape[0] - 1)
        lower_idx = torch.floor(scaled).to(torch.long)
        upper_idx = torch.clamp(lower_idx + 1, max=palette.shape[0] - 1)
        blend = (scaled - lower_idx.to(scaled.dtype)).unsqueeze(-1)

        lower_colors = palette[lower_idx]
        upper_colors = palette[upper_idx]
        return lower_colors + blend * (upper_colors - lower_colors)

    def finalize(self, output_dir: Path, file_name: str = "shading_factor.gif", delete: bool = True) -> None:
        super().finalize(output_dir, file_name, delete)


class AverageQuality(SolarBase):
    """Render per-face average quality values."""

    _config_key = "average_quality"
    _value_key = "qualities"
    _wandb_key = "AverageQuality"

    def _normalize_weights(
        self,
        weights: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        optimization_state: dict,
    ) -> torch.Tensor:
        # Use this so the qualities are scaled from a [-1,1] to a [0,1] range always
        # This ensures the actual quality values are rendered, makes differences from normal/softmax calculation visible
        return (weights + 1) / 2

    def _colors_from_weights(self, normalized_weights: torch.Tensor, device: torch.device) -> torch.Tensor:
        min_color = torch.tensor([0.9, 0.0, 0.25], dtype=torch.float32, device=device)
        max_color = torch.tensor([0.0, 0.9, 0.25], dtype=torch.float32, device=device)
        return min_color.unsqueeze(0) + normalized_weights.unsqueeze(-1) * (max_color - min_color).unsqueeze(0)

    def finalize(self, output_dir: Path, file_name: str = "average_quality.gif", delete: bool = True) -> None:
        super().finalize(output_dir, file_name, delete)


class OverallLosses(SolarBase):
    """Render per-face overall loss values."""

    _config_key = "overall_losses"
    _value_key = "losses"
    _wandb_key = "OverallLosses"

    def _colors_from_weights(self, normalized_weights: torch.Tensor, device: torch.device) -> torch.Tensor:
        palette = torch.tensor(
            [
                [0.078, 0.042, 0.206],
                [0.341, 0.063, 0.429],
                [0.736, 0.216, 0.330],
                [0.987, 0.554, 0.147],
                [0.988, 0.998, 0.645],
            ],
            dtype=torch.float32,
            device=device,
        )

        weights = normalized_weights.clamp(0.0, 1.0)
        scaled = weights * (palette.shape[0] - 1)
        lower_idx = torch.floor(scaled).to(torch.long)
        upper_idx = torch.clamp(lower_idx + 1, max=palette.shape[0] - 1)
        blend = (scaled - lower_idx.to(scaled.dtype)).unsqueeze(-1)

        lower_colors = palette[lower_idx]
        upper_colors = palette[upper_idx]
        return lower_colors + blend * (upper_colors - lower_colors)

    def finalize(self, output_dir: Path, file_name: str = "overall_losses.gif", delete: bool = True) -> None:
        super().finalize(output_dir, file_name, delete)
