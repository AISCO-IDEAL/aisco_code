from pathlib import Path

import torch

from subprojects.wp3_solar_estimator.geo_mesh_utils.mesh_properties import compute_face_normals, compute_triangle_areas
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base_class import RenderingMode


class SolarSelectionMode(RenderingMode):
    """
    Renders which faces were selected for solar performance, black/white difference, no softmax



    Configuration:
        - camera_position: [x, y, z] (default: [4000, 0, 0])
        - look_at: [x, y, z] (default: [0, 0, 0])
        - fps: GIF frame rate (default: 15)
        - render_interval: Render every N iterations (default: 10)
        - make_gif: false saves seperate image every interval iterations
        - fps: if make_gif is true, this sets the fps of the resulting gif, if false this is ignored
        - solar_color:  # Color for faces selected for solar performance
        - rest_color:  # Color for faces not selected for solar performance
    """

    def __init__(self, visualiser, config: dict):
        """Initialize with custom parameter."""
        super().__init__(visualiser, config)
        # Rendering parameters
        self.render_interval = config.get("render_interval", 1)
        self.make_gif = config.get("make_gif", True)
        self.fps = config.get("fps", 15)
        # Solar specific parameters
        self.solar_color = config["solar_color"]
        self.rest_color = config["rest_color"]
        solar_config = load_config(v="v1", cfg_name="solar_configs/solar_loss_config.yaml")["SolarLossConfigV3"]
        self.area_budget = solar_config["area_budget_mm2"]
        self.solar_angle = torch.tensor(solar_config["sun_dir"], device=self.visualiser.device)
        self.solar_angle = (self.solar_angle / torch.norm(self.solar_angle)).detach()  # Normalize; detach to drop grad_fn
        if self.solar_angle is None or self.solar_angle.shape != (3,):
            raise ValueError(
                "Solar angle must be provided in the config and be a 3D vector --> shape = (3,), given in form of [A,B,C]."
            )

    def update(self, optimization_state: dict) -> None:
        """
        Args:
            optimization_state: Dictionary containing:
                - 'vertices': Current mesh vertices (required)
                - 'faces': Current mesh faces (required)
                - 'iteration': Current iteration number (optional)
                - 'loss': Current loss value (optional)
                - 'model': The current neural network model (optional)
                - Any other custom keys
        """

        iteration = optimization_state.get("iteration", 0)
        # Only render at specified intervals
        if iteration % self.render_interval != 0:
            return

        vertices = optimization_state.get("vertices")
        faces = optimization_state.get("faces")

        if vertices is None or faces is None:
            raise Warning("Warning: No vertices/faces in optimization_state, skipping update")

        # Whole selection logic here:

        face_normals = compute_face_normals(vertices, faces)
        face_areas = compute_triangle_areas(vertices, faces)

        # Calculate dot products (cosine of angle with sun vector)
        solar_dot_prods = (face_normals * self.solar_angle).sum(dim=1)

        # Calculate score (energy) and cost (area)
        # We maximize total energy (score) subject to area (cost) <= budget
        score = solar_dot_prods * face_areas
        cost = face_areas
        budget = self.area_budget
        # Ratio for greedy selection (value density) = energy / area = dot_product
        ratio = solar_dot_prods
        # Sort by ratio descending
        order = torch.argsort(ratio, descending=True)
        sorted_cost = cost[order]
        # sorted_score = score[order] # Not strictly needed for logic but good for debugging if needed  # noqa: ERA001

        # Cumsum can be nondeterministic on GPU, but we need it for the selection logic, so we disable deterministic algorithms here
        with torch.no_grad():
            old_deterministic = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(False)
            cum_cost = torch.cumsum(sorted_cost, dim=0)
            torch.use_deterministic_algorithms(old_deterministic)

        # fully taken items
        full = cum_cost <= budget
        mask = torch.zeros_like(score, dtype=torch.float32)
        # assign full items
        if full.any():
            mask[order[full]] = 1.0

        # Create textures based on mask, using solar_color for selected faces and rest_color for others
        colors = torch.where(
            mask.unsqueeze(1) > 0.5,
            torch.tensor(self.solar_color, device=self.visualiser.device),
            torch.tensor(self.rest_color, device=self.visualiser.device),
        )
        textures = self.visualiser.face_colors_to_textures(face_colors=colors)

        # Render from fixed camera position
        image = self.visualiser.render(
            vertices, faces, textures, mode="rgb", camera_position=self.camera_position, camera_direction=self.camera_direction
        )

        # Store the frame (handle batch dimension if present)
        if hasattr(image, "__len__") and len(image) > 0:
            self.frame_storage.append(image[0])
        else:
            self.frame_storage.append(image)

    def finalize(self, output_dir: Path, file_name: str = "solar_selection.gif", delete: bool = False) -> None:
        """
        Save as gif or individual frames.
        """
        if not self.frame_storage:
            print(f"  {self.__class__.__name__}: No frames to save")
            return
        if self.make_gif:
            # Use base class to create GIF
            super().finalize(output_dir, file_name=file_name, delete=delete)

        else:
            # Save individual frames
            for idx, frame in enumerate(self.frame_storage):
                frame_path = output_dir / f"frame_{idx:04d}.png"
                self._save_image(frame, frame_path)
            print(f"  Saved {len(self.frame_storage)} individual frames to {output_dir}")

            if delete:
                self.clear_frames()
                print("  Deleted stored frames after saving individual images.")
