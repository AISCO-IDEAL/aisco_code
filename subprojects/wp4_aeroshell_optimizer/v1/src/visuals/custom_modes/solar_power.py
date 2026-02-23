
from pathlib import Path

import torch

from subprojects.wp3_solar_estimator.geo_mesh_utils.mesh_properties import compute_face_normals
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base_class import RenderingMode


class SolarPower(RenderingMode):
    """
    Renders the solar power of each face as a color, with a fixed camera position and automatic GIF/Image creation.



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
        #Rendering parameters
        self.render_interval = config.get("render_interval", 1)
        self.make_gif = config.get("make_gif", True)
        self.fps = config.get("fps", 15)
        #Solar specific parameters

        solar_config = load_config(v="v1", cfg_name="solar_loss_config.yaml")["general_config"]
        self.solar_angle=torch.tensor(solar_config["solar_angle"],device=self.visualiser.device)
        self.solar_angle = self.solar_angle / torch.norm(self.solar_angle) #Normalize to be safe
        if self.solar_angle is None or self.solar_angle.shape != (3,):
            raise ValueError("Solar angle must be provided in the config and be a 3D vector --> shape = (3,), given in form of [A,B,C].")
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


        #Whole selection logic here:

        face_normals = compute_face_normals(vertices, faces)
        power_dotprods = (face_normals * self.solar_angle).sum(dim=1)




        # Normalize power_dotprods to [0,1] range
        min_val = power_dotprods.min()
        max_val = power_dotprods.max()
        if max_val > min_val:
            normalized_power = (power_dotprods - min_val) / (max_val - min_val)
        else:
            normalized_power = torch.zeros_like(power_dotprods)

        # Create colors based on normalized power values
        colors = torch.zeros((faces.shape[0], 3), device=self.visualiser.device)
        colors[:, 0] = 1 - normalized_power  # Red channel
        colors[:, 1] = normalized_power  # Green channel
        colors[:, 2] = 0.5 * (1 - normalized_power)  # Blue channel

        textures = self.visualiser.face_colors_to_textures(face_colors=colors)

        # Render from fixed camera position
        image = self.visualiser.render(vertices,
                                        faces,
                                        textures,
                                        mode="rgb",
                                        camera_position=self.camera_position,
                                        camera_direction=self.camera_direction)




        # Store the frame (handle batch dimension if present)
        if hasattr(image, "__len__") and len(image) > 0:
            self.frame_storage.append(image[0])
        else:
            self.frame_storage.append(image)

    def finalize(self, output_dir: Path, file_name: str = "solar_power.gif", delete: bool = False) -> None:
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
