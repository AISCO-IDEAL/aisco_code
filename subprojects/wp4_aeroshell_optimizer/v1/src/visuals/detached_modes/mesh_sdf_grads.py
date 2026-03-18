import torch

from subprojects.wp4_aeroshell_optimizer.v1.logging_lib.image_uploads import upload_neural_renderer_gif
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base_class import RenderingMode
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser import Visualiser


class MeshSDFGrads(RenderingMode):
    def __init__(self, visualiser: Visualiser):
        config = load_config(v="v1", cfg_name="neural_renderer_v0.yaml")["detached_modes"]["MeshSDF_grads"]["config"]
        super().__init__(visualiser, config)

    def update(self, optimization_state: dict) -> None:
        """
        Render the mesh with per-face colors based on the dot product between vertex normals and SDF gradients.

        Inputs:
            optimization_state: dict containing "vertices", "faces", and "vertice_grads"
            vertices: (V, 3) tensor of vertex positions
            faces: (F, 3) tensor of vertex indices for each face
            vertice_grads: (V, 3) tensor of gradients of the loss w.r.t. vertex positions, ideally extracted from the MeshSDF loss component
        """

        vertices = optimization_state["vertices"].detach()
        faces = optimization_state["faces"].detach()
        vertice_grads = optimization_state["vertice_grads"].detach()

        if faces.numel() == 0:
            print(f"  {self.__class__.__name__}: No faces to render, skipping frame")
            return

        # compute unit normals for each VERTEX by averaging face normals of adjacent faces
        face_normals = torch.cross(vertices[faces][:, 1] - vertices[faces][:, 0], vertices[faces][:, 2] - vertices[faces][:, 0])
        face_normals = face_normals / (face_normals.norm(dim=-1, keepdim=True) + 1e-8)
        vertex_normals = torch.zeros_like(vertices)
        for i in range(faces.shape[0]):
            face = faces[i]
            normal = face_normals[i]
            vertex_normals[face] += normal

        # ensure normals are unit vectors
        vertex_normals = vertex_normals / (vertex_normals.norm(dim=-1, keepdim=True) + 1e-8)

        # take dot products between unit normals and non-unit vertice_grads to get per-face scalar values for visualisation
        dot_products = torch.einsum("ij,ij->i", vertex_normals, vertice_grads)

        # Make color range go from a standard dark grey (zero, in the middle of the range) to light yellow (positive, max value) and dark blue (negative, min value)

        base_grey = torch.tensor([0.5, 0.5, 0.5], device=faces.device)
        max_color = torch.tensor([1.0, 1.0, 0.0], device=faces.device)  # light yellow
        min_color = torch.tensor([0.0, 0.0, 0.5], device=faces.device)  # dark blue
        colors = base_grey.unsqueeze(0) + (dot_products / (dot_products.abs().max() + 1e-8)).unsqueeze(-1) * (
            max_color - min_color
        ).unsqueeze(0)
        self.visualiser.vertex_colors_to_textures(vertex_colors=colors, faces=faces)

        image = self._render(
            vertices=vertices,
            faces=faces,
            textures=self.visualiser.vertex_colors_to_textures(vertex_colors=colors, faces=faces),
            mode="rgb",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )
        # Store the rendered image (image[0] if batch dimension exists)
        if hasattr(image, "__len__") and len(image) > 0:
            self.frame_storage.append(image[0])
        else:
            self.frame_storage.append(image)

    def finalize(self, output_dir, file_name="MeshSDF_grads.gif", delete=True):
        """
        Save all rendered frames as GIF and sample PNGs.

        Args:
            output_dir: Directory for saving results (mode-specific subdirectory)
            file_name: Name of the output GIF file (default: "rendering.gif")
            delete: If True, deletes stored frames after saving (default: False)
        """
        if not self.frame_storage:
            print(f"  {self.__class__.__name__}: No frames to save")
            return

        # Create GIF animation
        gif_path = output_dir / file_name
        self._frames_to_gif(gif_path, fps=self.fps)
        print(f"    Saved GIF: {gif_path}")

        # log GIF to wandb
        try:
            upload_neural_renderer_gif(gif_path, key=f"animations/{self.__class__.__name__}_{file_name}")
        except Exception as e:
            print(f"    Warning: Failed to upload {file_name} to wandb: {e}")
        # Save first frame as reference
        first_frame_path = output_dir / f"{file_name}_first_frame.png"
        self._save_image(self.frame_storage[0], first_frame_path)

        # Save last frame as reference
        last_frame_path = output_dir / f"{file_name}_final_frame.png"
        self._save_image(self.frame_storage[-1], last_frame_path)
        # clear frame storage to free memory
        if delete:
            self.clear_frames()
        print(f"    Saved sample frames to {output_dir}")
