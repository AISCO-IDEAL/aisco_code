"""
This file is meant to contain any visualizations you'd like to implement using the Visualiser class.
That means, any storage logic or other logic that is not directly related to the rendering functionality
should be implemented here.

If you change functions of others, ensure that there are no other use cases or dependencies broken by your changes.
"""

from typing import Any

import cadquery as cq
import numpy as np
import torch
from numpy.typing import NDArray

from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.kaolin_conversions import get_diff_mesh
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.visualiser import Visualiser


def initialize_visualiser(
    image_size: int = 256,
    background_color: list[float] | None = None,
    anti_aliasing: bool = True,
    camera_mode: str = "look_at",
    viewing_angle: float = 30.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Visualiser:
    """
    Initializes and returns a Visualiser object with predefined parameters.

    Returns:
        Visualiser: An instance of the Visualiser class.
    """

    visualiser = Visualiser(
        image_size=image_size,
        background_color=background_color or [0, 0, 0],
        anti_aliasing=anti_aliasing,
        camera_mode=camera_mode,
        viewing_angle=viewing_angle,
        device=device,
    )
    return visualiser


def render_image(
    visualiser: Visualiser,
    model: BaseModel | None = None,
    verts: torch.Tensor | None = None,
    faces_in: torch.Tensor | None = None,
    position: list[float] | None = None,
    look_at: list[float] | None = None,
    save_frame: bool = False,
    clear_existing: bool = False,
) -> NDArray[Any]:  # kdiehl: consistency with actual returned type
    """
    Renders an image of the given model using the provided visualiser.

    Args:
        model (BaseModel): The model to be rendered.
        visualiser (Visualiser): The visualiser to use for rendering.

    Returns:
        ndarray: The rendered image as a numpy array.
    """
    if position is None:
        position = [4000.0, 0.0, 0.0]  # kdihel: visualiser expects floats
    if look_at is None:
        look_at = [0.0, 0.0, 0.0]

    if verts is None or faces_in is None:
        if model is None:
            raise ValueError("Either a model or vertices and faces must be provided for rendering.")
        vertices, faces = get_diff_mesh(model, grid_res=64, device=visualiser.device)
    else:
        vertices = verts
        faces = faces_in

    if save_frame:
        if clear_existing:
            visualiser.stored_frames.clear()

        looking_direction = [float(look_at[i] - position[i]) for i in range(3)]
        visualiser.render(
            vertices,
            faces,
            camera_position=position,
            camera_direction=looking_direction,
            save_frame=True,
        )

        return visualiser.stored_frames[-1]

    # Single render, no storage
    looking_direction = [float(look_at[i] - position[i]) for i in range(3)]
    image = visualiser.render(
        vertices,
        faces,
        camera_position=position,
        camera_direction=looking_direction,
        save_frame=False,
    )
    return image


def save_rendered_image(visualiser: Visualiser, image: torch.Tensor, file_path: str) -> None:
    """
    Saves the rendered image to the specified file path.

    Args:
        image (torch.Tensor): The rendered image tensor.
        file_path (str): The path where the image will be saved.
    """
    # Convert the image tensor to a numpy array if it's a torch.Tensor
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
    visualiser.save_image(image_np, file_path)


def assembly_to_kaolin_mesh(
    assembly: cq.Assembly, tolerance: float = 0.1, angular_tolerance: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a CadQuery Assembly to Kaolin-compatible mesh format.

    This function takes a CadQuery Assembly (containing one or more solids),
    converts it to a compound shape, tessellates it into triangles, and
    returns the result as PyTorch tensors compatible with the Kaolin library.

    Args:
        assembly: CadQuery Assembly object containing solids
        tolerance: Tessellation deflection tolerance in model units (default: 0.1).
                  Smaller values produce more detailed meshes.
        angular_tolerance: Angular tolerance in radians for tessellation (default: 0.1).
                          Smaller values produce smoother curved surfaces.

    Returns:
        tuple: (vertices, faces) where:
            - vertices: torch.Tensor of shape (N, 3) with vertex coordinates (float32)
            - faces: torch.Tensor of shape (F, 3) with face indices (int64)

    Example:
        >>> assembly = cq.Assembly()
        >>> # ... build your assembly ...
        >>> vertices, faces = assembly_to_kaolin_mesh(assembly, tolerance=0.05)
        >>> print(vertices.shape)  # (N, 3)
        >>> print(faces.shape)     # (F, 3)
    """
    # Convert assembly to a compound shape (combines all solids)
    compound = assembly

    # Tessellate the compound into triangles
    # Returns: (List[Vector], List[Tuple[int, int, int]]) # noqa: ERA001
    vertex_list, face_list = compound.tessellate(tolerance, angular_tolerance)

    # Convert vertices from CadQuery Vector objects to numpy array
    vertices_np = np.array([[v.x, v.y, v.z] for v in vertex_list], dtype=np.float32)

    # Convert face indices to numpy array
    faces_np = np.array(face_list, dtype=np.int64)

    # Convert to PyTorch tensors (Kaolin format)
    vertices = torch.from_numpy(vertices_np)
    faces = torch.from_numpy(faces_np)

    return vertices, faces
