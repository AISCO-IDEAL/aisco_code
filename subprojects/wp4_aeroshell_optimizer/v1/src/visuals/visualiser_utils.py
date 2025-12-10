import torch

from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.kaolin_conversions import get_diff_mesh
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser import Visualiser

"""
This file is meant to contain any visualizations you'd like to implement using the Visualiser class.
That means, any storage logic or other logic that is not directly related to the rendering functionality
should be implemented here.

If you change functions of others, ensure that there are no other use cases or dependencies broken by your changes.
"""


def initialize_visualiser(
    image_size: int = 256,
    background_color: list[float] | None = None,
    anti_aliasing: bool = True,
    camera_mode: str = 'look_at',
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
    position: list[int] | None = None,
    look_at: list[int] | None = None,
    save_frame: bool = False,
    clear_existing: bool = False,
) -> torch.Tensor:
    """
    Renders an image of the given model using the provided visualiser.

    Args:
        model (BaseModel): The model to be rendered.
        visualiser (Visualiser): The visualiser to use for rendering.

    Returns:
        torch.Tensor: The rendered image as a tensor.
    """
    if position is None:
        position = [4000, 0, 0]
    if look_at is None:
        look_at = [0, 0, 0]

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

        looking_direction = [look_at[i] - position[i] for i in range(3)]
        visualiser.render(
            vertices,
            faces,
            camera_position=position,
            camera_direction=looking_direction,
            save_frame=True,
        )

        return visualiser.stored_frames[-1]

    # Single render, no storage
    looking_direction = [look_at[i] - position[i] for i in range(3)]
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
    visualiser.save_image(image, file_path)
