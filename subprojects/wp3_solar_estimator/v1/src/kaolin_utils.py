import torch
from kaolin.ops.conversions import (
    FlexiCubes,
)

from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel


def get_diff_mesh(
    model: BaseModel,
    grid_res: list[int] | int = 128,
    box_dim: list = [-5000, 5000, -5000, 5000, -5000, 5000],  # noqa: B006
    device: str = "cuda",
):
    """
    Pre: Gets an arbitrary SDF Model together with parameters that influence the mesh estraction process
    Post: Outputs vertices and faces that fully describe the sdf, but now converted into a mesh.

    Args:
        model: Arbitrary SDF Network representing our shape.
        grid_res: The resolution that Flexicubes uses, can also be a list of ints corresponding to the [x,y,z] resolutions
        box_dim: Defines the box in which the SDF should be sampled, for better resolution per memory decrease size of box
        device: Run torch code over cuda/cpu
    Returns:
        Vertices and Faces as torch.Tensor
    """
    ####################### eschaerer #################
    # Enforce deterministic behavior for PyTorch operations (needed for backpropagation tests)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)  # Set a fixed seed
    ###################################################

    # At the moment unsure what happens when we change the size
    # of the sampled cube, so be careful here

    # 1. Sample sdf function inside box
    x = torch.linspace(box_dim[0], box_dim[1], grid_res, device=device)
    y = torch.linspace(box_dim[2], box_dim[3], grid_res, device=device)
    z = torch.linspace(box_dim[4], box_dim[5], grid_res, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    with torch.enable_grad():
        sdf_values = model(grid_points)
        # Ensure sdf_values is 1D: (num_vertices,)
        if sdf_values.dim() > 1:
            sdf_values = sdf_values.squeeze()

    # 2. Compute cube_idx of FlexiCubes: Has shape (num_cubes, 8), each row indexes the 8 corners of a cube
    flexicubes = FlexiCubes(device=device)
    ind = flexicubes.construct_voxel_grid(grid_res - 1)[1]

    # 3. Use FlexiCubes to extract the mesh
    ####################### eschaerer #################
    torch.cuda.synchronize()
    ##################################################

    vertices, faces, _ = flexicubes(
        voxelgrid_vertices=grid_points,  # Points
        scalar_field=sdf_values,  # Values
        cube_idx=ind,  # Cube Indices
        resolution=grid_res - 1,  # Resolution
    )

    # 54 Validate output
    if vertices.numel() == 0 or faces.numel() == 0:
        print("Warning: Empty mesh generated!")

    return vertices, faces
