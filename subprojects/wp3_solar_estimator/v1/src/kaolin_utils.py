import contextlib

import torch
from kaolin.ops.conversions import (
    FlexiCubes,
)

from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel

GRID_DIMS = 3


def get_diff_mesh(
    model: BaseModel,
    grid_res: list[int] | int = 128,
    box_dim: list = [-5000, 5000, -5000, 5000, -5000, 5000],  # noqa: B006
    device: str = "cuda",
    *,
    deterministic: bool = False,
    need_grad: bool = False,
    use_amp: bool = True,
):
    """
    Pre: Gets an arbitrary SDF Model together with parameters that influence the mesh estraction process
    Post: Outputs vertices and faces that fully describe the sdf, but now converted into a mesh.

    Args:
        model: Arbitrary SDF Network representing our shape.
        grid_res: The resolution that Flexicubes uses, can also be a list of ints corresponding to the [x,y,z] resolutions
        box_dim: Defines the box in which the SDF should be sampled, for better resolution per memory decrease size of box
        device: Run torch code over cuda/cpu
        deterministic: If True, enforce deterministic algorithms and fixed seed (slower). Defaults to False for speed.
        need_grad: If True, compute sdf with gradients enabled. Defaults to False for speed.
        use_amp: If True on CUDA, evaluate SDF in mixed precision for higher throughput.
    Returns:
        Vertices and Faces as torch.Tensor
    """
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(42)
    else:
        # Allow fast kernels on modern GPUs
        torch.use_deterministic_algorithms(False)
        with contextlib.suppress(Exception):
            torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    # Normalize inputs
    if isinstance(grid_res, int):
        gx = gy = gz = grid_res
    else:
        assert len(grid_res) == GRID_DIMS, "grid_res list must be [x,y,z]"
        gx, gy, gz = map(int, grid_res)
    res_arg = [gx - 1, gy - 1, gz - 1]

    # 1. Sample sdf function inside box
    x = torch.linspace(box_dim[0], box_dim[1], gx, device=device)
    y = torch.linspace(box_dim[2], box_dim[3], gy, device=device)
    z = torch.linspace(box_dim[4], box_dim[5], gz, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    # SDF evaluation
    if need_grad:
        ctx = torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda"), dtype=torch.float16)
        with ctx:
            sdf_values = model(grid_points)
            if sdf_values.dim() > 1:
                sdf_values = sdf_values.squeeze()
        # Ensure float32 for downstream ops that may expect it
        sdf_values = sdf_values.to(dtype=torch.float32)
    else:
        ctx = torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda"), dtype=torch.float16)
        with torch.no_grad(), ctx:
            sdf_values = model(grid_points)
            if sdf_values.dim() > 1:
                sdf_values = sdf_values.squeeze()
        sdf_values = sdf_values.to(dtype=torch.float32)

    # 2. Compute cube_idx of FlexiCubes: Has shape (num_cubes, 8), each row indexes the 8 corners of a cube
    flexicubes = FlexiCubes(device=device)
    ind = flexicubes.construct_voxel_grid(resolution=res_arg)[1]

    # 3. Use FlexiCubes to extract the mesh
    vertices, faces, _ = flexicubes(
        voxelgrid_vertices=grid_points,  # Points
        scalar_field=sdf_values,  # Values
        cube_idx=ind,  # Cube Indices
        resolution=res_arg,  # Resolution
    )

    # Validate output
    if vertices.numel() == 0 or faces.numel() == 0:
        print("Warning: Empty mesh generated!")

    return vertices, faces
