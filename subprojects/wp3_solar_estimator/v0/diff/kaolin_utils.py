
import os
from typing import Optional

import kaolin as kal
import torch
import trimesh
from kaolin.ops.conversions import FlexiCubes
from sdf import SDFNetwork

####################### eschaerer #################
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
###################################################

# All Code here by aalthaus, adjusted by @eschaerer for CUDA compatibility


def get_diff_mesh(
    model: SDFNetwork,
    grid_res: list[int] | int = 128,
    box_dim: list = [-5, 5, -5, 5, -5, 5],
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
        sdf_values = model(grid_points)[
            0
        ]  # Get SDF values from model, first element of tuple is the scalar SDF field
    ####################### eschaerer #################
    # Detach and clone to avoid gradient issues
    print(f"SDF min={sdf_values.min().item():.3f}, max={sdf_values.max().item():.3f}")
    sdf_values = sdf_values.detach().clone()  # Problem: Enabling gradients (torch.enable_grad()) inside mesh extraction can corrupt the mesh if sdf_values or grid_points are modified elsewhere
    grid_points = grid_points.detach().clone()
    ##################################################

    # 2. Compute cube_idx of FlexiCubes: Has shape (num_cubes, 8), each row indexes the 8 corners of a cube
    flexicubes = FlexiCubes(device=device)
    ind = flexicubes.construct_voxel_grid(grid_res - 1)[1]

    """
    Debug print:
    print("x_shape: ",x.shape)
    print("grid_x shape:",grid_x.shape)
    print("grid_points shape:", grid_points.shape)
    print("sdf_values shape:", sdf_values.shape)
    print("ind shape:", ind.shape)
    print("grid_res-1 :", grid_res-1)
    """

    # 3. Use FlexiCubes to extract the mesh
    ####################### eschaerer #################
    torch.cuda.synchronize()
    ##################################################

    vertices, faces, l_dev = flexicubes(
        voxelgrid_vertices=grid_points,  # Points
        scalar_field=sdf_values,  # Values
        cube_idx=ind,  # Cube Indices
        resolution=grid_res - 1,  # Resolution
    )

    # 54 Validate output
    if vertices.numel() == 0 or faces.numel() == 0:
        print("Warning: Empty mesh generated!")

    return vertices, faces


def kaolin_facenormal_angle(
    verts: torch.Tensor, faces: torch.Tensor, sun_direction=None
) -> torch.Tensor:
    """
    Calculates the angle in degrees between each face normal of the mesh and the sun direction.
    Returns a (F,) torch tensor of angles in degrees.
    Serves solely debugging purposes! Code is stolen from solar estimator
    """
    if sun_direction is None:
        sun_direction = torch.tensor([0.0, 0.0, 1.0], device=verts.device)

    # Get face vertices
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # Compute face normals (not normalized area)
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    normals = torch.nn.functional.normalize(normals, dim=1)

    sun_dir = sun_direction / torch.norm(sun_direction)

    # Cosine of angle between normal and sun
    cos_angle = normals @ sun_dir

    # Clamp (all values between 0 and 1)
    cos_angle_clamped = torch.clamp(cos_angle, min=0.0, max=1.0)

    # Convert to angle in degrees
    angle_rad = torch.acos(cos_angle_clamped)
    angle_deg = torch.rad2deg(angle_rad)

    return angle_deg


def kaolin_primitive_max_angle_loss(
    verts: torch.Tensor, faces: torch.Tensor, sun_direction=None
) -> torch.Tensor:
    """
    Calculates a loss based on the maximum angle between face normals and the sun direction.
    The loss is higher when the maximum angle is larger, encouraging faces to align with the sun direction.

    Note that this loss has no purpose other than to prove the mesh differentiablility works
    Therefore the loss is pretty bad and just a playground
    """
    if sun_direction is None:
        sun_direction = torch.tensor([0.0, 0.0, 1.0], device=verts.device)

    angles = kaolin_facenormal_angle(verts, faces, sun_direction)

    # Very primitive loss: just take the maximum angle for differentiation.
    # print("angles: ",angles.shape)
    if torch.numel(angles) == 0:
        return torch.tensor(0.0, device=verts.device)
    max_angle = torch.max(angles)
    # print("max_angle: ", max_angle.shape)
    loss = max_angle / 180.0  # Normalize to [0, 1]
    return loss


def flexicubes_to_PLY(
    verts: torch.tensor,
    faces: torch.tensor,
    save_mesh: bool = True,
    filename: str = "loading_test.ply",
    mesh_dir: Optional[str] = None,  # Default to None, not a space
):
    """
    Pre: Vertices and Faces generated from kaolin flexicubes algorithm
    Post: Mesh generated by trimesh from these faces

    Note that the PLY generation is not differentiable!
    This should be the end of the pipeline and serve only for visualization and Postprocessing of mesh
    """
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
    if save_mesh:
        if mesh_dir is not None:
            os.makedirs(mesh_dir, exist_ok=True)
            filepath = os.path.join(mesh_dir, filename)
        else:
            filepath = filename
        mesh.export(filepath)
        print(f"Mesh saved to {filepath}")
    return mesh
    return mesh
