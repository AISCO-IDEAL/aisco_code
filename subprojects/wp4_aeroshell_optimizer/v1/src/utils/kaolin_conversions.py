# here, our sdf2mesh stuff will be placed.

import os

import kaolin as kal
import torch
import trimesh
from kaolin.ops.conversions import (
    FlexiCubes,
    sdf_to_voxelgrids,
    voxelgrids_to_trianglemeshes,
)

from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel

####################### eschaerer #################
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
###################################################

# All Code here by aalthaus, adjusted by @eschaerer for CUDA compatibility


def get_diff_mesh(
    model: BaseModel,
    grid_res: list[int] | int = 128,
    box_dim: list = [-5000, 5000, -5000, 5000, -5000, 5000],
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

    ####################### eschaerer #################
    # Detach and clone to avoid gradient issues
    """
    print(f"SDF min={sdf_values.min().item():.3f}, max={sdf_values.max().item():.3f}")
    """
    # sdf_values = sdf_values.detach().clone()  # Problem: Enabling gradients (torch.enable_grad()) inside mesh extraction can corrupt the mesh if sdf_values or grid_points are modified elsewhere
    # grid_points = grid_points.detach().clone()
    ##################################################

    # 2. Compute cube_idx of FlexiCubes: Has shape (num_cubes, 8), each row indexes the 8 corners of a cube
    flexicubes = FlexiCubes(device=device)
    ind = flexicubes.construct_voxel_grid(grid_res - 1)[1]

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


def marching_tetrahedra(
    model: BaseModel,
    grid_res: list[int] | int = 32,
    origin: float = 0.0,
    box_size: float = 12000.0,
    upsampling: int = 4,
    dev: str = "cuda",
):
    """
    Alternative mesh extraction using marching tetrahedra from Kaolin.

    Should provide more stable normals than flexicubes, though it can't remove discontinuities completely
    Also really not differentiable on its own.

    Args:
        model: Arbitrary SDF Network representing our shape.
        grid_res: The initial resolution for the voxel grid.
        origin: Center of the bounding box, not tested what happens if not zero.
        box_size: Size of the bounding box (from origin box_size/2 in each direction).
        upsampling: Number of upsampling steps to increase resolution.
        dev: Device to run the computations on.

    """

    model.cpu()
    adjusted_model = lambda x: (model(x)[0].cpu()).squeeze(-1)  # return CPU for Kaolin

    grid = sdf_to_voxelgrids(
        [adjusted_model],
        bbox_center=origin,
        bbox_dim=box_size,
        init_res=grid_res,
        upsampling_steps=upsampling,
    )
    model.cuda()  # necessary?
    # move to GPU
    grid = grid.to(dev)

    # grid is already 0/1 occupancy
    voxels = grid.float()
    if voxels.sum() == 0:
        print("Warning: Empty voxel grid generated!")
    verts, faces = voxelgrids_to_trianglemeshes(voxels)
    verts = verts[0] / (voxels.shape[1] - 1)  # normalize to [0,1]
    verts = verts * box_size + (origin - box_size / 2)  # map back to real world coords
    # verts = verts.to(dev)
    # faces = faces.to(dev)
    return verts, faces[0]


def kaolin_to_PLY(
    verts: torch.tensor,
    faces: torch.tensor,
    save_mesh: bool = True,
    filename: str = "loading_test.ply",
    mesh_dir: str = None,  # Default to None, not a space
    verbose: bool = True,
):
    """
    Pre: Vertices and Faces generated from kaolin library functions (e.g. flexicubes or marching tetrahedra)
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
        if verbose:
            print(f"Mesh saved to {filepath}")
        return mesh
    return mesh


def kaolin_to_STL(
    verts: torch.tensor,
    faces: torch.tensor,
    save_mesh: bool = True,
    filename: str = "loading_test.stl",
    mesh_dir: str = None,  # Default to None, not a space
):
    """
    Same as kaolin_to_PLY (see above) but saves as STL. Needs Verts and faces in the kaolin format.
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
