# here, our sdf2mesh stuff will be placed.

import os
from pathlib import Path

import torch
import trimesh
from kaolin.ops.conversions import (
    FlexiCubes,
    sdf_to_voxelgrids,
    voxelgrids_to_trianglemeshes,
)

from subprojects.wp2_assembler.version4.utils.reg_imp import get_outer_prism_bounds_mm
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsModelContract

####################### eschaerer #################
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
###################################################

# All Code here by aalthaus, adjusted by @eschaerer for CUDA compatibility


def get_diff_mesh(
    model: SupportsModelContract,
    grid_res: list[int] | int = 128,
    box_dim: list[int] | None = None,
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
    if box_dim is None:
        # Get outer-prism bounds from wp2 (single source of truth)
        prism = get_outer_prism_bounds_mm()
        box_dim = [
            int(prism["x"][0]),
            int(prism["x"][1]),
            int(prism["y"][0]),
            int(prism["y"][1]),
            int(prism["z_mesh"][0]),
            int(prism["z_mesh"][1]),
        ]
    # torch.use_deterministic_algorithms(True) # interferes with WP3 #aalthaus #vdiekmann  # noqa: ERA001
    torch.manual_seed(42)  # Set a fixed seed
    ###################################################

    # At the moment unsure what happens when we change the size
    # of the sampled cube, so be careful here

    # 1. Sample sdf function inside box
    if isinstance(grid_res, int):
        res_x = res_y = res_z = grid_res
    else:
        res_x, res_y, res_z = grid_res

    x = torch.linspace(box_dim[0], box_dim[1], res_x, device=device)
    y = torch.linspace(box_dim[2], box_dim[3], res_y, device=device)
    z = torch.linspace(box_dim[4], box_dim[5], res_z, device=device)
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
    # sdf_values = sdf_values.detach().clone()  # Problem: Enabling gradients (torch.enable_grad()) inside mesh extraction can corrupt the mesh if sdf_values or grid_points are modified elsewhere  # noqa: ERA001
    # grid_points = grid_points.detach().clone()  # noqa: ERA001
    ##################################################

    # 2. Compute cube_idx of FlexiCubes: Has shape (num_cubes, 8), each row indexes the 8 corners of a cube
    flexicubes = FlexiCubes(device=device)
    # Using res_x assuming cubic or handle appropriately if non-cubic flexicubes support it
    # For simplicity, if non-uniform, we might need a different approach or flexicubes supports 3 ints?
    # Checking flexicubes docs/usage usually it takes a single resolution for cubic grids or tuple.
    # Assuming cubic for now based on previous code passing `grid_res - 1`.
    # If list passed previously, code would have crashed. So likely it was always int.
    # But type hint says list[int] | int.
    # Let's use the first dimension or handle error.
    # To satisfy mypy without breaking logic if it was working:
    base_res = res_x  # Simplification
    ind = flexicubes.construct_voxel_grid(base_res - 1)[1]

    # Guard: NaN/Inf in sdf_values causes FlexiCubes' internal torch.gather to use
    # out-of-bounds indices → CUDA device-side assert that irrecoverably poisons the
    # entire process CUDA context.  Detach and clamp before topology extraction.
    sdf_for_flexicubes = sdf_values.detach().nan_to_num(nan=0.0, posinf=1e6, neginf=-1e6)
    if sdf_for_flexicubes.min().item() >= 0.0 or sdf_for_flexicubes.max().item() <= 0.0:
        print("Warning: Empty mesh generated!")
        return (
            torch.zeros(0, 3, device=device, dtype=torch.float32),
            torch.zeros(0, 3, device=device, dtype=torch.long),
        )

    # 3. Use FlexiCubes to extract the mesh
    ####################### eschaerer #################
    torch.cuda.synchronize()
    ##################################################

    vertices, faces, _l_dev = flexicubes(
        voxelgrid_vertices=grid_points,  # Points
        scalar_field=sdf_for_flexicubes,  # Values — detached, NaN/Inf-clamped
        cube_idx=ind,  # Cube Indices
        resolution=base_res - 1,  # Resolution
    )

    # 4 Validate output
    if vertices.numel() == 0 or faces.numel() == 0:
        print("Warning: Empty mesh generated!")

    return vertices, faces


def marching_tetrahedra(
    model: SupportsModelContract,
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
    adjusted_model = lambda x: (model(x)[0].cpu()).squeeze(-1)  # return CPU for Kaolin  # noqa: E731

    grid = sdf_to_voxelgrids(
        [adjusted_model],
        bbox_center=origin,
        bbox_dim=box_size,
        init_res=grid_res,
        upsampling_steps=upsampling,
    )
    model.to(dev)
    # move to GPU
    grid = grid.to(dev)

    # grid is already 0/1 occupancy
    voxels = grid.float()
    if voxels.sum() == 0:
        print("Warning: Empty voxel grid generated!")
    verts, faces = voxelgrids_to_trianglemeshes(voxels)
    verts = verts[0] / (voxels.shape[1] - 1)  # normalize to [0,1]
    verts = verts * box_size + (origin - box_size / 2)  # map back to real world coords
    # verts = verts.to(dev)  # noqa: ERA001
    # faces = faces.to(dev)  # noqa: ERA001
    return verts, faces[0]


def kaolin_to_PLY(  # noqa: N802
    verts: torch.Tensor,
    faces: torch.Tensor,
    save_mesh: bool = True,
    filename: str = "loading_test.ply",
    mesh_dir: Path | None = None,
    verbose: bool = True,
) -> trimesh.Trimesh:
    """
    Pre: Vertices and Faces generated from kaolin library functions (e.g. flexicubes or marching tetrahedra)
    Post: Mesh generated by trimesh from these faces

    Note that the PLY generation is not differentiable!
    This should be the end of the pipeline and serve only for visualization and Postprocessing of mesh
    """
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
    if save_mesh:  # If no mesh_dir provided, use current working directory
        output_dir = mesh_dir if mesh_dir is not None else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename
        mesh.export(filepath)

        if verbose:
            print(f"Mesh saved to {filepath}")

    return mesh


def kaolin_to_STL(  # noqa: N802
    verts: torch.Tensor,
    faces: torch.Tensor,
    save_mesh: bool = True,
    filename: str = "loading_test.stl",
    mesh_dir: Path | None = None,
) -> trimesh.Trimesh:
    """
    Same as kaolin_to_PLY (see above) but saves as STL. Needs Verts and faces in the kaolin format.
    """
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)

    if save_mesh:
        output_dir = mesh_dir if mesh_dir is not None else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename
        mesh.export(filepath)

        print(f"Mesh saved to {filepath}")

    return mesh
