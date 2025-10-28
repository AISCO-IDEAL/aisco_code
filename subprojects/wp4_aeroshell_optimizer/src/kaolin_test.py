import torch
import kaolin as kal
from   kaolin.ops.conversions import FlexiCubes

from sdf import SDFNetwork  
#All Code her by aalthaus
def get_diff_mesh(model: SDFNetwork, grid_res: int = 128,box_dim: list=[-5,5,-5,5,-5,5], device: str = 'cuda'):
    # At the moment unsure what happens when we change the size 
    # of the sampled cube, so be careful here

    x = torch.linspace(box_dim[0],box_dim[1], grid_res, device=device)
    y = torch.linspace(box_dim[2],box_dim[3], grid_res, device=device)
    z = torch.linspace(box_dim[4],box_dim[5], grid_res, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    flexicubes=FlexiCubes()
    with torch.enable_grad():
        sdf_values = model(grid_points)[0]
        #sdf_grid = sdf_values.reshape(grid_res, grid_res, grid_res)
    
    # 1. Compute cube_idx: (num_cubes, 8), each row indexes the 8 corners of a cube
    
    ind=flexicubes.construct_voxel_grid(grid_res-1)[1] #takes the cube indices
    
    # 2. Use FlexiCubes to extract the mesh
    """
    Debug print:
    print("x_shape: ",x.shape)
    print("grid_x shape:",grid_x.shape)
    print("grid_points shape:", grid_points.shape)
    print("sdf_values shape:", sdf_values.shape)
    print("ind shape:", ind.shape)
    print("grid_res-1 :", grid_res-1)
    """

    vertices, faces,l_dev = flexicubes(
        voxelgrid_vertices=grid_points,  #Points
        scalar_field=sdf_values,         #Values
        cube_idx = ind,                  #Cube Indices   
        resolution=grid_res-1              #Resolution
    )

    return vertices, faces


def facenormal_angle(verts: torch.Tensor, faces: torch.Tensor, sun_direction=torch.tensor([0., 0., 1.], device='cuda')) -> torch.Tensor:
    """
    Calculates the angle in degrees between each face normal of the mesh and the sun direction.
    Returns a (F,) torch tensor of angles in degrees.
    """
    # Get face vertices
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # Compute face normals (not normalized area)
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    normals = torch.nn.functional.normalize(normals, dim=1)

    sun_dir = sun_direction / torch.norm(sun_direction)

    # Cosine of angle between normal and sun
    cos_angle = (normals @ sun_dir)

    # Clamp (all values between 0 and 1)
    cos_angle_clamped = torch.clamp(cos_angle, min=0.0, max=1.0)

    # Convert to angle in degrees
    angle_rad = torch.acos(cos_angle_clamped)
    angle_deg = torch.rad2deg(angle_rad)

    return angle_deg

def primitive_max_angle_loss(verts: torch.Tensor, faces: torch.Tensor, sun_direction=torch.tensor([0., 0., 1.], device='cuda')) -> torch.Tensor:
    """
    Calculates a loss based on the maximum angle between face normals and the sun direction.
    The loss is higher when the maximum angle is larger, encouraging faces to align with the sun direction.
    
    Note that this loss has no purpose other than to prove the mesh differentiablility works
    Therefore the loss is pretty bad and just a playground
    """
    angles = facenormal_angle(verts, faces, sun_direction)
    
    # Very primitive loss: just take the maximum angle for differentiation.
    #print("angles: ",angles.shape)
    if torch.numel(angles) == 0:
        return torch.tensor(0.0,device='cuda')
    max_angle = torch.max(angles)
    #print("max_angle: ", max_angle.shape)
    loss = max_angle / 180.0  # Normalize to [0, 1]
    return loss