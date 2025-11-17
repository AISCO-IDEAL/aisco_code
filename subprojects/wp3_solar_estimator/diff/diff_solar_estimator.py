import torch
import kaolin
from wp3_solar_estimator.diff.alex_input import soft_knapsack_selection
from wp4_aeroshell_optimizer.src.models.sdf import SDFNetwork
from wp4_aeroshell_optimizer.src.utils.kaolin_utils import (
    flexicubes_to_PLY,
    get_diff_mesh,
)
def diff_solar_loss(model: SDFNetwork, verts: torch.tensor, faces: torch.tensor, dev: str="cuda"):
    # Const Area for testing
    max_area = 4 # 4 m^2
    # ToDO Aldo
    sun_dir = torch.tensor([0.0, -0.7071, 0.7071], dtype=torch.float32,device=dev)
    
    #GPT fix
    verts_b = verts.unsqueeze(0)     # (1, V, 3)
    faces_b = faces.unsqueeze(0)   

    angles = facenormal_angle(verts_b, faces_b, sun_dir)
    powerratio = dot_product(angles,sun_dir)

    # ensure expected dtypes/devices/contiguity
    assert verts_b.ndim == 3 and verts_b.shape[2] == 3
    assert faces_b.ndim == 3 and faces_b.shape[2] == 3
    faces_b = faces_b.long()
    if not verts_b.is_contiguous():
        verts_b = verts_b.contiguous()
    if not faces_b.is_contiguous():
        faces_b = faces_b.contiguous()

    # gather v0,v1,v2 -> shapes (B, F, 3)
    v0 = torch.gather(verts_b, 1, faces_b[:, :, 0:1].expand(-1, -1, 3))
    v1 = torch.gather(verts_b, 1, faces_b[:, :, 1:2].expand(-1, -1, 3))
    v2 = torch.gather(verts_b, 1, faces_b[:, :, 2:3].expand(-1, -1, 3))

    # cross product of edges, to get area in mm^2
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)   # (B, F, 3)
    areas = 0.5 * torch.norm(cross, dim=-1)   
    loss, _, _, _ = soft_knapsack_selection(powerratio, areas, area_budget=max_area,tau=0.2,penalty=1.0)
    return loss


def facenormal_angle(verts: torch.Tensor, faces: torch.Tensor, sun_dir: torch.Tensor):
    """
    Differentiable incidence-angle computation.
    Returns angle in degrees in [0°, 90°].
    """
    # Normalize sun direction
    s = sun_dir / (torch.norm(sun_dir) + 1e-12)
    # Face vertices
    #The first [0] index is due to some nvidia fuckery
    v0 = verts[0][faces[:, 0]] 
    v1 = verts[0][faces[:, 1]]
    v2 = verts[0][faces[:, 2]]
    # Differentiable face normals
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-12)
    # Cosine of incidence angle
    cosang = torch.sum(normals * s[None, :], dim=-1)
    # Clamp to [0, 1] -> ensures angle = 0–90°
    #cosang_clamped = torch.clamp(cosang, min=0.0, max=1.0)
    # Compute angle in degrees
    angle_rad = torch.acos(cosang)
    angle_deg = angle_rad * 180.0 / torch.pi
    return angle_deg  # shape = (F,)

def dot_product(normals: torch.Tensor, sun: torch.Tensor):
    return (normals * sun).sum(dim=-1)

def powerratio_iam(theta_deg: torch.Tensor, b0: float = 0.05):
    """
    Fully differentiable ASHRAE IAM model with smooth 90° cutoff.
    theta_deg: (F,) angles in degrees.
    """
    # Convert degrees → radians
    theta_rad = theta_deg * torch.pi / 180.0
    # Stable cosines
    cos_t = torch.cos(theta_rad)
    cos_t_safe = torch.clamp(cos_t, min=1e-6)
    # ASHRAE IAM: K(θ) = 1 − b0 * (1/cos(θ) − 1)
    iam = 1.0 - b0 * (1.0 / cos_t_safe - 1.0)
    # Smooth cutoff for θ ≥ 90°
    sharpness = 40.0
    cutoff = torch.sigmoid(-(theta_deg - 90.0) * sharpness)
    # Final differentiable value
    out = iam * cutoff
    # Clamp to [0,1]
    return out












