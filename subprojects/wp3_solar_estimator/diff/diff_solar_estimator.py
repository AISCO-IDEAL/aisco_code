import torch
import kaolin
from subprojects.wp3_solar_estimator.diff.alex_input import soft_knapsack_selection
from subprojects.wp4_aeroshell_optimizer.src.models.sdf import SDFNetwork
from subprojects.wp4_aeroshell_optimizer.src.utils.kaolin_utils import (
    flexicubes_to_PLY,
    get_diff_mesh,
)
def diff_solar_loss(model: SDFNetwork, verts: torch.tensor, faces: torch.tensor, dev: str="cuda"):
    # Const Area for testing
    max_area = 4e6 # 4 m^2
    # ToDO Aldo
    sun_dir = torch.tensor([0.0, -0.7071, 0.7071], dtype=torch.float32,device=dev)
    angles = facenormal_angle(verts, faces, sun_dir)
    powerratio = powerratio_iam(angles)
    areas = kaolin.ops.mesh.face_areas(verts, faces)[0] # [0] needed cause kaolin is weird with batch size
    loss, _, _, _ = soft_knapsack_selection(powerratio, areas, area_budget=max_area)
    return loss
def facenormal_angle(verts: torch.Tensor, faces: torch.Tensor, sun_dir: torch.Tensor):
    """
    Differentiable incidence-angle computation.
    Returns angle in degrees in [0°, 90°].
    """
    # Normalize sun direction
    s = sun_dir / (torch.norm(sun_dir) + 1e-12)
    # Face vertices
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    # Differentiable face normals
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-12)
    # Cosine of incidence angle
    cosang = torch.sum(normals * s[None, :], dim=-1)
    # Clamp to [0, 1] -> ensures angle = 0–90°
    cosang_clamped = torch.clamp(cosang, min=0.0, max=1.0)
    # Compute angle in degrees
    angle_rad = torch.acos(cosang_clamped)
    angle_deg = angle_rad * 180.0 / torch.pi
    return angle_deg  # shape = (F,)

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












