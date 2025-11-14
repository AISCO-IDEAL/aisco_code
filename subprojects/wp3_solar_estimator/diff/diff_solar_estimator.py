import torch
from subprojects.wp4_aeroshell_optimizer.src.models.sdf import SDFNetwork
from subprojects.wp4_aeroshell_optimizer.src.utils.kaolin_utils import flexicubes_to_PLY,get_diff_mesh



def diff_solar_loss(model: SDFNetwork,verts: torch.tensor,faces: torch.tensor):

    
    return torch.tensor(0.0)
