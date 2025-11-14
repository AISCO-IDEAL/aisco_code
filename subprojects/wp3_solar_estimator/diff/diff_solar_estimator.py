#Kaolin mesh documentation, also explains format of verts and faces
import kaolin.ops.mesh

import torch



from subprojects.wp4_aeroshell_optimizer.src.utils.kaolin_utils import flexicubes_to_PLY,get_diff_mesh
from subprojects.wp3_solar_estimator.diff.alex_input import soft_knapsack_selection


def diff_solar_loss(verts: torch.tensor,faces: torch.tensor):

    #Const Area for testing
    max_area = 4

    #ToDO Aldo
    powerratios = torch.tensor(0.0)

    areas = torch.tensor(0.0)

    loss, _, _, _ =soft_knapsack_selection(powerratios,areas,area_budget=max_area)

    return loss
