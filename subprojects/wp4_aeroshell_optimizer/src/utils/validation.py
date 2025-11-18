import torch



from subprojects.wp4_aeroshell_optimizer.src.utils.losses import SDFLosses
from subprojects.wp4_aeroshell_optimizer.src.models.sdf import SDFNetwork
from wp2_assembler.version1.components.nbz import innerNBZ, outerNBZ
def inner_constraints_measure(model: SDFNetwork, assembly_points: list[list[float]], dev:str="cuda" ) :
    """
    args: SDF Model to Evaluate


    Returns: How much of the inner bounding box is violated in %
    
    """
    points = torch.Tensor(assembly_points,device=dev)
    
    points.requires_grad=False
    sdf_values= model(points)[0]

    return (sdf_values > 0).float().mean().item() * 100


def validation_score(model: SDFNetwork, loss_class: SDFLosses, assembly_points):
    score=(inner_constraints_measure(model, assembly_points)
           +loss_class.solar_area_loss(model)
           +loss_class.drag_loss(model)
    )
    return score