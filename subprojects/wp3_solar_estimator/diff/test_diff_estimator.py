import torch

from subprojects.wp4_aeroshell_optimizer.src.models.sdf import SDFNetwork
from subprojects.wp4_aeroshell_optimizer.src.utils.kaolin_utils import flexicubes_to_PLY,get_diff_mesh
from subprojects.wp3_solar_estimator.diff.diff_solar_estimator import diff_solar_loss


model=torch.load("wp4_aeroshell_optimizer/data/test_models/sdf_sphere_model_2025-11-04.pth")
model.to("cuda")
vertices,faces=get_diff_mesh(model)
verts_b = vertices.unsqueeze(0)     # (1, V, 3)
faces_b = faces.unsqueeze(0)     # (1, F, 3)




loss=diff_solar_loss(model,verts_b,faces_b)
print("Loss calculated")
loss.backward()
print("Loss propagated")