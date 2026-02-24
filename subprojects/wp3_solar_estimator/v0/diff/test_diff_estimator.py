import torch
from wp3_solar_estimator.diff.diff_solar_estimator import diff_solar_loss

from wp4_aeroshell_optimizer.src.models.sdf import SDFNetwork
from wp4_aeroshell_optimizer.src.utils.kaolin_utils import flexicubes_to_PLY, get_diff_mesh

model=torch.load("wp4_aeroshell_optimizer/data/test_models/sdf_sphere_model_2025-11-04.pth")
model.to("cuda")
for p in model.parameters():
    p.requires_grad_(True)
vertices,faces=get_diff_mesh(model)

print(vertices.requires_grad)
print(vertices.grad_fn)





loss=diff_solar_loss(model,vertices,faces)
print("Loss calculated")
loss.backward()
print("Loss propagated")
