import torch

state_dict = torch.load(
    "/cluster/home/eschaerer/AISCO_code/subprojects/wp4_aeroshell_optimizer/archive/dummy_models/sdf_sphere_model_2025-10-24.pth",
    map_location="cpu",
)
print(type(state_dict))
