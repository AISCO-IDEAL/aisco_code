#!/usr/bin/env python
"""
Standalone PLY pretraining script.
Usage:
    python /cluster/home/kdiehl/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/src/main_files/pretraining_only.py
"""

import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_v1_1 import pretraining
from subprojects.wp4_aeroshell_optimizer.v2.src.models.WIRE_v2 import WIREV2
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config

# Load model
model_config = load_config(v="v2", cfg_name="WIRE_v2.yaml")
model = WIREV2(**model_config)

# Run pretraining with a ply file
override = {
    "pretraining_shape": "ply",
    "pretraining_ply_path": "/cluster/work/fuge/AISCO/00_meshes_to_keep/original_silvretta.ply",  # "/cluster/work/fuge/AISCO/concentrate_curvature/DevelopabilityOfTriangleMeshes/assets/bunny.ply",
    "pretraining_ply_scale": 1000.0,
    "training_loop.epochs": 5000,
    "training_loop.random_batch_size": 4096,
    "training_loop.print_interval": 50,
}

trained = pretraining(model=model, override_config=override)

# Save the model
save_path = "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/pretrained_dataset/silvretta_pretrained1.pth"
torch.save(trained.state_dict(), save_path)
print(f"✓ Saved pretrained model to {save_path}")
