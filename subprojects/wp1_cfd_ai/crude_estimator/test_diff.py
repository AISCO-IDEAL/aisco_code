import os
import sys

import torch

# 1. Get the path to the current file (test_diff.py)
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up two levels to get to the 'subprojects' directory
subprojects_dir = os.path.dirname(os.path.dirname(current_file_dir))

# 3. Build the path to the 'wp4' source folder
#    (Assuming kaolin_test.py is in 'src' with your other wp4 files)
wp4_src_path = os.path.join(subprojects_dir, "wp4_aeroshell_optimizer", "src")

# 4. Add this path to sys.path
if wp4_src_path not in sys.path:
    sys.path.append(wp4_src_path)

# --- Now you can import it like it's in the same folder ---
import kaolin_test
from crude_estimator import calculateCdProxy
from crude_estimator_diff import calculateCdProxyDifferentiable, visualize_slices
from kaolin_test import flexicubes_to_PLY, get_diff_mesh

# --- Start of Test ---
print("--- Running Full Differentiation Test (Model -> Vertices -> Loss) ---")

try:
    # 1. Load Model
    print("Loading model...")
    model = torch.load(
        "AISCO-code/subprojects/wp1_cfd_ai/data/crude_estimator/sdf_model_10_31_09_32.pth",
        map_location="cuda",
    )
    # Ensure model is in training mode and gradients are enabled
    model.train()
    print(f"Model loaded onto {next(model.parameters()).device}.")

    # 2. Get Differentiable Mesh
    # get_diff_mesh should have torch.enable_grad() inside it
    print("Generating differentiable mesh from model...")
    vertices, faces = get_diff_mesh(model, 150, box_dim=[-2, 2, -2, 2, -2, 2], device="cuda")

    if vertices.shape[0] == 0:
        raise RuntimeError("get_diff_mesh produced 0 vertices.")

    print(f"Generated {vertices.shape[0]} vertices.")

    print(
        f"Calculated CD Proxy (easy) for SDF: {calculateCdProxy('/cluster/home/elihuber/AISCO-code/subprojects/wp1_cfd_ai/data/crude_estimator/test_output_diff_mesh.ply', True, '/cluster/home/elihuber/AISCO-code/subprojects/wp1_cfd_ai/data/crude_estimator')}"
    )

    # 3. Calculate Loss
    # We use the dynamic "std dev" bounds
    print("Calculating differentiable loss...")
    loss = calculateCdProxyDifferentiable(vertices, slice_bounds=None)

    if loss is None:
        raise RuntimeError("Loss calculation returned None.")

    print(f"Calculated Loss: {loss.item():.6f}")

    # 4. Perform Backpropagation
    # This calculates gradients from the 'loss' all the way
    # back to the 'model.parameters()'
    print("Performing backward pass...")
    loss.backward()

    # 5. Check for Gradients on the Model
    # The gradient should flow back to the model that *generated* the vertices
    grad_found = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Check if gradients are valid (not all NaN or Inf)
            if torch.isfinite(param.grad).all():
                grad_found = True
            else:
                print(f"⚠️ Warning: Found non-finite gradients (NaN/Inf) in {name}.")
            break  # We only need to find one valid gradient

    if grad_found:
        print("✅ Backpropagation Successful!")
        print("Valid gradients were computed for model parameters.")
    else:
        print("❌ Backpropagation FAILED: No gradients found on model parameters.")

except Exception as e:
    print(f"❌ Backpropagation FAILED with an error: {e}")

visualize_slices(
    vertices,
    "/cluster/home/elihuber/AISCO-code/subprojects/wp1_cfd_ai/data/crude_estimator/test_output_diff_mesh_slices_2.glb",
)
print("--- Test Complete ---")
