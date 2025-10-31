import torch
import trimesh
from crude_estimator_diff import calculateCdProxyDifferentiable

# --- CONFIGURATION ---
STL_FILE_PATH = "AISCO-code/subprojects/wp1_cfd_ai/data/crude_estimator/SilvrettaAeroskin.stl"  # <-- PUT YOUR STL PATH HERE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- Running Gradient Check on {STL_FILE_PATH} ---")
print(f"Using device: {DEVICE}")

try:
    # 1. Load the STL file using trimesh
    #    This loads it into numpy arrays
    mesh = trimesh.load_mesh(STL_FILE_PATH)
    print(f"Loaded mesh with {len(mesh.vertices)} vertices.")

    # 2. Convert numpy vertices to a PyTorch tensor
    #    THIS IS THE CRITICAL STEP:
    #    We tell PyTorch to track gradients for this tensor.
    vertices_tensor = torch.tensor(
        mesh.vertices,
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True,  # <--- THIS IS THE KEY
    )

    # 3. Calculate the loss
    #    We use dynamic bounds (slice_bounds=None)
    loss = calculateCdProxyDifferentiable(vertices_tensor)

    print(f"Calculated Loss: {loss.item():.6f}")

    # 4. Perform the backward pass
    print("Performing backward pass...")
    loss.backward()

    # 5. Check if the gradient exists
    if vertices_tensor.grad is not None:
        print("✅ Backpropagation Successful!")
        print(f"Gradients were computed (shape: {vertices_tensor.grad.shape}).")
    else:
        print("❌ Backpropagation FAILED: Gradients are None.")

except Exception as e:
    print(f"❌ Test FAILED with an error: {e}")

print("--- Test Complete ---")
