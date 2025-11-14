import os
import sys
import time

import torch

print("--- GPU Compute Test Starting ---")
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")

# 1. Check if GPU is available
if not torch.cuda.is_available():
    print("\n[ERROR] PyTorch cannot see a GPU. Exiting.")
    print(
        f"SLURM allocated GPU(s): {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}"
    )
    sys.exit()

device = torch.device("cuda")
print(
    f"\n[INFO] Success! PyTorch is using device: {torch.cuda.get_device_name(device)}"
)

# 2. Set up large matrices
# 15000 x 15000 matrices will make an RTX 4090 work for a moment
matrix_size = 15000
print(f"[INFO] Creating two random {matrix_size}x{matrix_size} matrices on the CPU...")

# Create on CPU first
a_cpu = torch.randn(matrix_size, matrix_size, dtype=torch.float32)
b_cpu = torch.randn(matrix_size, matrix_size, dtype=torch.float32)

# 3. Move data to GPU
print("[INFO] Moving matrices to the GPU...")
start_transfer = time.time()
a_gpu = a_cpu.to(device)
b_gpu = b_cpu.to(device)
# Wait for the transfer to complete (synchronize)
torch.cuda.synchronize()
end_transfer = time.time()
print(f"  -> Transfer to GPU took: {end_transfer - start_transfer:.4f} seconds.")

# 4. Perform computation on GPU
print(
    f"\n[INFO] Performing {matrix_size}x{matrix_size} matrix multiplication on the GPU (10 iterations)..."
)
start_compute = time.time()

c_gpu = None  # Initialize
for i in range(10):
    c_gpu = torch.matmul(a_gpu, b_gpu)

# Wait for all computations to finish
torch.cuda.synchronize()
end_compute = time.time()

total_time = end_compute - start_compute
avg_time = total_time / 10
print(f"  -> Total compute time for 10 iterations: {total_time:.4f} seconds.")
print(f"  -> Average time per multiplication: {avg_time:.4f} seconds.")

# 5. Move result back to CPU (optional, but good for cleanup)
print("\n[INFO] Moving result matrix back to the CPU...")
c_cpu = c_gpu.cpu()

print("\n--- GPU Compute Test Finished Successfully ---")
