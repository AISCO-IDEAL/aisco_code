import os
import sys

import torch

print("--- Test Script Running ---")
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")

# Check for GPU
if torch.cuda.is_available():
    print("\n✅ Success! PyTorch can see the GPU.")
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPU(s).")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # SLURM often sets this environment variable
    print(f"SLURM allocated GPU(s): {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
else:
    print("\nℹ️ Note: PyTorch cannot see a GPU.")
    print("This is expected if you are on a CPU-only node.")

print("--- Test Script Finished ---")
