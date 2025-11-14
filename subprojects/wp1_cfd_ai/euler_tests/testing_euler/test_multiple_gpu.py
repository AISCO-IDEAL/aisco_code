import os
import sys
import time

import torch
from torch import nn

print("--- Multi-GPU Compute Test Starting ---")
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")

# 1. Check if GPU is available
if not torch.cuda.is_available():
    print("\n[ERROR] PyTorch cannot see any GPUs. Exiting.")
    sys.exit()

gpu_count = torch.cuda.device_count()
print(f"\n[INFO] Found {gpu_count} GPU(s).")

if gpu_count < 2:
    print(
        f"[WARNING] Only found {gpu_count} GPU. Request 2 or more GPUs in your SLURM script to test parallelism."
    )
else:
    print("[INFO] This is a multi-GPU node. Proceeding with DataParallel test.")


# 2. Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(1024, 2048)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(2048, 1024)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))


# 3. Create and wrap the model for DataParallel
print("\n[INFO] Creating a simple model...")
model = SimpleModel()

# THIS IS THE KEY: Wrap the model to use all available GPUs
print("[INFO] Wrapping model with torch.nn.DataParallel...")
model = nn.DataParallel(model)

# Move the model to the primary device (GPU:0)
# DataParallel will handle distributing work to other GPUs
device = torch.device("cuda")
model.to(device)
print(f"[INFO] Model moved to primary device: {torch.cuda.get_device_name(device)}")

# 4. Create a large batch of dummy data
# Batch size 1024, data dimension 1024
batch_size = 1024 * gpu_count  # Make the batch bigger for more GPUs
input_dim = 1024
print(f"[INFO] Creating dummy data batch (size: {batch_size} x {input_dim}) on GPU...")
data = torch.randn(batch_size, input_dim).to(device)
labels = torch.randn(batch_size, 1024).to(device)
criterion = nn.MSELoss()

# 5. Perform mock training step (forward + backward pass)
print("[INFO] Performing 10 iterations of forward + backward pass...")
torch.cuda.synchronize()
start_time = time.time()

for i in range(10):
    output = model(data)
    loss = criterion(output, labels)
    loss.backward()

torch.cuda.synchronize()
end_time = time.time()
total_time = end_time - start_time
print(f"  -> Total compute time for 10 iterations: {total_time:.4f} seconds.")
print(f"  -> Average time per iteration: {total_time / 10:.4f} seconds.")

# 6. Check GPU memory usage (as a proxy for work)
for i in range(gpu_count):
    print(
        f"  -> GPU {i} ({torch.cuda.get_device_name(i)}) Memory Used: "
        f"{torch.cuda.memory_allocated(i) / 1024**3:.2f} GB"
    )

print("\n[INFO] If all GPUs have memory allocated, DataParallel worked!")
print("--- Multi-GPU Compute Test Finished Successfully ---")
