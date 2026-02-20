import os
import sys
import torch
from torch.utils.data import DataLoader

# Ensure Python can find the 'src' folder
sys.path.append(os.getcwd())

# Import the exact GAOT files you provided
from src.dataloader.pyg_datasets import VTKMeshDataset
from src.dataloader.collate_functions import create_collate_function

# 1. Mock the YAML Configuration
class DummyConfig:
    def __init__(self):
        # Point to the folder you generated yesterday
        self.processed_folder = "processed_drivaer"
        self.train_size = 400
        self.val_size = 50
        self.test_size = 50
        self.rand_dataset = False
        self.active_variables = None

def run_test():
    base_path = "/cluster/work/fuge/AISCO/wp1_cfd_ai"
    order_file = os.path.join(base_path, "processed_drivaer", "order.txt")

    print("--- 1. Testing VTKMeshDataset ---")
    dataset = VTKMeshDataset(
        root=base_path,
        order_file=order_file,
        dataset_config=DummyConfig(),
        split='train'
    )
    print(f"Dataset successfully loaded! Training subset size: {len(dataset)}")

    print("\n--- 2. Setting up UnifiedCollateFunction ---")
    # The GAOT collate function needs "latent tokens" (the AI's internal points)
    # to calculate the spatial edges. We will generate 1024 fake ones for this test.
    fake_latent_tokens = torch.rand(1024, 3)

    collate_fn = create_collate_function(
        coord_dim=3,
        magno_radius=0.033,
        magno_scales=[1.0],
        latent_tokens=fake_latent_tokens,
        neighbor_search_method="bidirectional",
        k_neighbors=1,
        asynchronous_graph_building=True
    )
    print("Collate function initialized with online graph building.")

    print("\n--- 3. Testing the PyTorch DataLoader ---")
    # We set batch_size=2 to ensure the collate function successfully merges two cars
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0 # Keep at 0 for a local test to see errors clearly
    )

    print("\n--- 4. Pulling the first batch (The Ultimate Test) ---")
    try:
        batch = next(iter(loader))

        print("\n🟢 SUCCESS! The DataLoader pipeline works perfectly.")
        print("Batch outputs:")
        print(f"  - Number of graphs stacked: {batch.num_graphs}")
        print(f"  - Total Physical Nodes (pos): {batch.pos.shape}")
        print(f"  - Total Normals (c): {batch.c.shape}")
        print(f"  - Target Drag Values (y): {batch.y}")

        if hasattr(batch, 'encoder_edge_index_s0'):
            print(f"  - Computed Bipartite Edges: {batch.encoder_edge_index_s0.shape}")

    except Exception as e:
        print(f"\n🔴 ERROR during data loading or collation: {e}")

if __name__ == "__main__":
    run_test()
