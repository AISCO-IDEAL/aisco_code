import torch
import os
from tqdm import tqdm

# Update this to your new data path
DATA_DIR = "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler/"

def calculate_metadata(data_dir):
    all_coords = []
    all_targets = []

    files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    print(f"Analyzing {len(files)} files using 'pos' as coordinates...")

    for f in tqdm(files):
        data = torch.load(os.path.join(data_dir, f), weights_only=False)

        # Pull coordinates (pos) and targets (y)
        all_coords.append(data.pos)
        all_targets.append(data.y)

    combined_x = torch.cat(all_coords, dim=0)
    # Convert to float to avoid mean/std calculation errors with integers
    combined_y = torch.cat(all_targets, dim=0).float()

    print("\n" + "="*40)
    print("COPY THESE VALUES TO src/dataloader/metadata.py")
    print("="*40)
    print(f"domain_x=({combined_x.min(0)[0].tolist()}, {combined_x.max(0)[0].tolist()}),")
    print(f"global_mean={combined_y.mean(0).tolist()},")
    print(f"global_std={combined_y.std(0).tolist()},")
    print("="*40)

if __name__ == "__main__":
    calculate_metadata(DATA_DIR)
