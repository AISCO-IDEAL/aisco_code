import os

import torch
from tqdm import tqdm

# RUN COMMAND: python /cluster/home/roarnold/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/src/dataloader/get_metadata.py

##                              !!!!! CHANGE USERNAME !!!!!

# Update this to your new data path
DATA_DIR = "/cluster/work/fuge/AISCO/wp1_cfd_ai/united_all_datasets"


def calculate_metadata(data_dir):
    all_coords = []
    all_targets = []

    files = [f for f in os.listdir(data_dir) if f.startswith("sample_") and f.endswith(".pt")]
    print(f"Analyzing {len(files)} files using 'pos' as coordinates and 'x' as targets...")

    for f in tqdm(files):
        # Load the PyTorch Geometric Data object
        data = torch.load(os.path.join(data_dir, f), weights_only=False)

        # Pull coordinates (pos) and the pressure field target (x)
        all_coords.append(data.pos)
        all_targets.append(data.x)

    print("\nConcatenating tensors... (If the script crashes here, you are out of RAM)")

    combined_x = torch.cat(all_coords, dim=0)
    # Convert to float to avoid mean/std calculation errors
    combined_y = torch.cat(all_targets, dim=0).float()

    print("\n" + "=" * 40)
    print("COPY THESE VALUES TO src/dataloader/metadata.py")
    print("=" * 40)
    print(f"domain_x=({combined_x.min(0)[0].tolist()}, {combined_x.max(0)[0].tolist()}),")
    print(f"global_mean={combined_y.mean(0).tolist()},")
    print(f"global_std={combined_y.std(0).tolist()},")
    print("=" * 40)


if __name__ == "__main__":
    calculate_metadata(DATA_DIR)
