import glob
import os

import pandas as pd
import pyvista as pv
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# expects .ply file with shape and .csv file with drag values for the respective shape

# saves .pt file containing position pos, normals c and target variable y (drag coefficient Cd)


def process_custom_drivaer(base_path, output_dir, target_col="Cd"):
    """
    Processes custom DrivAer run folders into PyG-compatible .pt files.
    """
    os.makedirs(output_dir, exist_ok=True)
    order_list = []

    # 1. Find all 'run_X' folders
    run_folders = glob.glob(os.path.join(base_path, "run_*"))

    # Sort them nicely (run_1, run_2, etc. instead of run_1, run_10, run_2)
    run_folders.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))

    for run_folder in tqdm(run_folders, desc="Processing runs"):
        folder_name = os.path.basename(run_folder)  # e.g., 'run_1'
        run_number = folder_name.split("_")[1]  # e.g., '1'

        # 2. Set paths to your specific files
        ply_file = os.path.join(run_folder, f"drivaer_{run_number}.ply")
        csv_file = os.path.join(run_folder, f"force_mom_{run_number}.csv")

        if not os.path.exists(ply_file) or not os.path.exists(csv_file):
            print(f"Missing files in {folder_name}. Skipping...")
            continue

        # 3. Extract Geometry (pos) and Normals (c)
        try:
            mesh = pv.read(ply_file)
            mesh = mesh.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)

            pos = torch.tensor(mesh.points, dtype=torch.float32)
            c = torch.tensor(mesh.point_data["Normals"], dtype=torch.float32)
        except Exception as e:
            print(f"Error reading 3D mesh for {folder_name}: {e}")
            continue

        # 4. Dummy Feature (x) - Required by your collate function and dataset
        # We use a tensor of 1s since you are predicting based purely on shape
        # x = torch.ones((pos.shape[0], 1), dtype=torch.float32) --> workaround for collate function,
        #  which expects x to exist and be 2D (num_nodes, num_features) but will try to solve it in collate functions

        # 5. Extract Target Variable (y) from CSV
        try:
            rd = pd.read_csv(csv_file)
            # Grab the last row's value (assuming the CFD simulation converged at the end)
            drag_value = rd[target_col].iloc[-1]
            y = torch.tensor([drag_value], dtype=torch.float32)
        except Exception as e:
            print(f"Error reading CSV for {folder_name}: {e}")
            continue

        # 6. Build and Save the PyG Data Object
        data = Data(pos=pos, c=c, y=y)
        data.x = None  # Explicitly set x to None since we're not using it
        data.filename = folder_name

        save_path = os.path.join(output_dir, f"{folder_name}.pt")
        torch.save(data, save_path)
        order_list.append(folder_name)

    # 7. Generate the order.txt file automatically!
    with open(os.path.join(output_dir, "order.txt"), "w") as f:
        f.writelines(f"{name}\n" for name in order_list)

    print(f"\nSuccess! Processed {len(order_list)} runs.")
    print(f"Data and order.txt saved to: {output_dir}")


if __name__ == "__main__":
    # Update these paths if your folder is located somewhere else
    base_path = "/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_complete"
    output_dir = "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_drivaer"

    # IMPORTANT: Change 'Cd' to whatever the drag column is actually named in force_mom_X.csv
    target_column_name = "Cd"

    process_custom_drivaer(base_path, output_dir, target_col=target_column_name)
