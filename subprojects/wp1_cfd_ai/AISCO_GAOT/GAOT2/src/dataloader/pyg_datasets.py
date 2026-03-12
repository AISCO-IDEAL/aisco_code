import os

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

# Shape dimension constants
EXPECTED_DATA_SHAPE_DIMS = 3


class EnrichedData(Data):
    """Custom Data class to handle increments for bipartite edge indices."""

    def __inc__(self, key, value, *args, **kwargs):
        """
        Specifies how attributes should be incremented when batching.
        key: The name of the attribute.
        value: The attribute's tensor value.
        """
        if key.startswith("encoder_edge_index"):
            # Encoder: edge_index[0] indexes PHYSICAL, edge_index[1] indexes LATENT
            # Increment row 0 by num_physical_nodes, row 1 by num_latent_nodes
            # Ensure 'num_latent_nodes' attribute exists in the Data object!
            return torch.tensor([[self.num_nodes], [self.num_latent_nodes]])
        elif key.startswith("decoder_edge_index"):
            # Decoder: edge_index[0] indexes LATENT, edge_index[1] indexes PHYSICAL
            # Increment row 0 by num_latent_nodes, row 1 by num_physical_nodes
            return torch.tensor([[self.num_latent_nodes], [self.num_nodes]])
        elif key.startswith(("encoder_query_counts", "decoder_query_counts")):
            # Counts should not be incremented during batching
            return torch.tensor(
                [0] * value.dim(), dtype=torch.long
            )  # Or return 0 for scalar? Check PyG docs if needed. Assuming tensor counts.
        else:
            return super().__inc__(key, value, *args, **kwargs)


class VTKMeshDataset(Dataset):
    """
    PyTorch Geometric Dataset for loading preprocessed VTK mesh data.
    Assumes data is preprocessed and saved as individual .pt files containing Data objects.
    """

    def __init__(self, root, order_file, dataset_config, split="train", transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            root (str): Root directory where the dataset should be saved/found.
                        Contains raw (optional) and processed directories.
            order_file (str): Path to the order.txt file.
            dataset_config (DatasetConfig): Configuration object with train/val/test sizes.
            split (str): One of 'train', 'val', 'test'.
            transform (callable, optional): Data transformation function applied after loading.
            pre_transform (callable, optional): Data transformation function applied before saving processed data.
            pre_filter (callable, optional): Data filtering function applied before saving.
        """
        self.order_file = order_file
        self.dataset_config = dataset_config
        self.split = split
        self.active_variables = dataset_config["active_variables"] # None
        # Assuming processed files are stored in root/processed/
        super().__init__(root, transform, pre_transform, pre_filter)
        # Load indices after processing ensures processed files are available
        self._load_split_indices()

    @property
    def raw_file_names(self):
        # List raw files if needed for processing (e.g., original VTK paths)
        # For simplicity, assume processing happens externally based on order_file
        return []  # Or return list based on order_file if process() needs it

    @property
    def processed_dir(self) -> str:
        """Returns the path to the directory containing processed pyg files."""
        return os.path.join(self.root, self.dataset_config["processed_folder"])

    @property
    def processed_file_names(self):
        # Check if processed files exist based on order_file
        # This is used by PyG to check if processing is needed
        with open(self.order_file) as f:
            order_list = [line.strip() for line in f if line.strip()]
        return [f"{fname}.pt" for fname in order_list]

    def _download(self):
        # Skip raw dir creation — data is already preprocessed externally
        pass

    def download(self):
        # No download necessary if data is local
        pass

    def _process(self):
        # Skip processed dir creation — data is already preprocessed externally
        pass

    def process(self):
        # This method is called if processed files don't exist.
        # Ideally, the external script already created the .pt files.
        # If not, you could implement the VTK -> Data object conversion here.
        print("Processing raw data (should ideally be done by external script)...")
        # Placeholder: Add the conversion logic from your script here if needed.
        # Ensure it saves files named according to processed_file_names in self.processed_dir

    def _load_split_indices(self):
        with open(self.order_file) as f:
            all_filenames = [line.strip() for line in f if line.strip()]

        total_samples = len(all_filenames)
        train_size = self.dataset_config["train_size"]
        val_size = self.dataset_config["val_size"]
        test_size = self.dataset_config["test_size"]

        # Generate indices based on dataset size and split
        indices = np.arange(total_samples)
        if self.dataset_config["rand_dataset"]: # False
            rng = np.random.default_rng(seed=42)
            rng.shuffle(indices)

        if self.split == "train":
            split_indices = indices[:train_size]
        elif self.split == "val":
            split_indices = indices[train_size : train_size + val_size]
        elif self.split == "test":
            split_indices = indices[-test_size:]
        else:
            raise ValueError(f"Invalid split: {self.split}")

        self.split_filenames = [f"{all_filenames[i]}.pt" for i in split_indices]
        print(f"Loaded {len(self.split_filenames)} samples for split '{self.split}'.")

    def len(self):
        return len(self.split_filenames)

    def get(self, idx):
        filepath = os.path.join(self.processed_dir, self.split_filenames[idx])
        # Also support subdirectory layout: euler_run_1/data.pt
        if not os.path.isfile(filepath):
            alt = os.path.join(self.processed_dir, self.split_filenames[idx].removesuffix(".pt"), "data.pt")
            if os.path.isfile(alt):
                filepath = alt
        try:
            data = torch.load(filepath, weights_only=False)
            if self.active_variables is not None and hasattr(data, "x") and data.x is not None:
                data.x = data.x[:, self.active_variables]
            if hasattr(data, "x") and data.x is not None and len(data.x.shape) == EXPECTED_DATA_SHAPE_DIMS:
                data.x = data.x.squeeze(-1)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Processed file not found: {filepath}. Ensure preprocessing script was run.") from exc
        except Exception as e:
            print(f"Error loading data for index {idx} (file: {filepath}): {e}")
            raise
        else:
            return data
