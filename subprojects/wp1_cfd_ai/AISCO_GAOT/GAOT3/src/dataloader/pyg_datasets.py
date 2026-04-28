import hashlib
import logging
import os
from collections import Counter
from collections.abc import Callable
from typing import cast

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data, Dataset

try:
    from torch_geometric.nn.pool import fps as _fps_fn
except ImportError:
    _fps_fn = None

# Shape dimension constants
EXPECTED_DATA_SHAPE_DIMS = 3

# ---------------------------------------------------------------------------
# Subsampling strategies
# ---------------------------------------------------------------------------


def _make_generator(seed: int | None = None) -> torch.Generator:
    gen = torch.Generator()
    if seed is None:
        seed = torch.initial_seed()
    gen.manual_seed(seed)
    return gen


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % ((1 << 63) - 1)


def _subsample_random(pos: torch.Tensor, n_target: int, seed: int | None = None) -> torch.Tensor:
    """Random uniform subsampling (original behaviour).

    Uses an independent RNG derived from either the supplied seed or
    ``torch.initial_seed``.
    """
    _gen = _make_generator(seed)
    return torch.randperm(pos.shape[0], generator=_gen)[:n_target]


def _subsample_voxel(pos: torch.Tensor, n_target: int, seed: int | None = None, tol: float = 0.10) -> torch.Tensor:
    """Voxel-grid subsampling: keep one random point per voxel.

    Ensures spatially uniform coverage regardless of local mesh density.
    The voxel size is found via binary search starting from an analytical
    estimate so the number of occupied voxels is within ``tol`` of
    ``n_target`` (default 10%).  Typically converges in 2-4 iterations.

    The voxel grid is shifted by a random offset each call so that points
    near voxel boundaries fall into different voxels each epoch.  This
    reduces the pairwise overlap between epochs from ~60% to ~47% while
    preserving full spatial coverage (no region is ever unrepresented).

    Uses an independent RNG derived from either the supplied seed or
    ``torch.initial_seed``.
    """
    # Independent generator — NOT affected by torch.manual_seed()
    _gen = _make_generator(seed)

    pos_min = pos.min(dim=0).values
    pos_max = pos.max(dim=0).values
    extent = (pos_max - pos_min).clamp(min=1e-8)

    # Random fractional grid offset, drawn once so it is fixed across the
    # binary search iterations (consistent count) but varies between epochs.
    # Shifting by rand * voxel_size moves boundary-straddling points into
    # different voxels and eliminates the singleton-voxel always-same-node
    # problem.
    _offset_frac = torch.rand(3, generator=_gen)  # in [0, 1) voxel units per axis

    def _count(voxel_size: float) -> tuple[int, torch.Tensor]:
        offset = _offset_frac * voxel_size
        vc = ((pos - pos_min + offset) / voxel_size).long()
        dims = vc.max(dim=0).values + 1
        flat = vc[:, 0] * (dims[1] * dims[2]) + vc[:, 1] * dims[2] + vc[:, 2]
        perm = torch.randperm(pos.shape[0], generator=_gen)
        flat_perm = flat[perm]
        sorted_vals, sorted_local = flat_perm.sort()
        mask = torch.cat([torch.tensor([True]), sorted_vals[1:] != sorted_vals[:-1]])
        kept = perm[sorted_local[mask]]
        return len(kept), kept

    # Analytical starting estimate: n_target occupied voxels ≈ volume / v^3
    volume = extent[0].item() * extent[1].item() * extent[2].item()
    v_init = max((volume / n_target) ** (1.0 / 3.0), 1e-8)

    lo = 0.0
    hi = extent.max().item()
    best_idx = None

    # Evaluate analytical estimate first, then binary search with early exit
    mid = v_init
    for _ in range(10):
        n, idx = _count(mid)
        best_idx = idx
        if abs(n - n_target) / n_target <= tol:
            break
        if n < n_target:
            hi = mid  # fewer voxels than wanted → smaller voxels
        else:
            lo = mid  # more voxels than wanted → larger voxels
        mid = (lo + hi) / 2.0

    assert best_idx is not None
    return best_idx


def _subsample_fps(pos: torch.Tensor, n_target: int, seed: int | None = None) -> torch.Tensor:
    """Farthest-point sampling: iteratively pick the point farthest from the
    already-selected set.  Gives excellent spatial coverage but is slower
    than random or voxel (~40 s for 135 K → 50 K on CPU).

    Uses ``torch_geometric.nn.pool.fps`` when available, otherwise falls
    back to a pure-PyTorch implementation.
    """
    ratio = min(n_target / pos.shape[0], 1.0)
    if _fps_fn is not None:
        batch_idx = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        idx = _fps_fn(pos, batch_idx, ratio=ratio, random_start=seed is None)
    else:
        # Pure-Python fallback (slow but correct)
        _gen = _make_generator(seed)
        n = pos.shape[0]
        selected: list[int] = [int(torch.randint(n, (1,), generator=_gen).item())]
        dists = torch.full((n,), float("inf"))
        for _ in range(min(n_target, n) - 1):
            last = pos[selected[-1]].unsqueeze(0)
            d = (pos - last).pow(2).sum(dim=-1)
            dists = torch.min(dists, d)
            selected.append(int(dists.argmax().item()))
        idx = torch.tensor(selected, dtype=torch.long)

    return idx[:n_target]


def _subsample_poisson_disk(pos: torch.Tensor, n_target: int, seed: int | None = None) -> torch.Tensor:
    """Poisson-disk sampling: greedily accept points only if they are at least
    ``r`` away from all already-accepted points.  Uses a spatial hash (voxel
    grid with cell size = r / √3) for O(1) neighbour checks.

    The radius ``r`` is estimated to achieve approximately ``n_target`` points
    and refined via a few coarse iterations if the first guess under- or over-
    shoots by more than 10 %.

    Quality is between voxel (fast, slightly clustered) and FPS (optimal but
    slow).  Speed is ~0.3-0.8 s for 135 K → 50 K.
    """
    n = pos.shape[0]
    pos_np = pos.detach().cpu().numpy()

    # Estimate radius: for N points uniformly filling a surface,
    # n_target ≈ A / (π r²) where A is total surface area.  We don't know A,
    # so use a heuristic based on k=1 nearest-neighbour distances.
    tree = cKDTree(pos_np)
    d1, _ = tree.query(pos_np, k=2, workers=-1)  # k=2 because self is always 0
    median_nn = float(np.median(d1[:, 1]))
    # Scale factor: if all n points have spacing median_nn, then after
    # Poisson disk at radius r we keep n * (median_nn / r)^2 points
    r_est = median_nn * (n / n_target) ** 0.5

    rng_seed = seed if seed is not None else torch.initial_seed()
    rng = np.random.default_rng(rng_seed)

    def _run(r: float) -> np.ndarray:
        """Single pass of greedy Poisson-disk acceptance."""
        cell = r / (3.0**0.5)
        inv_cell = 1.0 / cell
        grid: dict[tuple[int, int, int], bool] = {}
        order = rng.permutation(n)
        accepted = []

        for i in order:
            p = pos_np[i]
            gx, gy, gz = int(p[0] * inv_cell), int(p[1] * inv_cell), int(p[2] * inv_cell)

            # Check 3x3x3 neighbourhood in the grid
            too_close = False
            for dx in (-1, 0, 1):
                if too_close:
                    break
                for dy in (-1, 0, 1):
                    if too_close:
                        break
                    for dz in (-1, 0, 1):
                        nbr = (gx + dx, gy + dy, gz + dz)
                        if nbr in grid:
                            too_close = True
                            break

            if not too_close:
                accepted.append(i)
                grid[(gx, gy, gz)] = True

        return np.array(accepted, dtype=np.int64)

    # Adaptive radius: try a few times to hit close to n_target
    r = r_est
    for _ in range(5):
        idx = _run(r)
        ratio = len(idx) / n_target
        if 0.9 <= ratio <= 1.1:
            break
        # Adjust: more points → increase r, fewer → decrease r
        r *= ratio**0.5

    return torch.from_numpy(idx).long()


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


# These expected dims handle the squeezing of scalar fields (e.g., [N, 1] -> [N])
EXPECTED_DATA_SHAPE_DIMS = 2


class VTKMeshDataset(Dataset):
    """
    PyTorch Geometric Dataset for loading preprocessed VTK mesh data.
    Standardizes attributes across unified datasets to prevent DataLoader KeyErrors.
    """

    def __init__(self, root, order_file, dataset_config, split="train", transform=None, pre_transform=None, pre_filter=None):
        self.order_file = order_file
        self.dataset_config = dataset_config
        self.split = split
        self.active_variables = dataset_config.get("active_variables", None)
        self.drop_pressure = dataset_config.get("drop_pressure", False)
        self.subsample_nodes = dataset_config.get("subsample_nodes", None)
        self.subsample_strategy = dataset_config.get("subsample_strategy", "random")
        self.deterministic_subsampling = dataset_config.get("deterministic_subsampling", False)
        self.subsampling_seed = dataset_config.get("subsampling_seed", 42)

        _valid = {"random", "voxel", "fps", "poisson_disk"}
        if self.subsample_strategy not in _valid:
            raise ValueError(f"subsample_strategy must be one of {_valid}, got '{self.subsample_strategy}'")

        super().__init__(root, transform, pre_transform, pre_filter)
        self._load_split_indices()

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.dataset_config["processed_folder"])

    @property
    def processed_file_names(self):
        with open(self.order_file) as f:
            order_list = [line.strip() for line in f if line.strip()]
        return [f"{fname}.pt" for fname in order_list]

    def _load_split_indices(self):
        with open(self.order_file) as f:
            all_filenames = [line.strip() for line in f if line.strip()]

        total_samples = len(all_filenames)
        train_size = self.dataset_config["train_size"]
        val_size = self.dataset_config["val_size"]
        test_size = self.dataset_config["test_size"]

        if self.dataset_config.get("handpicked_val", False) and self._validation_list_path():
            split_indices = self._load_handpicked_validation_split_indices(
                all_filenames=all_filenames,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
            )
            self.split_filenames = [f"{all_filenames[i]}.pt" for i in split_indices]
            print(f"Loaded {len(self.split_filenames)} samples for split '{self.split}'.")
            return

        indices = np.arange(total_samples)
        if self.dataset_config.get("rand_dataset", False):
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

    def _load_handpicked_validation_split_indices(
        self,
        all_filenames: list[str],
        train_size: int,
        val_size: int,
        test_size: int,
    ) -> np.ndarray:
        validation_names = self._read_validation_file_names()
        order_name_to_index = {self._normalize_sample_name(name): index for index, name in enumerate(all_filenames)}

        if not validation_names:
            raise ValueError("Handpicked validation list is empty.")

        duplicate_names = sorted(name for name, count in Counter(validation_names).items() if count > 1)
        if duplicate_names:
            raise ValueError(f"Handpicked validation list contains duplicate sample names: {duplicate_names}")

        logger = logging.getLogger(__name__)
        missing_names = sorted(name for name in validation_names if name not in order_name_to_index)
        if missing_names:
            logger.warning("Skipping handpicked validation samples not present in order_file: %s", missing_names)

        validation_names = [name for name in validation_names if name in order_name_to_index]
        missing_files = sorted(
            name
            for name in validation_names
            if not self._processed_sample_exists(all_filenames[order_name_to_index[name]])
        )
        if missing_files:
            logger.warning("Skipping handpicked validation samples with missing processed files: %s", missing_files)

        validation_names = [
            name
            for name in validation_names
            if self._processed_sample_exists(all_filenames[order_name_to_index[name]])
        ]

        candidate_indices = np.array([order_name_to_index[name] for name in validation_names], dtype=int)
        candidate_order = np.arange(len(candidate_indices))
        if self.dataset_config.get("rand_dataset", False):
            rng = np.random.default_rng(seed=42)
            rng.shuffle(candidate_order)

        # val_files_path is a handpicked candidate pool.  If it contains more
        # samples than val_size, only the selected val_size samples are held out;
        # unselected candidates return to the ordinary train/test slicing pool.
        val_indices = candidate_indices[candidate_order[:val_size]]

        all_indices = np.arange(len(all_filenames))
        if self.dataset_config.get("rand_dataset", False):
            rng = np.random.default_rng(seed=42)
            rng.shuffle(all_indices)
        all_indices = np.array(
            [index for index in all_indices if self._processed_sample_exists(all_filenames[index])],
            dtype=int,
        )
        val_index_set = set(val_indices)
        remaining_indices = np.array([index for index in all_indices if index not in val_index_set], dtype=int)

        if len(val_indices) < val_size:
            fill_count = val_size - len(val_indices)
            if fill_count > len(remaining_indices):
                raise ValueError(
                    "Not enough available samples to fill handpicked validation split: "
                    f"need {fill_count}, have {len(remaining_indices)}"
                )
            logger.warning(
                "Handpicked validation list has only %s usable samples for val_size=%s; "
                "filling the remaining %s from available non-handpicked samples.",
                len(val_indices),
                val_size,
                fill_count,
            )
            val_indices = np.concatenate([val_indices, remaining_indices[:fill_count]])
            val_index_set = set(val_indices)
            remaining_indices = np.array([index for index in all_indices if index not in val_index_set], dtype=int)

        if train_size + test_size > len(remaining_indices):
            raise ValueError(
                "train_size + test_size is larger than the non-validation sample count: "
                f"{train_size} + {test_size} > {len(remaining_indices)}"
            )

        if self.split == "train":
            return remaining_indices[:train_size]
        if self.split == "val":
            return val_indices
        if self.split == "test":
            return remaining_indices[-test_size:] if test_size > 0 else np.array([], dtype=int)
        raise ValueError(f"Invalid split: {self.split}")

    def _read_validation_file_names(self) -> list[str]:
        path = self._validation_list_path()
        if path is None:
            return []
        with open(path) as f:
            raw_lines = f.readlines()
        names = []
        for raw_line in raw_lines:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("-"):
                line = line[1:].strip()
            names.append(self._normalize_sample_name(line.strip("'\"")))
        return names

    def _validation_list_path(self) -> str | None:
        configured_path = self.dataset_config.get("val_files_path") or self.dataset_config.get("val_file_list")
        if not configured_path:
            return None
        configured_path = os.fspath(configured_path)
        if os.path.isabs(configured_path):
            return configured_path
        if os.path.exists(configured_path):
            return configured_path
        return os.path.join(self.root, configured_path)

    @staticmethod
    def _normalize_sample_name(name: str) -> str:
        return name.strip().removesuffix(".pt")

    def _processed_sample_exists(self, name: str) -> bool:
        sample_name = self._normalize_sample_name(name)
        direct_path = os.path.join(self.processed_dir, f"{sample_name}.pt")
        nested_path = os.path.join(self.processed_dir, sample_name, "data.pt")
        return os.path.isfile(direct_path) or os.path.isfile(nested_path)

    def len(self):
        return len(self.split_filenames)

    def get(self, idx):  # noqa: PLR0912
        filepath = os.path.join(self.processed_dir, self.split_filenames[idx])

        if not os.path.isfile(filepath):
            alt = os.path.join(self.processed_dir, self.split_filenames[idx].removesuffix(".pt"), "data.pt")
            if os.path.isfile(alt):
                filepath = alt

        try:
            # Note: weights_only=False is used because Data objects contain custom attributes
            data = torch.load(filepath, weights_only=False)
            data.filename = self.split_filenames[idx]
            num_nodes = data.pos.shape[0]

            # --- SCHEMA STANDARDIZATION ---
            # Ensures all samples have the same keys before batching.
            # We add these BEFORE subsampling so the masks/slices apply to them too.

            if not hasattr(data, "wall_shear"):
                data.wall_shear = torch.zeros((num_nodes, 3), dtype=torch.float32)

            if not hasattr(data, "pressure"):
                data.pressure = torch.zeros((num_nodes, 1), dtype=torch.float32)

            if not hasattr(data, "cp"):
                data.cp = torch.zeros((num_nodes, 1), dtype=torch.float32)

            if not hasattr(data, "n_surface_faces"):
                # We use -1 or 0 to indicate it's missing/not applicable
                data.n_surface_faces = 0

            # Drop all string/dict metadata attributes that PyG cannot batch as tensors.
            for _str_attr in (
                "source_dataset",
                "source_case_id",
                "geom_name",
                "preprocess_version",
                "cd_mean50",
                "cd_std50",
                "converged",
                "final_residuals",
                "flow_conditions",
            ):
                if hasattr(data, _str_attr):
                    delattr(data, _str_attr)

            # Drop raw pressure field if requested (model usually trains on Cp)
            if self.drop_pressure and hasattr(data, "pressure"):
                del data.pressure

            # Drop face connectivity for subsampling consistency (as per your original logic)
            if self.subsample_nodes is not None and hasattr(data, "face"):
                del data.face

            # --- NODE SUBSAMPLING ---
            if self.subsample_nodes is not None and num_nodes > self.subsample_nodes:
                # Capture original count before modification
                original_num_nodes = num_nodes

                # Mapping strategies to functions (assumed to be in scope)
                _subsample_fns = {
                    "random": _subsample_random,
                    "voxel": _subsample_voxel,
                    "fps": _subsample_fps,
                    "poisson_disk": _subsample_poisson_disk,
                }

                _strategy_fn = cast(
                    Callable[[torch.Tensor, int, int | None], torch.Tensor], _subsample_fns[self.subsample_strategy]
                )
                subsample_seed = None
                if self.deterministic_subsampling:
                    subsample_seed = _stable_seed(self.subsampling_seed, self.split, idx, self.subsample_strategy)

                perm = _strategy_fn(data.pos, self.subsample_nodes, subsample_seed)

                # Apply permutation to every per-node tensor (including newly added zeros)
                for attr in list(data.keys()):
                    val = data[attr]
                    if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] == original_num_nodes:
                        data[attr] = val[perm]

            # --- FEATURE SELECTION & SQUEEZING ---
            if self.active_variables is not None and hasattr(data, "x") and data.x is not None:
                data.x = data.x[:, self.active_variables]

            # Squeeze scalar fields from [N, 1] to [N] if they match EXPECTED_DATA_SHAPE_DIMS
            # Keep everything as [N, 1] for consistency with MSE loss
            if hasattr(data, "x") and data.x is not None and data.x.dim() == 1:
                data.x = data.x.unsqueeze(-1)

        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Processed file not found: {filepath}.") from exc
        except Exception as e:
            print(f"Error loading data for index {idx} (file: {filepath}): {e}")
            raise
        else:
            return data
