import os
import re
from pathlib import Path

import torch


MAIN_CONFIG_PATH = Path(__file__).parent / "config" / "main_config.yaml"


class ShearForce:
    def __init__(self, n_per_sqm=0.5287701487541199, unit="m^2"):
        self.n_per_sqm = n_per_sqm
        self.factor = 1.0
        if unit == "mm^2":
            self.factor = 0.000001

    @staticmethod
    def _as_triangle_faces(faces: torch.Tensor) -> torch.Tensor:
        if faces.ndim != 2:
            raise ValueError(f"Expected faces to be a 2D tensor, got shape {tuple(faces.shape)}.")
        if faces.shape[1] == 3:
            return faces.long()
        if faces.shape[0] == 3:
            return faces.t().contiguous().long()
        raise ValueError(f"Expected faces shaped [F, 3] or [3, F], got {tuple(faces.shape)}.")

    @staticmethod
    def _face_areas(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        faces = ShearForce._as_triangle_faces(faces)
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        return 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)

    @staticmethod
    def _get_attr(sample, name: str):
        if isinstance(sample, dict):
            return sample.get(name)
        return getattr(sample, name, None)

    @staticmethod
    def _iter_samples(dataset):
        if hasattr(dataset, "pos") or isinstance(dataset, dict):
            yield dataset
            return
        yield from dataset

    def calculate_shear_force(self, verts: torch.Tensor, faces: torch.Tensor):
        total_area = torch.sum(self._face_areas(verts, faces))
        return self.n_per_sqm * total_area * self.factor

    def calculate_average_newton_per_sqm(
        self,
        dataset,
        shear_attr: str = "wall_shear",
        pos_attr: str = "pos",
        face_attr: str = "face",
    ) -> torch.Tensor:
        """
        Calculate the dataset-level area-weighted average shear stress in N/m^2.

        If a sample contains a wall-shear vector field, the method computes the
        area-weighted mean of its vector magnitude. Samples without usable
        wall-shear data fall back to this instance's constant ``n_per_sqm``.
        """
        total_weighted_shear = None
        total_area = None

        for sample in self._iter_samples(dataset):
            verts = self._get_attr(sample, pos_attr)
            faces = self._get_attr(sample, face_attr)
            if verts is None or faces is None or faces.numel() == 0:
                continue

            faces = self._as_triangle_faces(faces).to(device=verts.device)
            face_areas = self._face_areas(verts, faces)
            sample_area = face_areas.sum()
            if sample_area <= 0:
                continue

            wall_shear = self._get_attr(sample, shear_attr)
            if wall_shear is None or wall_shear.numel() == 0:
                n_per_sqm = torch.as_tensor(self.n_per_sqm, device=verts.device, dtype=verts.dtype)
                sample_weighted_shear = n_per_sqm * sample_area
            else:
                wall_shear = wall_shear.to(device=verts.device, dtype=verts.dtype)
                if wall_shear.shape[0] != verts.shape[0]:
                    raise ValueError(
                        f"Expected {shear_attr} to have one value per vertex ({verts.shape[0]}), got {wall_shear.shape[0]}."
                    )
                if wall_shear.ndim == 1:
                    shear_magnitude = wall_shear.abs()
                elif wall_shear.ndim == 2:
                    shear_magnitude = torch.norm(wall_shear, dim=1)
                else:
                    raise ValueError(f"Expected {shear_attr} shaped [N] or [N, C], got {tuple(wall_shear.shape)}.")

                vertex_areas = torch.zeros(verts.shape[0], device=verts.device, dtype=verts.dtype)
                face_area_share = (face_areas / 3.0).repeat_interleave(3)
                vertex_areas.index_add_(0, faces.reshape(-1), face_area_share)

                valid = torch.isfinite(shear_magnitude) & torch.isfinite(vertex_areas) & (vertex_areas > 0)
                if not torch.any(valid):
                    continue
                sample_weighted_shear = (shear_magnitude[valid] * vertex_areas[valid]).sum()
                sample_area = vertex_areas[valid].sum()

            total_weighted_shear = (
                sample_weighted_shear
                if total_weighted_shear is None
                else total_weighted_shear.to(sample_weighted_shear.device) + sample_weighted_shear
            )
            total_area = sample_area if total_area is None else total_area.to(sample_area.device) + sample_area

        if total_area is None or total_area <= 0:
            raise ValueError("Could not calculate average shear force: dataset has no valid mesh area.")

        return total_weighted_shear / total_area


def _load_dataset_config(config_path: Path) -> dict:
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    dataset_config = config.get("dataset", {})
    if not isinstance(dataset_config, dict):
        raise TypeError(f"Expected 'dataset' section in {config_path} to be a dictionary.")
    return dataset_config


def _expand_config_path(path: str) -> Path:
    def replace_oc_env(match: re.Match) -> str:
        env_name = match.group(1)
        return os.environ.get(env_name, os.environ.get("USERNAME", ""))

    expanded = re.sub(r"\$\{oc\.env:([^}]+)\}", replace_oc_env, path)
    expanded = os.path.expandvars(expanded)
    return Path(expanded).expanduser()


def _resolve_dataset_path(dataset_config: dict) -> Path:
    base_path = dataset_config.get("base_path")
    if base_path is None:
        raise ValueError("dataset.base_path is missing from the config.")

    base_path = _expand_config_path(str(base_path))
    processed_folder = str(dataset_config.get("processed_folder") or "")
    return base_path / processed_folder


def _resolve_val_files_path(dataset_config: dict, config_path: Path) -> Path:
    order_file = dataset_config.get("val_files_path")
    if order_file is None:
        raise ValueError("dataset.val_files_path is missing from the config.")

    order_file_path = _expand_config_path(str(order_file))
    if not order_file_path.is_absolute():
        order_file_path = config_path.parent / order_file_path
    return order_file_path


def _ordered_sample_paths(dataset_path: Path, order_file: Path) -> list[Path]:
    sample_paths = []
    with open(order_file) as f:
        sample_names = []
        for line in f:
            sample_name = line.split("#", 1)[0].strip()
            if not sample_name:
                continue
            if sample_name.startswith("-"):
                sample_name = sample_name[1:].strip()
            sample_names.append(sample_name.strip("'\"").removesuffix(".pt"))

    for sample_name in sample_names:
        direct_path = dataset_path / f"{sample_name}.pt"
        nested_path = dataset_path / sample_name / "data.pt"
        if direct_path.is_file():
            sample_paths.append(direct_path)
        elif nested_path.is_file():
            sample_paths.append(nested_path)

    return sample_paths


def _load_samples(sample_paths: list[Path]):
    for sample_path in sample_paths:
        yield torch.load(sample_path, map_location="cpu", weights_only=False)


def main():
    print("Start")
    config_path = MAIN_CONFIG_PATH
    dataset_config = _load_dataset_config(config_path)
    dataset_path = _resolve_dataset_path(dataset_config)
    order_file = _resolve_val_files_path(dataset_config, config_path)

    sample_paths = _ordered_sample_paths(dataset_path, order_file)
    if not sample_paths:
        raise FileNotFoundError(f"No samples from {order_file} were found in {dataset_path}.")

    value = ShearForce(n_per_sqm=1.0).calculate_average_newton_per_sqm(_load_samples(sample_paths))
    print(value.item())


if __name__ == "__main__":
    main()
