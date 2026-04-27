import json
import os
from datetime import datetime, timezone
from typing import Any

import torch

AIR_DENSITY = 1.225
A_SEA_LEVEL = 340.294
COND_EPS = 1e-6

V_INF_EULER = 25.0
V_INF_DRIVAER = 38.89
V_INF_AHMED = 1.0
PREPROCESS_CONTRACT_VERSION = "v0.7.0"

DEFAULT_OUTPUT_DIRS = {
    "euler": "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler_cp_full",
    "drivaer": "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_drivaer_cp_downsampled",
    "ahmed": "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_ahmed_cp_full",
    "blended_wing": "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_blended_wing_cp_full_filtered",
}

CANONICAL_CONTRACT: dict[str, Any] = {
    "required_keys": [
        "pos",
        "c",
        "face",
        "x",
        "cp",
        "pressure",
        "wall_shear",
        "cd",
        "force",
        "v_inf",
        "geom_extent",
        "geom_aspect",
        "conditioning",
        "filename",
        "source_dataset",
        "source_case_id",
    ],
    "tensor_shapes": {
        "pos": (None, 3),
        "c": (None, 3),
        "face": (3, None),
        "x": (None, 1),
        "cp": (None, 1),
        "pressure": (None, 1),
        "wall_shear": (None, 3),
        "cd": (1,),
        "force": (1,),
        "v_inf": (1,),
        "geom_extent": (3,),
        "geom_aspect": (3,),
        "conditioning": (1, 7),
    },
}


def compute_pressure_coefficient(
    pressure: torch.Tensor,
    v_inf: float,
    rho: float = AIR_DENSITY,
) -> torch.Tensor:
    dynamic_pressure = max(0.5 * rho * (v_inf**2), COND_EPS)
    return pressure.to(dtype=torch.float32) / dynamic_pressure


def compute_geom_conditioning(pos: torch.Tensor, v_inf: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    extent = pos.max(dim=0).values - pos.min(dim=0).values
    max_extent = torch.clamp(extent.max(), min=COND_EPS)
    aspect = extent / max_extent
    conditioning = torch.cat(
        [
            torch.tensor([v_inf], dtype=torch.float32),
            extent.to(dtype=torch.float32),
            aspect.to(dtype=torch.float32),
        ],
        dim=0,
    ).unsqueeze(0)
    return extent.to(dtype=torch.float32), aspect.to(dtype=torch.float32), conditioning


def _to_col_vector(values: torch.Tensor, n_rows: int, key_name: str) -> torch.Tensor:
    tensor = values.to(dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.shape != (n_rows, 1):
        raise ValueError(
            f"Canonical contract violation for '{key_name}': expected shape ({n_rows}, 1), got {tuple(tensor.shape)}"
        )
    return tensor


def _to_matrix(values: torch.Tensor, n_rows: int, n_cols: int, key_name: str) -> torch.Tensor:
    tensor = values.to(dtype=torch.float32)
    if tensor.shape != (n_rows, n_cols):
        raise ValueError(
            f"Canonical contract violation for '{key_name}': expected shape ({n_rows}, {n_cols}), got {tuple(tensor.shape)}"
        )
    return tensor


def apply_canonical_contract(  # noqa: PLR0912, PLR0915 - centralized contract application keeps adapters thin and consistent.
    data: Any,
    *,
    source_dataset: str,
    source_case_id: str,
    filename: str,
    geom_name: str = "",
    flow_conditions: dict[str, float] | None = None,
    preprocess_version: str = PREPROCESS_CONTRACT_VERSION,
) -> Any:
    """Populate missing canonical keys and enforce x/cp alignment.

    Args:
        data: PyG Data object to normalize to canonical schema.
        source_dataset: Dataset identifier (`euler`, `drivaer`, `ahmed`, `blended_wing`).
        source_case_id: Dataset-native sample identifier.
        filename: Stable filename stored on the sample.
        geom_name: Optional geometry family identifier.
        flow_conditions: Optional scalar-flow metadata.
        preprocess_version: Contract version string.

    Returns:
        The same Data object with canonical keys populated.
    """
    if not hasattr(data, "pos") or data.pos is None:
        raise ValueError(
            "Canonical contract violation: missing required key 'pos'. Add vertex coordinates before canonicalization."
        )

    data.pos = data.pos.to(dtype=torch.float32)
    n_rows = int(data.pos.shape[0])

    if hasattr(data, "cp") and data.cp is not None:
        cp_tensor = _to_col_vector(data.cp, n_rows, "cp")
    elif hasattr(data, "x") and data.x is not None:
        cp_tensor = _to_col_vector(data.x, n_rows, "x")
    else:
        cp_tensor = torch.full((n_rows, 1), float("nan"), dtype=torch.float32)

    data.cp = cp_tensor
    data.x = cp_tensor.clone()

    if hasattr(data, "pressure") and data.pressure is not None:
        data.pressure = _to_col_vector(data.pressure, n_rows, "pressure")
    else:
        data.pressure = torch.full((n_rows, 1), float("nan"), dtype=torch.float32)

    if hasattr(data, "wall_shear") and data.wall_shear is not None:
        data.wall_shear = _to_matrix(data.wall_shear, n_rows, 3, "wall_shear")
    else:
        data.wall_shear = torch.full((n_rows, 3), float("nan"), dtype=torch.float32)

    if hasattr(data, "c") and data.c is not None:
        data.c = _to_matrix(data.c, n_rows, 3, "c")
    else:
        data.c = torch.full((n_rows, 3), float("nan"), dtype=torch.float32)

    if not hasattr(data, "face") or data.face is None:
        data.face = torch.empty((3, 0), dtype=torch.long)
    else:
        data.face = data.face.to(dtype=torch.long)
        if data.face.ndim != 2:
            raise ValueError(f"Canonical contract violation for 'face': expected 2D tensor, got {data.face.ndim}D")
        if data.face.shape[0] != 3 and data.face.shape[1] == 3:
            data.face = data.face.t().contiguous()
        if data.face.shape[0] != 3:
            raise ValueError(f"Canonical contract violation for 'face': expected first dim 3, got {tuple(data.face.shape)}")

    if hasattr(data, "cd") and data.cd is not None:
        data.cd = data.cd.to(dtype=torch.float32).reshape(1)
    else:
        data.cd = torch.tensor([float("nan")], dtype=torch.float32)

    if hasattr(data, "force") and data.force is not None:
        data.force = data.force.to(dtype=torch.float32).reshape(1)
    else:
        data.force = torch.tensor([float("nan")], dtype=torch.float32)

    if hasattr(data, "v_inf") and data.v_inf is not None:
        data.v_inf = data.v_inf.to(dtype=torch.float32).reshape(1)
    else:
        data.v_inf = torch.tensor([float("nan")], dtype=torch.float32)

    v_inf_scalar = float(data.v_inf[0].item())
    if hasattr(data, "geom_extent") and data.geom_extent is not None:
        data.geom_extent = data.geom_extent.to(dtype=torch.float32).reshape(3)
    elif torch.isfinite(data.v_inf).all():
        data.geom_extent, _, _ = compute_geom_conditioning(data.pos, v_inf_scalar)
    else:
        data.geom_extent = torch.full((3,), float("nan"), dtype=torch.float32)

    if hasattr(data, "geom_aspect") and data.geom_aspect is not None:
        data.geom_aspect = data.geom_aspect.to(dtype=torch.float32).reshape(3)
    elif torch.isfinite(data.v_inf).all():
        _, data.geom_aspect, _ = compute_geom_conditioning(data.pos, v_inf_scalar)
    else:
        data.geom_aspect = torch.full((3,), float("nan"), dtype=torch.float32)

    if hasattr(data, "conditioning") and data.conditioning is not None:
        data.conditioning = data.conditioning.to(dtype=torch.float32).reshape(1, 7)
    elif torch.isfinite(data.v_inf).all():
        _, _, data.conditioning = compute_geom_conditioning(data.pos, v_inf_scalar)
    else:
        data.conditioning = torch.full((1, 7), float("nan"), dtype=torch.float32)

    data.filename = filename
    data.source_dataset = source_dataset
    data.source_case_id = source_case_id
    data.geom_name = geom_name
    data.flow_conditions = flow_conditions or {}
    data.preprocess_version = preprocess_version
    return data


def validate_canonical_contract(data: Any) -> None:
    """Validate canonical contract keys and core shape rules.

    Args:
        data: PyG Data object to validate.

    Returns:
        None. Raises ValueError on contract violations.
    """
    missing_keys = [key for key in CANONICAL_CONTRACT["required_keys"] if not hasattr(data, key) or getattr(data, key) is None]
    if missing_keys:
        raise ValueError(
            "Canonical contract violation: missing required key(s) "
            f"{missing_keys}. Please call apply_canonical_contract before saving this sample."
        )

    expected_shapes = CANONICAL_CONTRACT["tensor_shapes"]
    for key, shape_spec in expected_shapes.items():
        value = getattr(data, key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Canonical contract violation for '{key}': expected torch.Tensor, got {type(value).__name__}. "
                "Please write tensors for all numeric contract fields."
            )
        if value.ndim != len(shape_spec):
            raise ValueError(f"Canonical contract violation for '{key}': expected {len(shape_spec)}D tensor, got {value.ndim}D.")
        for axis, dim in enumerate(shape_spec):
            if dim is not None and value.shape[axis] != dim:
                raise ValueError(
                    f"Canonical contract violation for '{key}': expected shape spec {shape_spec}, got {tuple(value.shape)}."
                )

    if not torch.allclose(data.x, data.cp, equal_nan=True):
        raise ValueError("Canonical contract violation: x and cp differ. For Cp training, set x to the same values as cp.")

    if data.source_dataset not in DEFAULT_OUTPUT_DIRS:
        raise ValueError(
            f"Canonical contract violation: unsupported source_dataset '{data.source_dataset}'. "
            f"Use one of {list(DEFAULT_OUTPUT_DIRS.keys())}."
        )

    if not isinstance(data.filename, str) or not data.filename:
        raise ValueError("Canonical contract violation: 'filename' must be a non-empty string.")

    if not isinstance(data.source_case_id, str) or not data.source_case_id:
        raise ValueError("Canonical contract violation: 'source_case_id' must be a non-empty string.")


def write_dataset_manifest(
    *,
    output_dir: str,
    dataset_name: str,
    processed_count: int,
    total_candidates: int,
    failed_count: int,
    contract_version: str,
    source_root: str,
) -> None:
    """Write per-dataset preprocessing manifest.

    Args:
        output_dir: Dataset output directory where manifest will be written.
        dataset_name: Name of the source dataset.
        processed_count: Number of successfully available samples.
        total_candidates: Number of candidate samples considered.
        failed_count: Number of failed/skipped samples.
        contract_version: Canonical contract version.
        source_root: Root source path used for this dataset.

    Returns:
        None.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest = {
        "dataset": dataset_name,
        "processed_count": int(processed_count),
        "total_candidates": int(total_candidates),
        "failed_count": int(failed_count),
        "contract_version": contract_version,
        "source_root": source_root,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - keep mypy-compatible timezone alias.
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
