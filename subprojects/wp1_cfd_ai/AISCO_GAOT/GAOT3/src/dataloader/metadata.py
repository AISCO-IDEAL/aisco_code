"""Utility functions for reading the datasets."""

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class Metadata:
    periodic: bool
    group_u: str
    group_c: str | None
    group_x: str
    type: Literal["gaot"]
    fix_x: bool
    domain_x: tuple[Sequence[float], Sequence[float]]
    domain_t: tuple[int, int] | None
    active_variables: Sequence[int] | None  # Index of variables in input/output
    chunked_variables: Sequence[int] | None  # Index of variable groups
    num_variable_chunks: int  # Number of variable chunks
    signed: dict[str, bool | Sequence[bool | None] | None]
    names: dict[str, Sequence[str | None]]
    global_mean: Sequence[float]
    global_std: Sequence[float]


DATASET_METADATA = {
    "incompressible_fluids/euler_of": Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        domain_x=(
            [-4.996915817260742, -2.8957769870758057, -1.5078723430633545],
            [4.699999809265137, 2.501098394393921, 3.508775472640991],
        ),  # x=flow, y=lateral, z=vertical (meters)
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False], "c": [None]},
        names={"u": ["$p$"], "c": [None]},
        global_mean=[-0.39602479338645935],
        global_std=[0.7298364639282227],
    ),
    "incompressible_fluids/euler": Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        domain_x=(
            [-4.996915340423584, -2.75, -1.5078723430633545],
            [4.699999809265137, 2.5010437965393066, 2.0490450859069824],
        ),  # x=flow, y=lateral, z=vertical (meters)
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False], "c": [None]},
        names={"u": ["$p$"], "c": [None]},
        global_mean=[-0.23824286460876465],
        global_std=[0.3600948452949524],
    ),
    "incompressible_fluids/united_all_cp_no_of": Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        domain_x=(
            [-4.996915340423584, -2.75, -1.5078723430633545],
            [4.699999809265137, 2.5010437965393066, 2.0490450859069824],
        ),  # x=flow, y=lateral, z=vertical (meters)
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False], "c": [None]},
        names={"u": ["$p$"], "c": [None]},
        global_mean=[-0.18390923738479614],
        global_std=[0.2756047248840332],
    ),
    "incompressible_fluids/of_only_clean": Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        domain_x=(
            [-3.771361827850342, -2.1578757762908936, 0.10000000149011612],
            [4.699999809265137, 2.501098394393921, 3.181083917617798],
        ),  # x=flow, y=lateral, z=vertical (meters); 135 strict-QA-pass OF-only samples
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False], "c": [None]},
        names={"u": ["$p$"], "c": [None]},
        global_mean=[-0.36467502041657324],
        global_std=[0.6112995356484017],
    ),
    "incompressible_fluids/united_clean_datasets": Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        domain_x=(
            [-4.996915340423584, -2.75, -1.5078723430633545],
            [4.699999809265137, 2.501098394393921, 3.181083917617798],
        ),  # x=flow, y=lateral, z=vertical (meters); strict-QA clean OF (135/301 kept)
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False], "c": [None]},
        names={"u": ["$p$"], "c": [None]},
        global_mean=[-0.20199349095393843],
        global_std=[0.2992259276674006],
    ),
    "incompressible_fluids/united_all_datasets": Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        domain_x=(
            [-4.996915817260742, -2.8957769870758057, -1.5078723430633545],
            [4.699999809265137, 2.501098394393921, 3.508775472640991],
        ),  # x=flow, y=lateral, z=vertical (meters)
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False], "c": [None]},
        names={"u": ["$p$"], "c": [None]},
        global_mean=[-0.23023323714733124],
        global_std=[0.409395694732666],
    ),
    "incompressible_fluids/euler_drag": Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        domain_x=([-5.0, -1.3, -1.2], [4.7, 1.7, 1.6]),  # meters, x-negated so nose at -x matching DrivAer
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False], "c": [None]},
        names={"u": ["$p$"], "c": [None]},
        global_mean=[-63.66019058227539],
        global_std=[54.39186477661133],
    ),
    "incompressible_fluids/only_of": Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        domain_x=(
            [-4.996915817260742, -2.8957769870758057, 0.09999565035104752],
            [4.699999809265137, 2.501098394393921, 3.508775472640991],
        ),  # meters, x-negated so nose at -x matching DrivAer
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False], "c": [None]},
        names={"u": ["$p$"], "c": [None]},
        global_mean=[-0.578277051448822],
        global_std=[0.9675652384757996],
    ),
    # PLY-based DrivAer (multi-head: cd only, force=NaN), z-up after y↔z swap in preprocessing
    "incompressible_fluids/drivaer_drag": Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        domain_x=([-0.9, -1.1, -0.35], [4.1, 1.1, 1.3]),  # x=flow, y=lateral, z=vertical (meters)
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False], "c": [None]},
        names={"u": ["$p$"], "c": [None]},
        global_mean=[0.0],
        global_std=[1.0],
    ),
    # Blended Wing Body: VTK with cp field, CD/CL targets; chord-normalized (x=flow, y=span, z=thickness)
    "incompressible_fluids/blended_wing": Metadata(
        periodic=False,
        group_u="u",
        group_c="c",
        group_x="x",
        type="gaot",
        domain_x=([0.0, -0.73, -0.10], [1.0, 0.73, 0.10]),  # chord-normalized, x=flow, y=span, z=thickness
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False], "c": [None]},
        names={"u": ["$c_p$"], "c": [None]},
        global_mean=[0.0],
        global_std=[1.0],
    ),
    "incompressible_fluids/drivaernet_pressure": Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        domain_x=([-1.16, -1.20, 0.0], [4.21, 1.19, 1.77]),
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False], "c": [None]},
        names={"u": ["$p$"], "c": [None]},
        global_mean=[-93.4105],  # -94.5
        global_std=[120.7879],  # 117.25
    ),
    "incompressible_fluids/drivaernet_shearstress": Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        domain_x=([-1.16, -1.20, 0.0], [4.21, 1.19, 1.77]),
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False], "c": [None]},
        names={"u": ["$p$"], "c": [None]},
        global_mean=[-0.6717, 0.0364, -0.0846],
        global_std=[0.8199, 0.4510, 0.7811],
    ),
    "incompressible_fluids/nasa_crm": Metadata(
        periodic=False,
        group_u="u",
        group_c="c",
        group_x="x",
        type="gaot",
        domain_x=([2.3495, -29.460142, 2.3101413], [66.744965, 29.460142, 8.833843]),
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False, False, False, False], "c": [False, False]},
        names={"u": ["$p$", "sfc_x", "sfc_y", "sfc_z"], "c": ["Mach", "AOA"]},
        global_mean=[-3.3177e-02, 1.4710e-03, 6.4260e-06, -2.2570e-06],
        global_std=[0.3108, 0.0010, 0.0005, 0.0007],
    ),
    "incompressible_fluids/nasa_crm_pressure": Metadata(
        periodic=False,
        group_u="u",
        group_c="c",
        group_x="x",
        type="gaot",
        domain_x=([2.3495, -29.460142, 2.3101413], [66.744965, 29.460142, 8.833843]),
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False], "c": [False, False]},
        names={"u": ["$p$"], "c": ["Mach", "AOA"]},
        global_mean=[-3.3177e-02],
        global_std=[0.3108],
    ),
    "incompressible_fluids/nasa_crm_sfc": Metadata(
        periodic=False,
        group_u="u",
        group_c="c",
        group_x="x",
        type="gaot",
        domain_x=([2.3495, -29.460142, 2.3101413], [66.744965, 29.460142, 8.833843]),
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False, False, False], "c": [False, False]},
        names={"u": ["sfc_x", "sfc_y", "sfc_z"], "c": ["Mach", "AOA"]},
        global_mean=[1.4710e-03, 6.4260e-06, -2.2570e-06],
        global_std=[0.0010, 0.0005, 0.0007],
    ),
    "incompressible_fluids/drivaerml_pressure": Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        domain_x=([-0.943, -1.14, -0.318], [4.14, 1.14, 1.25]),
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False], "c": [None]},
        names={"u": ["$p$"], "c": [None]},
        global_mean=[-0.3046],
        global_std=[0.3560],
    ),
    "incompressible_fluids/drivaerml_wss": Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        domain_x=([-0.943, -1.14, -0.318], [4.14, 1.14, 1.25]),
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False, False, False], "c": [None]},
        names={"u": ["wss_x", "wss_y", "wss_z"], "c": [None]},
        global_mean=[-1.2049, 0.0015, -0.0724],
        global_std=[2.0773, 1.3518, 1.1098],
    ),
    "incompressible_fluids/drivaerml": Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        domain_x=([-0.943, -1.14, -0.318], [4.14, 1.14, 1.25]),
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False, False, False, False], "c": [None]},
        names={"u": ["$p$", "wss_x", "wss_y", "wss_z"], "c": [None]},
        global_mean=[-0.3046, -1.2049, 0.0015, -0.0724],
        global_std=[0.3560, 2.0773, 1.3518, 1.1098],
    ),
    "incompressible_fluids/euler_pressure": Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        domain_x=([-4.7, -1.16, -1.00], [5.00, 1.66, 1.65]),
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False], "c": [None]},
        names={"u": ["$p$"], "c": [None]},
        global_mean=[-58.516],
        global_std=[97.740],
    ),
    "incompressible_fluids/ahmed_pressure": Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        domain_x=([-1.19, -0.19, 0.05], [0.001, 0.19, 0.31]),
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False], "c": [None]},
        names={"u": ["$p$"], "c": [None]},
        global_mean=[-0.0795],
        global_std=[0.1582],
    ),
    "incompressible_fluids/drivaer_pressure": Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        domain_x=([-0.93, -1.09, -0.32], [4.00, 1.09, 1.25]),
        domain_t=None,
        fix_x=False,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False], "c": [None]},
        names={"u": ["$p$"], "c": [None]},
        global_mean=[-231.697],
        global_std=[269.949],
    ),
}


def get_dataset_metadata(dataset_config: dict) -> Metadata:
    """Return static metadata or infer it for newly assembled PyG datasets."""
    metaname = dataset_config["metaname"]
    if metaname in DATASET_METADATA:
        return DATASET_METADATA[metaname]

    logger = logging.getLogger(__name__)
    metadata_file = _metadata_file_path(dataset_config)
    force_recompute = dataset_config.get(
        "force_recompute_metadata",
        dataset_config.get("force_recompute_stats", False),
    )

    if not force_recompute:
        cached = _load_dynamic_metadata(metadata_file)
        if cached is not None:
            DATASET_METADATA[metaname] = cached
            logger.info("Loaded dynamic dataset metadata from %s", metadata_file)
            return cached

    logger.info(
        "Dataset metadata %r is not registered; inferring it from %s",
        metaname,
        dataset_config["base_path"],
    )
    metadata = _infer_dynamic_metadata(dataset_config)
    DATASET_METADATA[metaname] = metadata
    _save_dynamic_metadata(metadata_file, metadata)
    logger.info("Saved dynamic dataset metadata to %s", metadata_file)
    return metadata


def _metadata_file_path(dataset_config: dict) -> str:
    return os.path.join(dataset_config["base_path"], _metadata_file_name(dataset_config))


def _metadata_file_name(dataset_config: dict) -> str:
    prefixes = _normalized_dataset_prefixes(dataset_config.get("allowed_dataset_prefixes"))
    if prefixes is None:
        return f"{dataset_config['name']}_metadata.json"
    return f"{dataset_config['name']}_prefixes_{'_'.join(prefixes)}_metadata.json"


def _load_dynamic_metadata(metadata_file: str) -> Metadata | None:
    if not os.path.isfile(metadata_file):
        return None
    try:
        with open(metadata_file, encoding="utf-8") as handle:
            payload = json.load(handle)
        return Metadata(
            periodic=payload.get("periodic", False),
            group_u=payload.get("group_u", "u"),
            group_c=payload.get("group_c"),
            group_x=payload.get("group_x", "x"),
            type="gaot",
            fix_x=payload.get("fix_x", False),
            domain_x=(payload["domain_x"][0], payload["domain_x"][1]),
            domain_t=payload.get("domain_t"),
            active_variables=payload.get("active_variables"),
            chunked_variables=payload.get("chunked_variables"),
            num_variable_chunks=payload.get("num_variable_chunks", 1),
            signed=payload.get("signed", {"u": [False], "c": [None]}),
            names=payload.get("names", {"u": ["$c_p$"], "c": [None]}),
            global_mean=payload.get("global_mean", [0.0]),
            global_std=payload.get("global_std", [1.0]),
        )
    except Exception as exc:  # noqa: BLE001 - stale cache should not block a fresh scan
        logging.getLogger(__name__).warning("Ignoring invalid metadata cache %s: %s", metadata_file, exc)
        return None


def _save_dynamic_metadata(metadata_file: str, metadata: Metadata) -> None:
    payload = {
        "periodic": metadata.periodic,
        "group_u": metadata.group_u,
        "group_c": metadata.group_c,
        "group_x": metadata.group_x,
        "type": metadata.type,
        "fix_x": metadata.fix_x,
        "domain_x": [list(metadata.domain_x[0]), list(metadata.domain_x[1])],
        "domain_t": metadata.domain_t,
        "active_variables": metadata.active_variables,
        "chunked_variables": metadata.chunked_variables,
        "num_variable_chunks": metadata.num_variable_chunks,
        "signed": metadata.signed,
        "names": metadata.names,
        "global_mean": metadata.global_mean,
        "global_std": metadata.global_std,
    }
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    tmp_file = f"{metadata_file}.{os.getpid()}.tmp"
    with open(tmp_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
        handle.write("\n")
    os.replace(tmp_file, metadata_file)


def _infer_dynamic_metadata(dataset_config: dict) -> Metadata:
    order_file = _order_file_path(dataset_config)
    if not os.path.isfile(order_file):
        raise FileNotFoundError(
            f"Metadata inference needs the dataset order file, but it does not exist: {order_file}"
        )

    with open(order_file, encoding="utf-8") as handle:
        sample_names = [_normalize_sample_name(line.strip()) for line in handle if line.strip()]
    sample_names = _filter_names_by_allowed_prefixes(sample_names, dataset_config.get("allowed_dataset_prefixes"))
    if not sample_names:
        raise ValueError(
            "allowed_dataset_prefixes selected no samples while inferring metadata: "
            f"{dataset_config.get('allowed_dataset_prefixes')}"
        )

    processed_dir = _processed_dir(dataset_config)
    pos_min = None
    pos_max = None
    target_dim = None
    usable_samples = 0
    missing_samples = []

    for sample_name in sample_names:
        sample_path = _sample_path(processed_dir, sample_name)
        if sample_path is None:
            missing_samples.append(sample_name)
            continue

        data = torch.load(sample_path, map_location="cpu", weights_only=False)
        if not hasattr(data, "pos") or data.pos is None:
            raise ValueError(f"Sample has no pos tensor and cannot be used for metadata: {sample_path}")
        pos = torch.as_tensor(data.pos).detach().cpu().float()
        sample_min = pos.amin(dim=0)
        sample_max = pos.amax(dim=0)
        pos_min = sample_min if pos_min is None else torch.minimum(pos_min, sample_min)
        pos_max = sample_max if pos_max is None else torch.maximum(pos_max, sample_max)
        target_dim = target_dim or _infer_target_dim(data, dataset_config)
        usable_samples += 1

    if missing_samples:
        logging.getLogger(__name__).warning(
            "Skipped %s missing samples while inferring metadata. First missing samples: %s",
            len(missing_samples),
            missing_samples[:5],
        )
    if usable_samples == 0 or pos_min is None or pos_max is None:
        raise ValueError(f"No usable .pt samples found under {processed_dir} while inferring metadata.")

    target_dim = target_dim or 1
    target_names = ["$c_p$"] if target_dim == 1 else [f"target_{index}" for index in range(target_dim)]
    return Metadata(
        periodic=False,
        group_u="u",
        group_c=None,
        group_x="x",
        type="gaot",
        fix_x=False,
        domain_x=(pos_min.tolist(), pos_max.tolist()),
        domain_t=None,
        active_variables=None,
        chunked_variables=None,
        num_variable_chunks=1,
        signed={"u": [False] * target_dim, "c": [None]},
        names={"u": target_names, "c": [None]},
        global_mean=[0.0] * target_dim,
        global_std=[1.0] * target_dim,
    )


def _order_file_path(dataset_config: dict) -> str:
    order_file_name = dataset_config.get("order_file_name")
    if order_file_name:
        return os.path.join(dataset_config["base_path"], order_file_name)
    return os.path.join(
        dataset_config["base_path"],
        f"order_{dataset_config.get('processed_folder') or ''}.txt",
    )


def _processed_dir(dataset_config: dict) -> str:
    return os.path.join(dataset_config["base_path"], dataset_config.get("processed_folder") or "")


def _sample_path(processed_dir: str, sample_name: str) -> str | None:
    direct_path = os.path.join(processed_dir, f"{sample_name}.pt")
    if os.path.isfile(direct_path):
        return direct_path
    nested_path = os.path.join(processed_dir, sample_name, "data.pt")
    if os.path.isfile(nested_path):
        return nested_path
    return None


def _infer_target_dim(data, dataset_config: dict) -> int:
    for attr_name in ("x", "y", "cp", "pressure"):
        value = getattr(data, attr_name, None)
        if isinstance(value, torch.Tensor):
            if value.dim() <= 1:
                return 1
            return int(value.shape[-1])
    active_heads = dataset_config.get("active_heads") or []
    return max(1, len(active_heads))


def _filter_names_by_allowed_prefixes(sample_names: list[str], raw_prefixes) -> list[str]:
    prefixes = _normalized_dataset_prefixes(raw_prefixes)
    if prefixes is None:
        return sample_names
    prefix_set = set(prefixes)
    return [name for name in sample_names if _dataset_prefix_from_sample_name(name) in prefix_set]


def _normalized_dataset_prefixes(raw_prefixes) -> list[str] | None:
    if raw_prefixes is None:
        return None
    if isinstance(raw_prefixes, (str, int)):
        raw_prefixes = [raw_prefixes]
    prefixes = {str(prefix).strip() for prefix in raw_prefixes if str(prefix).strip()}
    if not prefixes:
        return None
    invalid = sorted(prefix for prefix in prefixes if not prefix.isdigit())
    if invalid:
        raise ValueError(f"allowed_dataset_prefixes must contain only digits, got: {invalid}")
    return sorted(prefixes, key=int)


def _dataset_prefix_from_sample_name(name: str) -> str | None:
    sample_name = _normalize_sample_name(name)
    if not sample_name.startswith("sample_"):
        return None
    suffix = sample_name.removeprefix("sample_")
    if len(suffix) <= 5 or not suffix.isdigit():
        return None
    prefix = suffix[:-5]
    return prefix if prefix else None


def _normalize_sample_name(name: str) -> str:
    return name.strip().removesuffix(".pt")
