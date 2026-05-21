"""Backend service for the Latent Space Explorer GUI page.

Handles checkpoint loading, latent-space metadata extraction, mesh extraction
via marching cubes, and GLB serialisation.  All heavyweight model loading is
done once and cached; subsequent requests for the same checkpoint reuse the
cached model.

This file intentionally does NOT import ``latent_rollout_gif`` because that
module imports ``pyrender`` at module level, which requires EGL and would crash
on GUI server startup.  Instead the required WP4 model-loading helpers are
replicated here (they are ~80 lines of pure-stdlib + torch code).
"""

from __future__ import annotations

import glob
import importlib.util
import json
import re
import sys
import types
from collections import OrderedDict
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import torch
import trimesh
from torch import Tensor

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

# parents[4]: shared → v3 → wp7_gui → subprojects → AISCO_code
_REPO_ROOT = Path(__file__).resolve().parents[4]
_MODELS_DIR = _REPO_ROOT / "subprojects" / "wp4_aeroshell_optimizer" / "v2" / "src" / "models"

# Prism bounds in mm — must match wp2 assembler definition.
_PRISM_X = (-2900.0, 2900.0)
_PRISM_Y = (-1150.0, 1150.0)
_PRISM_Z = (0.0, 1650.0)

# ---------------------------------------------------------------------------
# Pretrained checkpoint registry
# ---------------------------------------------------------------------------

# Set this path once the 8 pretrained subfolders are ready.
# Each subfolder must contain a .pth checkpoint file (anywhere in its tree) and
# optionally .glb assembly files and a model_info_dict.json.
# Set to None to leave the GUI selector active but not yet backed by real files.
PRETRAINED_ROOT: Path | None = None

# Maps (wheels, solar, drag) → subfolder name within PRETRAINED_ROOT.
# wheels: 3 or 4 · solar: True = on · drag: True = on
_PRETRAINED_SUBFOLDER_MAP: dict[tuple[int, bool, bool], str] = {
    (3, True, True): "wheels3_solar_drag",
    (3, True, False): "wheels3_solar_nodrag",
    (3, False, True): "wheels3_nosolar_drag",
    (3, False, False): "wheels3_nosolar_nodrag",
    (4, True, True): "wheels4_solar_drag",
    (4, True, False): "wheels4_solar_nodrag",
    (4, False, True): "wheels4_nosolar_drag",
    (4, False, False): "wheels4_nosolar_nodrag",
}


def resolve_pretrained_path(wheels: int, solar: bool, drag: bool) -> Path:
    """Return the subfolder path for a pretrained (wheels, solar, drag) combination.

    Args:
        wheels: Number of wheels (3 or 4).
        solar: Whether the solar objective is active.
        drag: Whether the drag objective is active.

    Returns:
        Path to the pretrained subfolder inside PRETRAINED_ROOT.

    Raises:
        ValueError: When PRETRAINED_ROOT is not configured or the key is unknown.
        FileNotFoundError: When the resolved subfolder does not exist on disk.
    """
    if PRETRAINED_ROOT is None:
        raise ValueError("PRETRAINED_ROOT is not configured. Set it in subprojects/wp7_gui/v3/shared/latent_explorer.py.")
    key = (wheels, solar, drag)
    if key not in _PRETRAINED_SUBFOLDER_MAP:
        raise ValueError(
            f"Unknown pretrained key: wheels={wheels}, solar={solar}, drag={drag}. "
            "Expected wheels ∈ {3, 4}, solar/drag ∈ {True, False}."
        )
    folder = PRETRAINED_ROOT / _PRETRAINED_SUBFOLDER_MAP[key]
    if not folder.exists():
        raise FileNotFoundError(
            f"Pretrained subfolder not found: {folder}. Ensure the folder exists and contains a .pth checkpoint."
        )
    return folder


# ---------------------------------------------------------------------------
# Security: allowed path roots
# ---------------------------------------------------------------------------

_ALLOWED_ROOTS: list[Path] = [
    Path("/cluster/work/fuge/AISCO"),
    _REPO_ROOT,
]


def is_checkpoint_path_allowed(ckpt_path: Path) -> bool:
    """Return True iff *ckpt_path* resolves to a location inside an allowed root.

    Also accepts paths under PRETRAINED_ROOT when it is configured.

    Args:
        ckpt_path: Candidate checkpoint path (may be relative or unresolved).

    Returns:
        True when the resolved path is under an allowed root, False otherwise.
    """
    try:
        resolved = ckpt_path.resolve()
        roots = _ALLOWED_ROOTS[:]
        if PRETRAINED_ROOT is not None:
            roots.append(PRETRAINED_ROOT)
        return any(resolved.is_relative_to(root) for root in roots)
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

_EPOCH_RE = re.compile(r"_epoch(\d+)")


def _epoch_from_name(name: str) -> int:
    """Extract epoch number from a checkpoint stem; return -1 if not found."""
    match = _EPOCH_RE.search(name)
    return int(match.group(1)) if match else -1


def _find_assembly_glb(ckpt_path: Path) -> Path | None:
    """Find the slot_00_mesh.glb from mesh_assembly_v4/ nearest to the checkpoint epoch.

    Walks up from the checkpoint file looking for a sibling ``mesh_assembly_v4/``
    directory (typically the run root is the grandparent of the .pth file).
    Returns None when no such directory or GLB is found.
    """
    epoch = _epoch_from_name(ckpt_path.stem)
    search = ckpt_path.parent
    for _ in range(3):
        assembly_dir = search / "mesh_assembly_v4"
        if assembly_dir.is_dir():
            break
        search = search.parent
    else:
        # mesh_assembly_v4 not found; fall back to any direct .glb in the hierarchy
        # (e.g. pretrained subfolders that ship a pre-baked assembly.glb).
        fallback = ckpt_path.parent
        for _ in range(3):
            direct = sorted(fallback.glob("*.glb"))
            if direct:
                return direct[-1]
            fallback = fallback.parent
        return None

    available = sorted(assembly_dir.glob("epoch_*/slot_00_mesh.glb"))
    if not available:
        return None
    if epoch < 0:
        return available[-1]
    # Prefer exact match or closest earlier epoch; fall back to earliest.
    candidates = [p for p in available if int(p.parent.name.split("_")[1]) <= epoch]
    return candidates[-1] if candidates else available[0]


def find_last_epoch_checkpoint(path: Path) -> Path:
    """Resolve *path* to the highest-epoch ``.pth`` file.

    The function accepts three input shapes:

    * A direct ``.pth`` file path → returned unchanged.
    * A ``models/`` directory → scans for ``*_epoch*.pth`` and returns the
      file with the highest epoch number.
    * A run-root directory (parent of ``models/``) → recursively searches
      for ``*_epoch*.pth`` files and returns the one with the highest epoch.

    Args:
        path: Path to a .pth file, a models/ directory, or a run root.

    Returns:
        Path to the highest-epoch .pth checkpoint found.

    Raises:
        FileNotFoundError: When no suitable checkpoint is found.
        ValueError: When *path* does not exist.
    """
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")

    if path.is_file():
        if path.suffix != ".pth":
            raise ValueError(f"Expected a .pth file, got: {path}")
        return path

    # Directory search
    candidates = sorted(path.rglob("*_epoch*.pth"), key=lambda p: _epoch_from_name(p.stem))
    if not candidates:
        raise FileNotFoundError(f"No epoch checkpoint (.pth) files found under: {path}")
    return candidates[-1]


# ---------------------------------------------------------------------------
# WP4 model loading (replicated from latent_rollout_gif.py, no pyrender)
# ---------------------------------------------------------------------------


def _load_module_direct(module_name: str, file_path: Path) -> types.ModuleType:
    """Load a .py file directly by path into sys.modules under *module_name*.

    Bypasses the package __init__ chain to avoid triggering uncompiled CUDA
    extensions.  Idempotent: returns the cached module if already loaded.

    Args:
        module_name: Dotted module name to register in sys.modules.
        file_path: Absolute path to the .py source file.

    Returns:
        The loaded module.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]  # type: ignore[return-value]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod  # register before exec to handle circular refs
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _ensure_wp4_model_imports() -> None:
    """Pre-populate sys.modules so that importing ConcatWIREV5 never triggers
    the v2 package __init__.py (which requires uncompiled CUDA extensions).

    Strategy:
    1. Register stub Modules for every parent package.
    2. Register a minimal BaseModel stub (ConcatWIREV5 extends it).
    3. Directly load concat_baseclass, WIRE_v5, concat_WIRE_v5 from source.
    """
    repo_root = str(_REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Parent package stubs
    for pkg in [
        "subprojects.wp4_aeroshell_optimizer",
        "subprojects.wp4_aeroshell_optimizer.v2",
        "subprojects.wp4_aeroshell_optimizer.v2.src",
        "subprojects.wp4_aeroshell_optimizer.v2.src.models",
    ]:
        if pkg not in sys.modules:
            stub = types.ModuleType(pkg)
            stub.__path__ = []  # type: ignore[attr-defined]
            sys.modules[pkg] = stub

    # Minimal model_baseclass stub
    mb_key = "subprojects.wp4_aeroshell_optimizer.v2.src.models.model_baseclass"
    if mb_key not in sys.modules:
        from torch import nn  # noqa: PLC0415

        mb_stub = types.ModuleType(mb_key)

        class _BaseModel(nn.Module):
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                super().__init__()

        mb_stub.BaseModel = _BaseModel  # type: ignore[attr-defined]
        sys.modules[mb_key] = mb_stub

    # Load real model files
    for mod_name, fname in [
        ("subprojects.wp4_aeroshell_optimizer.v2.src.models.concat_baseclass", "concat_baseclass.py"),
        ("subprojects.wp4_aeroshell_optimizer.v2.src.models.WIRE_v5", "WIRE_v5.py"),
        ("subprojects.wp4_aeroshell_optimizer.v2.src.models.concat_WIRE_v5", "concat_WIRE_v5.py"),
    ]:
        _load_module_direct(mod_name, _MODELS_DIR / fname)


def _build_model(info: dict[str, Any], device: torch.device) -> Any:
    """Instantiate a ConcatWIREV5 model from *info* dict and move to *device*.

    Args:
        info: Architecture JSON dict (from Architecture_*.json or model_info_dict.json).
        device: Target torch device.

    Returns:
        Instantiated ConcatWIREV5 model in eval mode on *device*.
    """
    _ensure_wp4_model_imports()
    ConcatWIREV5 = sys.modules["subprojects.wp4_aeroshell_optimizer.v2.src.models.concat_WIRE_v5"].ConcatWIREV5  # type: ignore[attr-defined]
    return ConcatWIREV5(
        z_dim=info["z_dim"],
        hidden_features=info["hidden_features"],
        hidden_layers=info["hidden_layers"],
        coords=info.get("coords", 3),
        out_features=info.get("out_features", 1),
        first_omega_min=info["first_omega_min"],
        first_omega_max=info["first_omega_max"],
        first_scale_min=info["first_scale_min"],
        first_scale_max=info["first_scale_max"],
        hidden_omega_min=info["hidden_omega_min"],
        hidden_omega_max=info["hidden_omega_max"],
        hidden_scale_min=info["hidden_scale_min"],
        hidden_scale_max=info["hidden_scale_max"],
        skip_layer_idx=info.get("skip_layer_idx", 2),
        train_omega=info.get("trainable_omega", False),
        train_scale=info.get("trainable_scale", False),
        use_spectral_norm=info.get("use_spectral_norm", False),
        use_smooth_output=info.get("use_smooth_output", False),
        bbox_center=info.get("bbox_center"),
        bbox_half_extent=float(info.get("bbox_half_extent", 1000.0)),
    ).to(device)


def _find_model_info(ckpt_path: Path) -> Path:
    """Walk up the directory tree from *ckpt_path* to find a model info JSON.

    Checks for ``model_info_dict.json`` and ``Architecture_*.json`` at each
    level, up to 6 levels above the checkpoint file.

    Args:
        ckpt_path: Path to the .pth checkpoint file.

    Returns:
        Path to the model info JSON.

    Raises:
        FileNotFoundError: When no suitable JSON is found within 6 levels.
    """
    search = ckpt_path.parent
    for _ in range(6):
        candidate = search / "model_info_dict.json"
        if candidate.exists():
            return candidate
        arch_matches = glob.glob(str(search / "Architecture_*.json"))
        if arch_matches:
            return Path(arch_matches[0])
        search = search.parent
    raise FileNotFoundError(
        f"Could not find model_info_dict.json or Architecture_*.json in any "
        f"ancestor (up to 6 levels) of {ckpt_path.parent}. "
        f"Ensure the checkpoint is inside a valid run directory."
    )


# ---------------------------------------------------------------------------
# FIFO model cache
# ---------------------------------------------------------------------------

# Maps resolved checkpoint path string → (model, z_dim, z_radius, z_pretrained_list | None)
_MODEL_CACHE: OrderedDict[str, tuple[Any, int, float, list[list[float]] | None]] = OrderedDict()
_CACHE_MAX = 5


def _cache_put(key: str, value: tuple[Any, int, float, list[list[float]] | None]) -> None:
    """Insert *value* at *key* in the FIFO cache, evicting oldest if over limit."""
    if key in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(key)
    _MODEL_CACHE[key] = value
    while len(_MODEL_CACHE) > _CACHE_MAX:
        _MODEL_CACHE.popitem(last=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_checkpoint_info(ckpt_path: Path, device: torch.device) -> dict[str, Any]:
    """Load (or retrieve from cache) a checkpoint and return latent metadata.

    The model is stored in a FIFO cache keyed by the resolved path string.
    Subsequent calls for the same path return the cached model.

    Args:
        ckpt_path: Path to a ``.pth`` checkpoint file.
        device: Torch device to load the model onto (e.g. ``torch.device("cuda")``).

    Returns:
        Dictionary with keys:
            ``z_dim`` (int): Dimension of the latent vector.
            ``z_radius`` (float): Mean radius of the pretrained latent anchors.
            ``z_pretrained`` (list[list[float]] | None): List of [z0, z1] pairs,
                or None if the checkpoint has no ``z_pretrained`` tensor.

    Raises:
        FileNotFoundError: When the checkpoint or its model info JSON is missing.
        ValueError: When the checkpoint path is not under an allowed root.
    """
    if not is_checkpoint_path_allowed(ckpt_path):
        raise ValueError(
            f"Checkpoint path is not in an allowed directory: {ckpt_path}. Allowed roots: {[str(r) for r in _ALLOWED_ROOTS]}"
        )

    cache_key = str(ckpt_path.resolve())
    if cache_key in _MODEL_CACHE:
        _cached_model, cached_z_dim, cached_z_radius, cached_z_pretrained = _MODEL_CACHE[cache_key]
        return {"z_dim": cached_z_dim, "z_radius": cached_z_radius, "z_pretrained": cached_z_pretrained}

    info_path = _find_model_info(ckpt_path)
    with open(info_path, encoding="utf-8") as fh:
        arch_info = json.load(fh)

    model = _build_model(arch_info, device)
    raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    model.load_state_dict(state_dict)
    model.eval()

    z_pretrained: list[list[float]] | None = None
    z_radius = 2.0
    if isinstance(raw, dict) and "z_pretrained" in raw:
        Z: Tensor = raw["z_pretrained"].to(device)
        z_pretrained = [[float(Z[i, 0].item()), float(Z[i, 1].item())] for i in range(Z.shape[0])]
        if Z.shape[0] > 1:
            radii = [float(torch.norm(Z[i]).item()) for i in range(1, Z.shape[0])]
            z_radius = float(sum(radii) / len(radii))

    z_dim: int = int(arch_info["z_dim"])
    _cache_put(cache_key, (model, z_dim, z_radius, z_pretrained))

    return {"z_dim": z_dim, "z_radius": z_radius, "z_pretrained": z_pretrained}


def extract_mesh_glb(
    ckpt_path: Path,
    z0: float,
    z1: float,
    resolution: int,
    device: torch.device,
    bbox_padding: float = 0.0,
) -> bytes:
    """Extract the aeroshell mesh at latent point *(z0, z1)* and return GLB bytes.

    The model must have been loaded (or cached) via :func:`load_checkpoint_info`
    before calling this function.  If the cache does not contain the model, it
    is loaded on-demand.

    Args:
        ckpt_path: Path to the checkpoint file.
        z0: First latent coordinate.
        z1: Second latent coordinate.
        resolution: Marching cubes grid resolution (e.g. 35).
        device: Torch device for inference.
        bbox_padding: Optional padding (mm) added to the prism bounding box
            on all sides except Z-bottom.

    Returns:
        GLB-encoded bytes of the extracted mesh.

    Raises:
        ValueError: When the extracted mesh is empty or the path is not allowed.
        RuntimeError: When mesh extraction raises an unexpected error.
    """
    # Ensure model is loaded/cached
    load_checkpoint_info(ckpt_path, device)

    cache_key = str(ckpt_path.resolve())
    model, z_dim, _z_radius, _z_pretrained = _MODEL_CACHE[cache_key]

    from subprojects.wp5.unified_mesh_extraction import (  # noqa: PLC0415
        MeshSDFExtractor,
        clear_mesh_cache,
        load_mesh_extraction_config,
    )

    cfg = load_mesh_extraction_config()
    cfg = dc_replace(cfg, device=str(device))
    extractor = MeshSDFExtractor(config=cfg)
    extractor.set_grid(
        bbox=[
            [_PRISM_X[0] - bbox_padding, _PRISM_X[1] + bbox_padding],
            [_PRISM_Y[0] - bbox_padding, _PRISM_Y[1] + bbox_padding],
            [_PRISM_Z[0], _PRISM_Z[1] + bbox_padding],
        ],
        resolution=resolution,
    )

    z_vec = torch.zeros(z_dim, device=device)
    z_vec[0] = z0
    z_vec[1] = z1
    model.set_latent_vector(z_vec)
    clear_mesh_cache()

    def _sdf(pts: Tensor) -> Tensor:
        return model(pts)

    try:
        verts, faces = extractor.extract_mesh(_sdf)
    except Exception as exc:
        raise RuntimeError(f"Mesh extraction failed at z=({z0}, {z1}): {exc}") from exc

    if verts is None or verts.numel() == 0:
        raise ValueError(f"Empty mesh extracted at z=({z0}, {z1}). Try a different latent point.")

    sdf_mesh = trimesh.Trimesh(
        vertices=verts.cpu().numpy(),
        faces=faces.cpu().numpy(),
        process=False,
    )
    # Aeroshell blue (#5DADE2 = 93,173,226) — matches theme palette.
    sdf_mesh.visual.face_colors = [93, 173, 226, 210]  # type: ignore[assignment]

    # Try to build a full-assembly scene (sdf_mesh + inner_assembly + outer_bbox)
    # by loading the nearest pre-generated slot_00_mesh.glb and replacing its sdf_mesh.
    assembly_glb = _find_assembly_glb(ckpt_path)
    if assembly_glb is not None:
        try:
            scene = trimesh.load(str(assembly_glb), force="scene")
            if hasattr(scene, "geometry") and "sdf_mesh" in scene.geometry:
                scene.geometry["sdf_mesh"] = sdf_mesh
                return scene.export(file_obj=None, file_type="glb")  # type: ignore[return-value]
        except Exception:
            pass  # fall through to single-mesh export

    # Fallback: export only the extracted SDF mesh.
    return sdf_mesh.export(file_type="glb")  # type: ignore[return-value]
