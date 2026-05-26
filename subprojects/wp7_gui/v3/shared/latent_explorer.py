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

import contextlib
import glob
import importlib.util
import json
import re
import sys
import types
import warnings
from collections import OrderedDict
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
import yaml
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
    Path("/home/aisco/AISCO_rollout"),
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


# ---------------------------------------------------------------------------
# Metric estimation: drag (GAOT3), solar (WP3 SR), cruising speed (WP4 CSE)
#
# All heavy imports are deferred inside function bodies (PLC0415) so that this
# module can be imported at GUI server startup without requiring EGL/pyrender
# or the full WP1/WP3/WP4 import chains.
# ---------------------------------------------------------------------------

# Physical / optimisation defaults — mirror latent_rollout_gif_pyvista.py.
_SOLAR_SUN_DIR: tuple[float, float, float] = (-1.0, 0.0, 1.0)
_SOLAR_AREA_BUDGET_MM2: float = 4_200_000.0
_SOLAR_EFFECTIVE_IRRADIANCE_W_M2: float = 250.0
_CSE_MASS_KG: float = 210.0
_CSE_C_LOSSES_W: float = 20.0
_CSE_ROLLING_FRICTION: float = 0.006

# GAOT3 drag-estimator cache (keyed by device string).
_GAOT_ESTIMATOR_CACHE: dict[str, Any] = {}


def _load_wp4_default_cfd_config() -> dict[str, Any]:
    """Load the CFD block from WP4 v2 losses_v0_config.yaml.

    Returns:
        Dict with keys like ``v_inf``, ``rho``, ``model_id``, ``version``, …
        Returns an empty dict on any read/parse error.
    """
    cfg_path = _REPO_ROOT / "subprojects" / "wp4_aeroshell_optimizer" / "v2" / "configs" / "losses_v0_config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as exc:
        warnings.warn(
            f"[latent_explorer] failed to read default CFD config: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return {}
    return cfg.get("cfd", {}) if isinstance(cfg, dict) else {}


def _mesh_has_valid_face_indices(mesh: trimesh.Trimesh) -> bool:
    """Return True when all triangle indices are within vertex bounds."""
    try:
        faces = np.asarray(mesh.faces)
        n_verts = len(mesh.vertices)
    except Exception:
        return False
    if faces.ndim != 2 or faces.shape[1] != 3 or n_verts <= 0:
        return False
    if faces.size == 0:
        return False
    return int(faces.min()) >= 0 and int(faces.max()) < n_verts


def _get_gaot_estimator(cfd_config: dict[str, Any], device: torch.device) -> Any | None:
    """Return a cached GAOTPlyInference instance for GAOT3 drag estimation.

    The estimator is keyed by device string and re-used across requests.

    Args:
        cfd_config: CFD configuration dict (must contain ``v_inf``, ``rho``).
        device: Torch device for the estimator.

    Returns:
        A GAOTPlyInference instance, or None if initialisation fails.
    """
    device_str = str(device)
    if device_str in _GAOT_ESTIMATOR_CACHE:
        return _GAOT_ESTIMATOR_CACHE[device_str]
    try:
        from subprojects.postprocessing_visuals.cp_field_viz import GAOTPlyInference  # noqa: PLC0415

        estimator = GAOTPlyInference(cfd_config=cfd_config, device=device_str)
        _GAOT_ESTIMATOR_CACHE[device_str] = estimator
    except Exception as exc:
        warnings.warn(
            f"[latent_explorer] failed to initialise GAOT3 drag estimator: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return None
    else:
        return estimator


def _rdbu_colormap(values_01: np.ndarray) -> np.ndarray:
    """Map normalised values [0, 1] to a RdBu-style RGB array (uint8).

    0.0 → blue  (low pressure / suction side)
    0.5 → white (neutral)
    1.0 → red   (high pressure / stagnation)

    Args:
        values_01: 1-D float array with values in [0, 1].

    Returns:
        (N, 3) uint8 RGB array.
    """
    v = np.clip(values_01, 0.0, 1.0)
    r = np.clip(2.0 * v, 0.0, 1.0)
    b = np.clip(2.0 * (1.0 - v), 0.0, 1.0)
    g = np.clip(1.0 - np.abs(2.0 * v - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _solar_quality_colormap(values_01: np.ndarray) -> np.ndarray:
    """Map solar illumination quality [0, 1] to dark-grey → golden-yellow RGB.

    Args:
        values_01: 1-D float array with values in [0, 1].

    Returns:
        (N, 3) uint8 RGB array.
    """
    v = np.clip(values_01, 0.0, 1.0)[:, None]
    low = np.array([35, 35, 35], dtype=np.float32)  # dark grey (shadowed)
    high = np.array([255, 210, 0], dtype=np.float32)  # golden yellow (sun-facing)
    return np.clip(low + v * (high - low), 0, 255).astype(np.uint8)


def _make_vertex_colored_glb(mesh: trimesh.Trimesh, rgb: np.ndarray) -> bytes:
    """Return GLB bytes for *mesh* with per-vertex RGB colouring.

    Args:
        mesh: Source trimesh (geometry is copied; existing colours are discarded).
        rgb: (N_vertices, 3) uint8 colour array.

    Returns:
        GLB-encoded bytes.
    """
    alpha = np.full((len(rgb), 1), 220, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    colored = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        process=False,
    )
    colored.visual = trimesh.visual.ColorVisuals(mesh=colored, vertex_colors=rgba)
    result: bytes = colored.export(file_type="glb")  # type: ignore[assignment]
    return result


def _make_face_colored_glb(mesh: trimesh.Trimesh, rgb: np.ndarray) -> bytes:
    """Return GLB bytes for *mesh* with per-face RGB colouring.

    Args:
        mesh: Source trimesh (geometry is copied; existing colours are discarded).
        rgb: (N_faces, 3) uint8 colour array.

    Returns:
        GLB-encoded bytes.
    """
    alpha = np.full((len(rgb), 1), 220, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    colored = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        process=False,
    )
    colored.visual.face_colors = rgba  # type: ignore[assignment]
    result: bytes = colored.export(file_type="glb")  # type: ignore[assignment]
    return result


def _estimate_drag_newtons(
    gaot_loader: Any,
    mesh: trimesh.Trimesh,
    cfd_config: dict[str, Any],
) -> tuple[float | None, np.ndarray | None]:
    """Estimate drag force (N) and per-vertex CP field using GAOT3.

    Args:
        gaot_loader: A GAOTPlyInference instance.
        mesh: The extracted trimesh (vertices in mm).
        cfd_config: CFD config dict with keys ``v_inf`` and ``rho``.

    Returns:
        A 2-tuple ``(drag_newtons, cp_field)``.  ``cp_field`` is a 1-D float32
        numpy array of length N_vertices, or None on failure.
    """
    if not _mesh_has_valid_face_indices(mesh):
        return None, None
    if not (hasattr(gaot_loader, "_preprocess") and hasattr(gaot_loader, "_latent_tokens")):
        return None, None
    try:
        from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.shear_force import ShearForce  # noqa: PLC0415
        from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.utils.pressure_integration import cp_field_to_drag  # noqa: PLC0415

        loader_device = torch.device(getattr(gaot_loader, "device", "cpu"))
        verts_t = torch.tensor(mesh.vertices, dtype=torch.float32, device=loader_device)
        faces_t = torch.tensor(mesh.faces, dtype=torch.long, device=loader_device)

        with torch.no_grad():
            batch = gaot_loader._preprocess(verts_t, faces_t)
            output = gaot_loader.gaot.forward(batch=batch, tokens_pos=gaot_loader._latent_tokens)
            pressure_field = output["field"]

            integration_normals = batch.c if hasattr(batch, "c") and batch.c is not None else batch.c_raw
            integration_pos = batch.pos_raw if hasattr(batch, "pos_raw") and batch.pos_raw is not None else (verts_t * 1e-3)
            pressure_drag = cp_field_to_drag(
                cp_field=pressure_field,
                normals=integration_normals,
                face=faces_t.T,
                pos=integration_pos,
                v_inf=float(cfd_config["v_inf"]),
                rho=float(cfd_config["rho"]),
            )

            if not hasattr(gaot_loader, "_shear_force_calculator"):
                gaot_loader._shear_force_calculator = ShearForce(unit="mm^2")
            shear_drag = gaot_loader._shear_force_calculator.calculate_shear_force(verts_t, faces_t)
            drag_val = float((pressure_drag + shear_drag).detach().cpu().item())

        # Extract per-vertex CP values for heatmap visualisation.
        cp_np: np.ndarray | None = None
        with contextlib.suppress(Exception):
            cp_np = pressure_field.detach().cpu().squeeze(-1).numpy().astype(np.float32)

        if not np.isfinite(drag_val):
            return None, cp_np
        return float(abs(drag_val)), cp_np
    except Exception as exc:
        warnings.warn(
            f"[latent_explorer] drag estimation failed: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return None, None


def _estimate_solar_power_watts(
    mesh: trimesh.Trimesh,
    device: torch.device,
) -> float | None:
    """Estimate solar power (W) using the SR solar loss on the current mesh.

    Uses module-level defaults for sun direction, area budget, and irradiance.

    Args:
        mesh: The extracted trimesh (vertices in mm).
        device: Torch device for computation.

    Returns:
        Solar power in Watts, or None on failure.
    """
    if not _mesh_has_valid_face_indices(mesh):
        return None
    try:
        from subprojects.wp3_solar_estimator.rollout_losses.SR_SolarLoss import solar_loss_sr  # noqa: PLC0415

        verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
        sun_dir = torch.tensor(_SOLAR_SUN_DIR, dtype=torch.float32, device=device)

        power_w, _diagnostics = solar_loss_sr(
            vertices=verts,
            faces=faces,
            sun_dir=sun_dir,
            area_budget_mm2=_SOLAR_AREA_BUDGET_MM2,
            effective_irradiance_W_m2=_SOLAR_EFFECTIVE_IRRADIANCE_W_M2,
            cone_face_mask=None,
            epoch_name=None,
        )
        power_val = float(power_w.detach().cpu().item())
        return power_val if np.isfinite(power_val) else None
    except Exception as exc:
        warnings.warn(
            f"[latent_explorer] solar estimation failed: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return None


def _compute_cse_speed_ms(
    drag_n: float,
    solar_w: float,
) -> float | None:
    """Compute CSE cruising speed (m/s) from drag and solar estimates.

    Args:
        drag_n: Drag force in Newtons (must be > 0).
        solar_w: Solar power in Watts.

    Returns:
        Cruising speed in m/s, or None on failure.
    """
    if drag_n <= 0.0:
        return None
    try:
        from subprojects.wp4_aeroshell_optimizer.v1.src.losses.cruising_speed import cruising_speed  # noqa: PLC0415

        drag_t = torch.tensor(float(drag_n), dtype=torch.float32)
        solar_t = torch.tensor(float(max(solar_w, 0.0)), dtype=torch.float32)
        speed_ms, _diag = cruising_speed(
            solar_power=solar_t,
            drag_force_at_25ms=drag_t,
            mass=_CSE_MASS_KG,
            c_losses=_CSE_C_LOSSES_W,
            rolling_friction_coefficient=_CSE_ROLLING_FRICTION,
            max_iter=12,
            tol=1e-4,
        )
        speed_val = float(speed_ms.detach().cpu().item())
        return speed_val if np.isfinite(speed_val) else None
    except Exception as exc:
        warnings.warn(
            f"[latent_explorer] CSE computation failed: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return None


def compute_mesh_metrics(
    mesh: trimesh.Trimesh,
    device: torch.device,
) -> dict[str, Any]:
    """Compute drag, solar, CSE metrics and secondary visualization GLBs.

    This function is intentionally fault-tolerant: failures in any stage
    produce ``None`` for that entry rather than raising.

    Args:
        mesh: The extracted aeroshell trimesh (vertices in mm).
        device: Torch device for all computations.

    Returns:
        Dict with scalar keys ``drag_value`` (N), ``solar_power_w`` (W),
        ``cruising_speed_ms`` (m/s) and visualization keys ``cp_glb`` (bytes),
        ``solar_glb`` (bytes).  All values are ``None`` on failure.
    """
    cfd_config = _load_wp4_default_cfd_config()
    drag_n: float | None = None
    solar_w: float | None = None
    speed_ms: float | None = None
    cp_glb: bytes | None = None
    solar_glb: bytes | None = None

    if cfd_config:
        gaot_estimator = _get_gaot_estimator(cfd_config, device)
        if gaot_estimator is not None:
            drag_n, cp_field_np = _estimate_drag_newtons(gaot_estimator, mesh, cfd_config)
            if cp_field_np is not None and len(cp_field_np) == len(mesh.vertices):
                try:
                    cp_min, cp_max = float(cp_field_np.min()), float(cp_field_np.max())
                    span = cp_max - cp_min
                    cp_norm = (cp_field_np - cp_min) / span if span > 1e-9 else np.full_like(cp_field_np, 0.5)
                    cp_glb = _make_vertex_colored_glb(mesh, _rdbu_colormap(cp_norm))
                except Exception as exc:
                    warnings.warn(f"[latent_explorer] CP GLB build failed: {exc}", UserWarning, stacklevel=2)

    solar_w = _estimate_solar_power_watts(mesh, device)

    # Build solar quality GLB from face normals + sun direction (no extra model call).
    try:
        face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
        sun = np.array(_SOLAR_SUN_DIR, dtype=np.float32)
        sun /= max(float(np.linalg.norm(sun)), 1e-9)
        illumination = np.clip((face_normals @ sun), 0.0, 1.0)
        solar_glb = _make_face_colored_glb(mesh, _solar_quality_colormap(illumination))
    except Exception as exc:
        warnings.warn(f"[latent_explorer] solar GLB build failed: {exc}", UserWarning, stacklevel=2)

    if drag_n is not None and solar_w is not None:
        speed_ms = _compute_cse_speed_ms(drag_n, solar_w)

    return {
        "drag_value": drag_n,
        "solar_power_w": solar_w,
        "cruising_speed_ms": speed_ms,
        "cp_glb": cp_glb,
        "solar_glb": solar_glb,
    }


def extract_mesh_glb(
    ckpt_path: Path,
    z0: float,
    z1: float,
    resolution: int,
    device: torch.device,
    bbox_padding: float = 0.0,
) -> tuple[bytes, dict[str, Any], bytes | None, bytes | None]:
    """Extract the aeroshell mesh at latent point *(z0, z1)* and return GLB bytes plus metrics.

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
        A 4-tuple ``(glb_bytes, metrics, cp_glb, solar_glb)``.
        ``metrics`` is a dict with ``drag_value``, ``solar_power_w``,
        ``cruising_speed_ms`` (all ``float | None``).
        ``cp_glb`` and ``solar_glb`` are GLB bytes for the CP field and solar
        quality heatmaps respectively, or None when unavailable.

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

    # Compute drag / solar / CSE metrics + visualization GLBs on the raw SDF mesh.
    all_metrics = compute_mesh_metrics(sdf_mesh, device)
    cp_glb: bytes | None = all_metrics.pop("cp_glb", None)
    solar_glb: bytes | None = all_metrics.pop("solar_glb", None)
    metrics: dict[str, Any] = all_metrics

    # Try to build a full-assembly scene (sdf_mesh + inner_assembly + outer_bbox)
    # by loading the nearest pre-generated slot_00_mesh.glb and replacing its sdf_mesh.
    assembly_glb = _find_assembly_glb(ckpt_path)
    if assembly_glb is not None:
        try:
            scene = trimesh.load(str(assembly_glb), force="scene")
            if hasattr(scene, "geometry") and "sdf_mesh" in scene.geometry:
                scene.geometry["sdf_mesh"] = sdf_mesh
                glb_bytes: bytes = scene.export(file_obj=None, file_type="glb")  # type: ignore[assignment]
                return glb_bytes, metrics, cp_glb, solar_glb
        except Exception:
            pass  # fall through to single-mesh export

    # Fallback: export only the extracted SDF mesh.
    return sdf_mesh.export(file_type="glb"), metrics, cp_glb, solar_glb  # type: ignore[return-value]
