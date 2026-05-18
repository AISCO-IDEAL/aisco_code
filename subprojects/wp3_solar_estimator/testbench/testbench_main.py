import hashlib
import math
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
import yaml

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.solar_loss.solar_MR import SolarMR
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.solar_loss.solar_MRRT import SolarMRRT
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.solar_loss.solar_SR import SolarSR
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.solar_loss.solar_SRRT import SolarSRRT
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.solar_loss.solar_v3B import SolarV3B
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.solar_loss.solar_v3B_2 import SolarV3B2
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.solar_loss.solar_v5 import SolarV5
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.solar_loss.solar_v6 import SolarV6
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.solar_loss.solar_v7 import SolarV7
from subprojects.wp4_aeroshell_optimizer.v2.src.models.blendednet.gpu_sdf import mesh_sdf
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import (
    NullVisManager,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import (
    CentralVisManagerV2,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.storage_utils import _load_mesh


class MeshSDFCallable:
    """Wraps a pre-built mesh as a differentiable SDF callable via precomputed grid + trilinear interpolation.

    Implements the minimal model contract expected by solar loss wrappers:
    - __call__(pts) -> signed distances  (differentiable w.r.t. pts via F.grid_sample)
    - get_model_info_dict() -> dict

    Rather than calling the O(Q×T) brute-force mesh_sdf per query, the SDF is evaluated
    once on a dense grid at init and all future queries use F.grid_sample (O(Q×8)).
    This avoids OOM from large meshes and makes the callable differentiable w.r.t. query
    coordinates, which is required for compute_gradient_flow in the solar loss extractors.

    Args:
        verts: (V, 3) mesh vertices
        faces: (F, 3) face indices
        grid_resolution: side length R of the precomputed R×R×R SDF grid
        cache_dir: directory to store the precomputed grid; None to skip disk caching
        precompute_chunk_size: batch size used during the one-time mesh_sdf precomputation
    """

    def __init__(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        grid_resolution: int = 128,
        cache_dir: Path | None = None,
        precompute_chunk_size: int = 4096,
    ):
        self.verts = verts
        self.faces = faces
        self._grid_resolution = grid_resolution

        # bbox with 5% padding around the mesh
        vmin = verts.min(dim=0).values
        vmax = verts.max(dim=0).values
        padding = 0.05 * (vmax - vmin)
        self._bbox_min = (vmin - padding).to(verts.device)
        self._bbox_max = (vmax + padding).to(verts.device)

        cache_file: Path | None = None
        if cache_dir is not None:
            cache_dir.mkdir(exist_ok=True)
            mesh_hash = hashlib.md5(verts.cpu().numpy().tobytes() + faces.cpu().numpy().tobytes()).hexdigest()[:12]
            cache_file = cache_dir / f"{mesh_hash}_grid{grid_resolution}.pt"
            if cache_file.exists():
                print(f"[SDF grid] loading precomputed {grid_resolution}³ grid from cache")
                self._sdf_grid = torch.load(cache_file, map_location=verts.device, weights_only=True)
                return

        print(f"[SDF grid] precomputing {grid_resolution}³ SDF grid (chunk_size={precompute_chunk_size})...")
        t0 = time.perf_counter()
        self._sdf_grid = self._precompute_grid(grid_resolution, precompute_chunk_size)
        print(f"[SDF grid] done in {time.perf_counter() - t0:.1f}s")

        if cache_file is not None:
            torch.save(self._sdf_grid.cpu(), cache_file)
            print(f"[SDF grid] saved to {cache_file}")

    def _precompute_grid(self, res: int, chunk_size: int) -> torch.Tensor:
        device = self.verts.device
        x_lin = torch.linspace(self._bbox_min[0].item(), self._bbox_max[0].item(), res)
        y_lin = torch.linspace(self._bbox_min[1].item(), self._bbox_max[1].item(), res)
        z_lin = torch.linspace(self._bbox_min[2].item(), self._bbox_max[2].item(), res)
        # (D, H, W) = (z, y, x) ordering to match F.grid_sample's expected layout
        gz, gy, gx = torch.meshgrid(z_lin, y_lin, x_lin, indexing="ij")
        pts_grid = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).to(device)

        sdf_chunks = []
        with torch.no_grad():
            for i in range(0, pts_grid.shape[0], chunk_size):
                chunk = pts_grid[i : i + chunk_size]
                sdf_chunks.append(mesh_sdf(chunk, self.verts, self.faces).cpu())

        return torch.cat(sdf_chunks).reshape(res, res, res).to(device)  # (D, H, W)

    def __call__(self, pts: torch.Tensor) -> torch.Tensor:
        # Normalize pts to [-1, 1] within bbox.
        # F.grid_sample coord order: (x→W, y→H, z→D), matching the (D, H, W) grid layout.
        bbox_min = self._bbox_min.to(pts.device)
        bbox_max = self._bbox_max.to(pts.device)
        pts_norm = 2.0 * (pts - bbox_min) / (bbox_max - bbox_min) - 1.0  # (N, 3)
        grid = self._sdf_grid.to(pts.device)[None, None]  # (1, 1, D, H, W)
        sample_coords = pts_norm.view(1, 1, 1, -1, 3)  # (1, 1, 1, N, 3)
        result = F.grid_sample(grid, sample_coords, mode="bilinear", padding_mode="border", align_corners=True)
        return result.view(-1)  # (N,)

    def get_model_info_dict(self) -> dict:
        return {
            "model_type": "mesh_sdf_grid",
            "num_vertices": self.verts.shape[0],
            "num_faces": self.faces.shape[0],
            "grid_resolution": self._grid_resolution,
        }


SOLAR_LOSS_REGISTRY = {
    "SolarV3B": SolarV3B,
    "SolarV3B2": SolarV3B2,
    "SolarV5": SolarV5,
    "SolarV6": SolarV6,
    "SolarV7": SolarV7,
    "SolarSR": SolarSR,
    "SolarSRRT": SolarSRRT,
    "SolarMR": SolarMR,
    "SolarMRRT": SolarMRRT,
}


def load_config() -> dict:
    cfg_path = Path(__file__).parent / "testbench_cfg.yaml"
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def apply_loading_overrides(
    cfg: dict,
    ply_path: str | None = None,
    ply_scaling_factor: float | None = None,
) -> None:
    """Override the input mesh without editing testbench_cfg.yaml."""
    env_ply_path = os.getenv("AISCO_TESTBENCH_PLY_PATH", "").strip()
    env_scale = os.getenv("AISCO_TESTBENCH_PLY_SCALE", "").strip()

    if ply_path or env_ply_path:
        cfg["loading"]["ply_path"] = ply_path or env_ply_path
    if ply_scaling_factor is not None:
        cfg["loading"]["ply_scaling_factor"] = float(ply_scaling_factor)
    elif env_scale:
        cfg["loading"]["ply_scaling_factor"] = float(env_scale)


def load_ply_file(ply_path: str, scaling_factor: float) -> tuple[torch.Tensor, torch.Tensor]:
    mesh = _load_mesh(ply_path)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32) * scaling_factor
    faces = torch.tensor(mesh.faces, dtype=torch.long)
    return verts, faces


def _make_testbench_run_dir(cfg: dict) -> Path | None:
    """Create the top-level output directory shared by one testbench invocation."""
    if not cfg["vis_manager"]["enabled"]:
        return None

    base_run_dir = Path(cfg["vis_manager"]["run_dir"])
    base_run_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(cfg["vis_manager"].get("run_id") or "testbench")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slurm_job_id = os.getenv("SLURM_JOB_ID", "").strip()
    run_name = f"{timestamp}_{run_id}_pid{os.getpid()}"
    if slurm_job_id:
        run_name = f"{run_name}_job{slurm_job_id}"

    run_dir = base_run_dir / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = base_run_dir / f"{run_name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_vis_manager(cfg: dict, loss_class: str | None = None, run_dir: Path | None = None) -> tuple:
    """Setup visualizer manager and return (manager, save_dir).

    Returns:
        (vis_manager, save_dir_path) where save_dir_path is None if vis is disabled
    """
    if cfg["vis_manager"]["enabled"]:
        resolved_run_dir = run_dir or _make_testbench_run_dir(cfg)
        if resolved_run_dir is None:
            return NullVisManager(), None
        loss_run_dir = resolved_run_dir / loss_class if loss_class is not None else resolved_run_dir
        loss_run_dir.mkdir(parents=True, exist_ok=True)

        vis_manager = CentralVisManagerV2(run_dir=loss_run_dir, run_id=loss_class or "data")
        return vis_manager, loss_run_dir
    else:
        return NullVisManager(), None


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_sdf_model(cfg: dict, verts: torch.Tensor, faces: torch.Tensor, device: str) -> "MeshSDFCallable":
    sdf_cfg = cfg.get("sdf_model", {})
    grid_resolution = int(sdf_cfg.get("grid_resolution", 128))
    cache_dir_str = sdf_cfg.get("cache_dir", "sdf_cache")
    cache_dir = Path(__file__).parent / cache_dir_str if cache_dir_str else None
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    precompute_chunk_size = int(sdf_cfg.get("precompute_chunk_size", n_cpus * 4096))
    return MeshSDFCallable(
        verts.to(device),
        faces.to(device),
        grid_resolution=grid_resolution,
        cache_dir=cache_dir,
        precompute_chunk_size=precompute_chunk_size,
    )


def run_one(
    loss_class: str,
    cfg: dict,
    sdf_model: "MeshSDFCallable",
    device: str,
    losses_config: dict,
    run_dir: Path | None = None,
) -> None:
    vis_enabled = cfg["vis_manager"]["enabled"]
    true_scale = cfg["vis_manager"]["true_scale"]

    vis_manager, save_dir = setup_vis_manager(cfg, loss_class=loss_class, run_dir=run_dir)
    if save_dir is not None:
        print(f"  vis → {save_dir}")

    loss_cls = SOLAR_LOSS_REGISTRY[loss_class]
    solar_loss = loss_cls(device=device, weight=[1.0, 1.0], losses_config=losses_config)
    solar_loss._vis_manager = vis_manager

    skip_telemetry = not vis_enabled
    t0 = time.perf_counter()
    result = solar_loss.get_loss(model=sdf_model, epoch_name=0, skip_telemetry=skip_telemetry)
    elapsed = time.perf_counter() - t0

    print(f"  loss={result.item():.2f} W  elapsed={elapsed:.2f}s")

    if vis_enabled and hasattr(solar_loss, "extractor") and solar_loss.extractor.last_dL_dv is not None:
        vis_manager.vis_mesh_gradient(
            0,
            loss_class.lower(),
            solar_loss.extractor.last_verts,
            solar_loss.extractor.last_faces,
            solar_loss.extractor.last_dL_dv,
            true_scale=true_scale,
        )

    vis_manager.finalize()


def single_solar_test(ply_path: str | None = None, ply_scaling_factor: float | None = None) -> None:
    cfg = load_config()
    apply_loading_overrides(cfg, ply_path, ply_scaling_factor)
    loss_class = cfg["solar_loss"]["class"]
    device = get_device()

    print("=== Solar Testbench ===")
    print(f"Loss class : {loss_class}")
    print(f"Vis enabled: {cfg['vis_manager']['enabled']}")
    print(f"Device     : {device}")
    run_dir = _make_testbench_run_dir(cfg)
    if run_dir is not None:
        print(f"Run dir    : {run_dir}")

    ply_path = cfg["loading"]["ply_path"]
    ply_scaling = cfg["loading"]["ply_scaling_factor"]
    print(f"\nLoading PLY: {ply_path}  (scale={ply_scaling})")
    verts, faces = load_ply_file(ply_path, ply_scaling)
    print(f"  {verts.shape[0]} vertices, {faces.shape[0]} faces")

    from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config as _load_wp4_config  # noqa: PLC0415
    losses_config = _load_wp4_config(v="v2", cfg_name="losses_v0_config.yaml")

    sdf_model = _build_sdf_model(cfg, verts, faces, device)

    print(f"\nRunning {loss_class}...")
    run_one(loss_class, cfg, sdf_model, device, losses_config, run_dir=run_dir)


# Losses that support vision cone exclusion — skips legacy V3B/V5/V6/V7
CONE_LOSS_CLASSES = ["SolarSR", "SolarSRRT", "SolarMR", "SolarMRRT"]


def all_solar_tests(ply_path: str | None = None, ply_scaling_factor: float | None = None) -> None:
    cfg = load_config()
    apply_loading_overrides(cfg, ply_path, ply_scaling_factor)
    device = get_device()

    print("=== Solar Testbench — all cone-compatible losses ===")
    print(f"Vis enabled: {cfg['vis_manager']['enabled']}")
    print(f"Device     : {device}")
    run_dir = _make_testbench_run_dir(cfg)
    if run_dir is not None:
        print(f"Run dir    : {run_dir}")

    ply_path = cfg["loading"]["ply_path"]
    ply_scaling = cfg["loading"]["ply_scaling_factor"]
    print(f"\nLoading PLY: {ply_path}  (scale={ply_scaling})")
    verts, faces = load_ply_file(ply_path, ply_scaling)
    print(f"  {verts.shape[0]} vertices, {faces.shape[0]} faces")

    from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config as _load_wp4_config  # noqa: PLC0415
    losses_config = _load_wp4_config(v="v2", cfg_name="losses_v0_config.yaml")

    # SDF grid is built once and reused across all losses (cached after first call)
    sdf_model = _build_sdf_model(cfg, verts, faces, device)

    results: list[tuple[str, float, float]] = []
    for loss_class in CONE_LOSS_CLASSES:
        print(f"\n--- {loss_class} ---")
        try:
            t0 = time.perf_counter()
            run_one(loss_class, cfg, sdf_model, device, losses_config, run_dir=run_dir)
            results.append((loss_class, time.perf_counter() - t0, float("nan")))
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((loss_class, float("nan"), float("nan")))

    print("\n=== Summary ===")
    for loss_class, elapsed, _ in results:
        status = f"{elapsed:.1f}s" if math.isfinite(elapsed) else "FAILED"
        print(f"  {loss_class:12s}  {status}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the WP3 solar loss testbench.")
    parser.add_argument("--all", action="store_true", help="Run all cone-compatible rollout solar losses.")
    parser.add_argument("--ply-path", default=None, help="Override loading.ply_path from testbench_cfg.yaml.")
    parser.add_argument(
        "--ply-scaling-factor",
        type=float,
        default=None,
        help="Override loading.ply_scaling_factor from testbench_cfg.yaml.",
    )
    args = parser.parse_args()

    if args.all:
        all_solar_tests(args.ply_path, args.ply_scaling_factor)
    else:
        single_solar_test(args.ply_path, args.ply_scaling_factor)
