import hashlib
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
import yaml

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
}


def load_config() -> dict:
    cfg_path = Path(__file__).parent / "testbench_cfg.yaml"
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def load_ply_file(ply_path: str, scaling_factor: float) -> tuple[torch.Tensor, torch.Tensor]:
    mesh = _load_mesh(ply_path)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32) * scaling_factor
    faces = torch.tensor(mesh.faces, dtype=torch.long)
    return verts, faces


def setup_vis_manager(cfg: dict, loss_class: str | None = None) -> tuple:
    """Setup visualizer manager and return (manager, save_dir).

    Returns:
        (vis_manager, save_dir_path) where save_dir_path is None if vis is disabled
    """
    if cfg["vis_manager"]["enabled"]:
        # Resolve base run_dir
        base_run_dir = Path(cfg["vis_manager"]["run_dir"])
        base_run_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique run_id with timestamp and loss class name
        if loss_class is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_dir_name = f"{timestamp}_{loss_class}"
            run_dir = base_run_dir / unique_dir_name
        else:
            run_dir = base_run_dir / cfg["vis_manager"]["run_id"]

        vis_manager = CentralVisManagerV2(run_dir=run_dir, run_id="data")
        return vis_manager, run_dir
    else:
        return NullVisManager(), None


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def single_solar_test():
    cfg = load_config()
    loss_class = cfg["solar_loss"]["class"]
    vis_enabled = cfg["vis_manager"]["enabled"]
    print("=== Solar Testbench ===")
    print(f"Loss class : {loss_class}")
    print(f"Vis enabled: {vis_enabled}")

    device = get_device()
    print(f"Device     : {device}")

    ply_path = cfg["loading"]["ply_path"]
    ply_scaling = cfg["loading"]["ply_scaling_factor"]
    print(f"\nLoading PLY: {ply_path}  (scale={ply_scaling})")
    verts, faces = load_ply_file(ply_path, ply_scaling)
    print(f"  {verts.shape[0]} vertices, {faces.shape[0]} faces")

    vis_manager, save_dir = setup_vis_manager(cfg, loss_class=loss_class)

    if save_dir is not None:
        print(f"\nVisualizer will save files to: {save_dir}")

    print(f"\nInitialising {loss_class}...")
    loss_cls = SOLAR_LOSS_REGISTRY[loss_class]
    solar_loss = loss_cls(device=device, weight=[1.0, 1.0])
    solar_loss._vis_manager = vis_manager

    sdf_cfg = cfg.get("sdf_model", {})
    grid_resolution = int(sdf_cfg.get("grid_resolution", 128))
    cache_dir_str = sdf_cfg.get("cache_dir", "sdf_cache")
    cache_dir = Path(__file__).parent / cache_dir_str if cache_dir_str else None
    sdf_model = MeshSDFCallable(verts.to(device), faces.to(device), grid_resolution=grid_resolution, cache_dir=cache_dir)

    skip_telemetry = not vis_enabled
    print(f"\nRunning loss evaluation (skip_telemetry={skip_telemetry})...")
    t0 = time.perf_counter()
    result = solar_loss.get_loss(model=sdf_model, epoch_name=-42, skip_telemetry=skip_telemetry)
    elapsed = time.perf_counter() - t0

    print(f"\nSolar loss : {result.item():.4f} W")
    print(f"Elapsed    : {elapsed:.2f}s")
    true_scale = cfg["vis_manager"]["true_scale"]
    if vis_enabled and hasattr(solar_loss, "extractor") and solar_loss.extractor.last_dL_dv is not None:
        vis_manager.vis_mesh_gradient(
            -42,
            loss_class.lower(),
            solar_loss.extractor.last_verts,
            solar_loss.extractor.last_faces,
            solar_loss.extractor.last_dL_dv,
            true_scale=true_scale,
        )

    vis_manager.finalize()


if __name__ == "__main__":
    single_solar_test()
