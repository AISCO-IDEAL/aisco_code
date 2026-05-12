import hashlib
import time
from pathlib import Path

import torch
import yaml

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.solar_loss.solar_SR import SolarSR
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
    """Wraps a pre-built mesh as an SDF callable for testbench use.

    Implements the minimal model contract expected by solar loss wrappers:
    - __call__(pts) -> signed distances
    - get_model_info_dict() -> dict (for logging/hashing when rendering is enabled)

    If cache_dir is provided, SDF results are cached to disk keyed by (mesh, query_pts)
    so repeated testbench runs skip the expensive brute-force evaluation.
    """

    def __init__(self, verts: torch.Tensor, faces: torch.Tensor, cache_dir: Path | None = None):
        self.verts = verts
        self.faces = faces
        self._cache_dir = cache_dir
        self._cache_hits = 0
        self._cache_misses = 0
        if cache_dir is not None:
            cache_dir.mkdir(exist_ok=True)
            mesh_bytes = verts.cpu().numpy().tobytes() + faces.cpu().numpy().tobytes()
            self._mesh_hash = hashlib.md5(mesh_bytes).hexdigest()[:12]
            existing = len(list(cache_dir.glob(f"{self._mesh_hash}_*.pt")))
            print(f"[SDF cache] dir={cache_dir}  mesh_hash={self._mesh_hash}  existing entries={existing}")

    def __call__(self, pts: torch.Tensor) -> torch.Tensor:
        if self._cache_dir is not None:
            pts_hash = hashlib.md5(pts.cpu().numpy().tobytes()).hexdigest()[:12]
            cache_file = self._cache_dir / f"{self._mesh_hash}_{pts_hash}.pt"
            if cache_file.exists():
                self._cache_hits += 1
                return torch.load(cache_file, map_location=pts.device, weights_only=True)
            self._cache_misses += 1
            result = mesh_sdf(pts, self.verts, self.faces)
            torch.save(result.cpu(), cache_file)
            return result
        return mesh_sdf(pts, self.verts, self.faces)

    def get_model_info_dict(self) -> dict:
        return {"model_type": "mesh_sdf", "num_vertices": self.verts.shape[0], "num_faces": self.faces.shape[0]}


SOLAR_LOSS_REGISTRY = {
    "SolarV3B":  SolarV3B,
    "SolarV3B2": SolarV3B2,
    "SolarV5":   SolarV5,
    "SolarV6":   SolarV6,
    "SolarV7":   SolarV7,
    "SolarSR":   SolarSR,
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


def setup_vis_manager(cfg: dict):
    if cfg["vis_manager"]["enabled"]:
        # Resolve relative run_dir against the testbench folder, not CWD
        run_dir = Path(__file__).parent / cfg["vis_manager"]["run_dir"]
        return CentralVisManagerV2(run_dir=run_dir, run_id=cfg["vis_manager"]["run_id"])
    else:
        return NullVisManager()


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

    vis_manager = setup_vis_manager(cfg)

    print(f"\nInitialising {loss_class}...")
    loss_cls = SOLAR_LOSS_REGISTRY[loss_class]
    solar_loss = loss_cls(device=device, weight=[1.0, 1.0])
    solar_loss._vis_manager = vis_manager

    cache_dir = Path(__file__).parent / "sdf_cache"
    sdf_model = MeshSDFCallable(verts.to(device), faces.to(device), cache_dir=cache_dir)

    skip_telemetry = not vis_enabled
    print(f"\nRunning loss evaluation (skip_telemetry={skip_telemetry})...")
    t0 = time.perf_counter()
    result = solar_loss.get_loss(model=sdf_model, epoch_name="-42", skip_telemetry=skip_telemetry)
    elapsed = time.perf_counter() - t0

    if sdf_model._cache_dir is not None:
        total = sdf_model._cache_hits + sdf_model._cache_misses
        print(f"[SDF cache] {sdf_model._cache_hits}/{total} chunks from cache, {sdf_model._cache_misses} computed")

    print(f"\nSolar loss : {result.item():.4f} W")
    print(f"Elapsed    : {elapsed:.2f}s")
    vis_manager.finalize()


if __name__ == "__main__":
    single_solar_test()
