from pathlib import Path

import torch
import yaml

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.solar_loss.solar_SR import SolarSR
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.solar_loss.solar_v3B import SolarV3B
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.solar_loss.solar_v3B_2 import SolarV3B2
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.solar_loss.solar_v5 import SolarV5
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.solar_loss.solar_v6 import SolarV6
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.solar_loss.solar_v7 import SolarV7
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import (
    NullVisManager,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import (
    CentralVisManagerV2,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.storage_utils import _load_mesh

SOLAR_LOSS_REGISTRY = {
    "SolarV3B": SolarV3B,
    "SolarV3B2": SolarV3B2,
    "SolarV5": SolarV5,
    "SolarV6": SolarV6,
    "SolarV7": SolarV7,
    "SolarSR": SolarSR,
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
    print(f"Config: loss={cfg['solar_loss']['class']}, vis_enabled={cfg['vis_manager']['enabled']}")

    ply_path = cfg["loading"]["ply_path"]
    ply_scaling = cfg["loading"]["ply_scaling_factor"]
    verts, faces = load_ply_file(ply_path, ply_scaling)
    print(f"Loaded PLY: {verts.shape[0]} vertices, {faces.shape[0]} faces")

    vis_manager = setup_vis_manager(cfg)
    device = get_device()

    loss_cls = SOLAR_LOSS_REGISTRY[cfg["solar_loss"]["class"]]
    solar_loss = loss_cls(device=device, weight=[1.0, 1.0])
    solar_loss._vis_manager = vis_manager

    skip_telemetry = not cfg["vis_manager"]["enabled"]
    result = solar_loss.get_loss(
        model=(verts, faces),
        epoch_name="-42",
        skip_telemetry=skip_telemetry,
    )

    print(f"Solar loss: {result.item():.4f} W")
    vis_manager.finalize()


if __name__ == "__main__":
    single_solar_test()
