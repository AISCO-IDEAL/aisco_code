import math
from pathlib import Path

import torch
import yaml

from subprojects.wp2_assembler.version4.components.nbz import get_training_frame_offset
from subprojects.wp2_assembler.version4.utils.reg_imp import get_regulation
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.assembly_id_utils import resolve_assembly_id


class WheelAlignmentV0(ParentLoss):
    """Wheel ground-tangency constraint (GINN-style L2-norm).

    Penalises misalignment of ∇sdf with ẑ at wheel contact points.
    Returns sqrt(Σ (1 − (∇̂sdf · ẑ)ᵢ)²) — L2 norm of angle deviations.
    Separate ALM slot from WheelInterfaceV0 so position and alignment
    have independent λ/μ that can grow at their own pace.
    """

    def _setup(self) -> None:
        self.__config = self.losses_config["wheel_loss_config"]
        self.__bb_config = self.losses_config["bb_config"]
        self.assembly_id = resolve_assembly_id(self.__bb_config)
        print(f"[WheelAlignmentV0] Using assembly_id={self.assembly_id}")
        self._manual_z_shift = self.__config.get("manual_z_shift", 0.0)
        # GINN-style "satisfied" deadband on the aggregated L2 angle-deviation loss.
        self._tolerance = float(self.__config.get("alignment_tolerance", 0.0))

        assembly_map_path = Path(__file__).parents[6] / "subprojects/wp2_assembler/version4/config/assembly_map.yaml"
        with assembly_map_path.open("r") as f:
            assembly_map = yaml.safe_load(f) or {}
        params = assembly_map[self.assembly_id]

        wheel_coords = [(p["x"], p["y"], p["z"]) for key, p in params.items() if key.startswith("wheel_")]
        cx, cy, cz = get_training_frame_offset(self.assembly_id)

        width = get_regulation("wheel.tire_width")
        radius = get_regulation("wheel.tire_diameter") / 2
        gc = get_regulation("no_build_zones.outer_prism.ground_clearance")
        C = radius - gc
        L = math.sqrt(radius**2 - C**2)

        sampling_density = 4
        dx_vals = torch.linspace(-1, 1, sampling_density)
        dy_vals = torch.linspace(-1, 1, sampling_density)
        grid_dx, grid_dy = torch.meshgrid(dx_vals, dy_vals, indexing="ij")
        grid_dx = grid_dx.flatten()
        grid_dy = grid_dy.flatten()

        wheel_point_grids = []
        for wx, wy, wz in wheel_coords:
            xs = (wx - cx) + grid_dx * L
            ys = (wy - cy) + grid_dy * (width / 2)
            zs = torch.full_like(xs, (wz - cz) + radius - C + self._manual_z_shift)
            wheel_point_grids.append(torch.stack([xs, ys, zs], dim=-1))

        self.points = torch.cat(wheel_point_grids, dim=0)

    def _loss(self, model) -> torch.Tensor:
        device = next(model.parameters()).device
        points = self.points.to(device).detach().clone().requires_grad_(True)

        sdf_raw = model(points).squeeze(-1)
        grad_sdf = torch.autograd.grad(
            outputs=sdf_raw,
            inputs=points,
            grad_outputs=torch.ones_like(sdf_raw),
            create_graph=True,
            retain_graph=True,
        )[0]

        # SDF convention: negative inside, positive outside.
        # At the wheel contact patch, the car interior is above → ∇SDF points downward.
        ground_normal = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=grad_sdf.dtype)
        grad_unit = grad_sdf / grad_sdf.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        angle_dev_sq = (1.0 - (grad_unit * ground_normal).sum(dim=-1)).pow(2)

        loss = torch.sqrt(angle_dev_sq.sum() + 1e-12)
        return torch.clamp(loss - self._tolerance, min=0.0)
