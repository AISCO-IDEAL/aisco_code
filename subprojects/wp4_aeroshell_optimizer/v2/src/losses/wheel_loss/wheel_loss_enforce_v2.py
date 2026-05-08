import math
from pathlib import Path

import torch
import yaml

from subprojects.wp2_assembler.version4.utils.reg_imp import get_regulation
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.models.enforce_projector import enforce_layer_context


class WheelLossEnforceV2(ParentLoss):
    """
    Two-phase ENFORCE-aware wheel loss.

    Phase 1 (enforce projector inactive): primary loss is huber(sdf_raw, 0) — a linear
    distance-to-zero penalty at wheel-contact sample points. Same flavour as the legacy
    wheel_loss_enforce / wheel_loss_v3 raw signal.

    Phase 2 (projector active, last `1 - activation_epoch_ratio` of training): primary
    loss switches to `(sdf_raw - sdf_proj.detach())²` where sdf_proj is the Newton-
    projected target (which collapses to 0 at wheel-contact points). Mathematically this
    reduces to sdf_raw² — a quadratic distance-to-zero penalty. The stronger pull near
    the boundary helps the zero-crossing actually settle on the wheel-contact patch.

    A small-weight ground-tangency aux runs in BOTH phases: ∇sdf_raw should be parallel
    to ẑ at the contact patch, so the surface ends up tangent to the ground plane.
    """

    def _setup(self) -> None:
        self.__config = self.losses_config["wheel_loss_config"]
        self.assembly_id = int(self.losses_config["bb_config"].get("assembly_id", 1))
        self._manual_z_shift = self.__config.get("manual_z_shift", 0.0)
        self.__threshold_huber = float(self.__config.get("threshold_huber", 10.0))
        self.__alignment_weight = float(self.__config.get("enforce_alignment_weight", 0.05))
        self.__enforce_registered = False

        assembly_map_path = Path(__file__).parents[6] / "subprojects/wp2_assembler/version4/config/assembly_map.yaml"
        with assembly_map_path.open("r") as f:
            assembly_map = yaml.safe_load(f) or {}
        params = assembly_map[self.assembly_id]

        wheel_coords = [(p["x"], p["y"], p["z"]) for key, p in params.items() if key.startswith("wheel_")]

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
            xs = wx + grid_dx * L
            ys = wy + grid_dy * (width / 2)
            zs = torch.full_like(xs, wz + radius - C + self._manual_z_shift)
            wheel_point_grids.append(torch.stack([xs, ys, zs], dim=-1))

        self.points = torch.cat(wheel_point_grids, dim=0).to(self.device)

    def _loss(self, model) -> torch.Tensor:
        if not self.__enforce_registered and hasattr(model, "register_enforce_constraint"):

            def wheel_constraint(_coords, sdf):
                return sdf.squeeze(-1).abs()

            model.register_enforce_constraint("wheel", wheel_constraint)
            self.__enforce_registered = True

        # Leaf with grad enabled — needed for autograd.grad in the alignment aux.
        points = self.points.detach().clone().requires_grad_(True)

        sdf_raw = model(points).squeeze(-1)
        grad_sdf = torch.autograd.grad(
            outputs=sdf_raw,
            inputs=points,
            grad_outputs=torch.ones_like(sdf_raw),
            create_graph=True,
            retain_graph=True,
        )[0]

        enforce_layer = getattr(model, "enforce_layer", None)
        if enforce_layer is None or not enforce_layer.is_active:
            primary_loss = torch.nn.functional.huber_loss(sdf_raw, torch.zeros_like(sdf_raw), delta=self.__threshold_huber)
        else:
            with enforce_layer_context(model, "wheel"):
                target = model(points).squeeze(-1).detach()
            primary_loss = (sdf_raw - target).pow(2).mean()

        ground_normal = torch.tensor([0.0, 0.0, 1.0], device=grad_sdf.device, dtype=grad_sdf.dtype)
        grad_unit = grad_sdf / grad_sdf.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        alignment_loss = (1.0 - (grad_unit * ground_normal).sum(dim=-1)).pow(2).mean()

        return primary_loss + self.__alignment_weight * alignment_loss
