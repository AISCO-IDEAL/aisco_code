import torch

from subprojects.wp2_assembler.version4.components.nbz import call_assembly
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.assembly_id_utils import resolve_assembly_id


def _as_bool(value: object, default: bool = False) -> bool:
    """Parse bool-like config values used in legacy YAML files."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return default


def _aabb_sdf(points: torch.Tensor, box_min: torch.Tensor, box_max: torch.Tensor) -> torch.Tensor:
    """Signed distance to an axis-aligned bounding box for batched points.

    Args:
        points: Tensor of shape [N, 3].
        box_min: Tensor of shape [3].
        box_max: Tensor of shape [3].

    Returns:
        Tensor of shape [N] with signed distances (negative inside).
    """
    center = 0.5 * (box_min + box_max)
    half_extent = 0.5 * (box_max - box_min)

    q = torch.abs(points - center) - half_extent
    outside = torch.linalg.norm(torch.relu(q), dim=-1)
    inside = torch.minimum(torch.amax(q, dim=-1), torch.zeros_like(outside))
    return outside + inside


def _required_int(config: dict, key: str) -> int:
    """Read an integer config value and raise a clear error if missing."""
    value = config.get(key)
    if value is None:
        raise ValueError(f"Missing required pointgrid parameter '{key}' in bb_config.pointgrid_params")
    return int(value)


class BbV43(ParentLoss):
    """Analytic BB loss based on WP2 inner/outer AABB signed distances.

    This loss keeps the model SDF between outer and inner analytic box SDFs:
        sdf_outer(p) <= sdf_model(p) <= sdf_inner(p)

    It avoids CadQuery-to-SDF and mesh-to-SDF conversion in the BB path and keeps
    the constraint fully differentiable w.r.t. model parameters.
    """

    def _setup(self) -> None:
        self.__config = self.losses_config["bb_config"]
        self.__analytic_cfg = self.__config.get("analytic_options", {})
        self.assembly_id = resolve_assembly_id(self.__config)
        print(f"[BbV43] Using assembly_id={self.assembly_id}")

        inner_assembly, outer_solid, __ = call_assembly(id_number=self.assembly_id)
        inner_bbox = inner_assembly.toCompound().BoundingBox()
        outer_bbox = outer_solid.BoundingBox()

        self.__inner_box_min = torch.tensor([inner_bbox.xmin, inner_bbox.ymin, inner_bbox.zmin], dtype=torch.float32)
        self.__inner_box_max = torch.tensor([inner_bbox.xmax, inner_bbox.ymax, inner_bbox.zmax], dtype=torch.float32)
        self.__outer_box_min = torch.tensor([outer_bbox.xmin, outer_bbox.ymin, outer_bbox.zmin], dtype=torch.float32)
        self.__outer_box_max = torch.tensor([outer_bbox.xmax, outer_bbox.ymax, outer_bbox.zmax], dtype=torch.float32)

        self.__pointgrid: list[list[float]] = self.__pointgrid_geni(self.__config["pointgrid_params"])

        self.__inner_margin = float(self.__analytic_cfg.get("inner_margin", 0.0))
        self.__outer_margin = float(self.__analytic_cfg.get("outer_margin", 0.0))
        self.__penalty = str(self.__analytic_cfg.get("penalty", "relu")).lower()
        self.__softplus_beta = float(self.__analytic_cfg.get("softplus_beta", 10.0))

    def __pointgrid_geni(self, pointgrid_config: dict) -> list[list[float]]:
        dx = _required_int(pointgrid_config, "dx")
        dy = _required_int(pointgrid_config, "dy")
        dz = _required_int(pointgrid_config, "dz")
        rand = _required_int(pointgrid_config, "rand")

        dr = 1
        ranges = [
            (int(-2900 - rand + dr), int(2900 + rand)),
            (int(-1150 - rand + dr), int(1150 + rand)),
            (int(-rand + dr), int(1650 + rand)),
        ]
        base_steps = (dx, dy, dz)
        fine_steps = (max(1, dx // 2), max(1, dy // 2), max(1, dz // 2))

        ix0, iy0, iz0 = [int(v) for v in self.__inner_box_min.tolist()]
        ix1, iy1, iz1 = [int(v) for v in self.__inner_box_max.tolist()]

        points: list[list[float]] = []
        seen: set[tuple[int, int, int]] = set()

        for x in range(ranges[0][0], ranges[0][1], base_steps[0]):
            for y in range(ranges[1][0], ranges[1][1], base_steps[1]):
                for z in range(ranges[2][0], ranges[2][1], base_steps[2]):
                    if ix0 <= x <= ix1 and iy0 <= y <= iy1 and iz0 <= z <= iz1:
                        continue
                    key = (x, y, z)
                    if key not in seen:
                        seen.add(key)
                        points.append([float(x), float(y), float(z)])

        x_start = max(ranges[0][0], ix0)
        x_end = min(ranges[0][1], ix1)
        y_start = max(ranges[1][0], iy0)
        y_end = min(ranges[1][1], iy1)
        z_start = max(ranges[2][0], iz0)
        z_end = min(ranges[2][1], iz1)

        for x in range(x_start, x_end, fine_steps[0]):
            for y in range(y_start, y_end, fine_steps[1]):
                for z in range(z_start, z_end, fine_steps[2]):
                    key = (x, y, z)
                    if key not in seen:
                        seen.add(key)
                        points.append([float(x), float(y), float(z)])

        print(f"Pointgrid for BB loss samples {len(points)} points")
        return points

    def _violation(self, value: torch.Tensor) -> torch.Tensor:
        if self.__penalty == "softplus":
            return torch.nn.functional.softplus(self.__softplus_beta * value) / self.__softplus_beta
        return torch.relu(value)

    def _loss(self, model) -> torch.Tensor:
        with torch.enable_grad():
            target_device = next(model.parameters()).device
            self.device = target_device

            p = torch.tensor(self.__pointgrid, dtype=torch.float32, device=self.device)

            inner_box_min = self.__inner_box_min.to(self.device)
            inner_box_max = self.__inner_box_max.to(self.device)
            outer_box_min = self.__outer_box_min.to(self.device)
            outer_box_max = self.__outer_box_max.to(self.device)

            sdf_inner = _aabb_sdf(p, inner_box_min, inner_box_max)
            sdf_outer = _aabb_sdf(p, outer_box_min, outer_box_max)

            output = model(p)
            sdf_pred = output[0] if isinstance(output, tuple) else output
            sdf_pred = sdf_pred.squeeze(-1)

            upper_violation = self._violation(sdf_pred - (sdf_inner - self.__inner_margin))
            lower_violation = self._violation((sdf_outer + self.__outer_margin) - sdf_pred)

            return (upper_violation + lower_violation).mean()
