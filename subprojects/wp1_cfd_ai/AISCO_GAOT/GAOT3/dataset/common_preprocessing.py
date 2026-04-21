import torch

AIR_DENSITY = 1.225
A_SEA_LEVEL = 340.294
COND_EPS = 1e-6

V_INF_EULER = 25.0
V_INF_DRIVAER = 38.89
V_INF_AHMED = 1.0

DEFAULT_OUTPUT_DIRS = {
    "euler": "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler_cp_full",
    "drivaer": "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_drivaer_cp_full",
    "ahmed": "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_ahmed_cp_full",
    "blended_wing": "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_blended_wing_cp_full_filtered",
}


def compute_pressure_coefficient(
    pressure: torch.Tensor,
    v_inf: float,
    rho: float = AIR_DENSITY,
) -> torch.Tensor:
    dynamic_pressure = max(0.5 * rho * (v_inf**2), COND_EPS)
    return pressure.to(dtype=torch.float32) / dynamic_pressure


def compute_geom_conditioning(pos: torch.Tensor, v_inf: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    extent = pos.max(dim=0).values - pos.min(dim=0).values
    max_extent = torch.clamp(extent.max(), min=COND_EPS)
    aspect = extent / max_extent
    conditioning = torch.cat(
        [
            torch.tensor([v_inf], dtype=torch.float32),
            extent.to(dtype=torch.float32),
            aspect.to(dtype=torch.float32),
        ],
        dim=0,
    ).unsqueeze(0)
    return extent.to(dtype=torch.float32), aspect.to(dtype=torch.float32), conditioning
