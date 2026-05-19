"""Visual sanity check for the BB-loss geometric safety buffer (BbInnerV1 / BbOuterV1).

Renders, in a single interactive 3D plotly scene, the four sample clouds the V1
losses actually use during training:

- inner raw cloud (orange)           — points sampled on the true inner solid surfaces (pre-offset)
- inner loss cloud (red)             — same points shifted outward by inner_outward_offset_mm along their face normals
- inner masked-out cloud (grey)      — wheel bottom-cap samples dropped by the mask
- outer raw cloud (light blue)       — points sampled on the true outer NBZ surface
- outer loss cloud — sides (dark blue) and bottom (cyan) shifted inward by their respective offsets

Use this once after editing the surface_sampling block in losses_v0_config.yaml
to confirm:
- only the wheel bottom cap is masked (seat bottom stays in the loss cloud),
- the inner red cloud sits ~offset mm outside the orange cloud,
- the outer dark-blue cloud sits ~offset mm inside the light-blue cloud,
- bottom cyan samples only move by outer_bottom_offset_mm,
- nothing dips below ground_clearance.

Run from the repo root:

    python -m subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.bb_offset_visualizer \\
        --assembly_id 42 --inner_outward_offset_mm 8 --outer_inward_offset_mm 10

Output HTML goes to ``./bb_offset_visualizer_out/`` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import plotly.graph_objects as go
import torch
import yaml

from subprojects.wp2_assembler.version4.utils.reg_imp import get_outer_prism_bounds_mm
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.bb_loss.bb_inner_v1 import BbInnerV1
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.bb_loss.bb_outer_v1 import BbOuterV1

_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "losses_v0_config.yaml"

_TRACE_SPECS: list[dict] = [
    {"key": "inner_raw", "name": "inner raw", "color": "purple", "size": 1.5, "opacity": 1.0, "group": "inner"},
    {"key": "inner_offset", "name": "inner loss (offset)", "color": "red", "size": 2.0, "opacity": 0.8, "group": "inner"},
    {
        "key": "inner_masked",
        "name": "inner masked-out (wheel cap)",
        "color": "black",
        "size": 2.5,
        "opacity": 1.0,
        "group": "inner",
    },
    {"key": "outer_raw", "name": "outer raw", "color": "green", "size": 1.5, "opacity": 0.6, "group": "outer"},
    {
        "key": "outer_offset_sides",
        "name": "outer loss — sides",
        "color": "darkblue",
        "size": 1.8,
        "opacity": 1.0,
        "group": "outer",
    },
    {"key": "outer_offset_bottom", "name": "outer loss — bottom", "color": "cyan", "size": 1.8, "opacity": 1.0, "group": "outer"},
]


def _outer_prism_wireframe() -> go.Scatter3d:
    p = get_outer_prism_bounds_mm()
    x0, x1 = p["x"]
    y0, y1 = p["y"]
    z0, z1 = p["z"]
    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
        (x0, y0, z1),
        (None, None, None),
        (x1, y0, z0),
        (x1, y0, z1),
        (None, None, None),
        (x1, y1, z0),
        (x1, y1, z1),
        (None, None, None),
        (x0, y1, z0),
        (x0, y1, z1),
    ]
    return go.Scatter3d(
        x=[c[0] for c in corners],
        y=[c[1] for c in corners],
        z=[c[2] for c in corners],
        mode="lines",
        name="outer prism (regulatory)",
        line={"color": "black", "width": 2},
        showlegend=True,
    )


def build_figure(bb_inner: BbInnerV1, bb_outer: BbOuterV1) -> go.Figure:
    outer_is_bottom = bb_outer._outer_is_bottom_mask
    outer_offset_all = bb_outer._outer_surface_points
    data: dict[str, torch.Tensor] = {
        "inner_raw": bb_inner._inner_surface_points_raw,
        "inner_offset": bb_inner._inner_surface_points,
        "inner_masked": bb_inner._inner_masked_out_points,
        "outer_raw": bb_outer._outer_surface_points_raw,
        "outer_offset_sides": outer_offset_all[~outer_is_bottom],
        "outer_offset_bottom": outer_offset_all[outer_is_bottom],
    }

    fig = go.Figure()
    fig.add_trace(_outer_prism_wireframe())
    for spec in _TRACE_SPECS:
        pts = data[spec["key"]].detach().cpu().numpy()
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name=f"{spec['name']} (n={pts.shape[0]})",
                marker={"size": spec["size"], "color": spec["color"], "opacity": spec["opacity"]},
            )
        )

    trace_indices = list(range(1, 1 + len(_TRACE_SPECS)))
    base_sizes = [s["size"] for s in _TRACE_SPECS]
    slider_steps = [
        {
            "label": f"{mult:g}x",
            "method": "restyle",
            "args": [{"marker.size": [b * mult for b in base_sizes]}, trace_indices],
        }
        for mult in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    ]

    sampling_cfg = bb_inner.losses_config.get("bb_config", {}).get("surface_sampling", {})
    fig.update_layout(
        title=(
            f"BB-loss offset check — assembly_id={bb_inner.losses_config.get('bb_config', {}).get('assembly_id')} | "
            f"inner_outward=+{sampling_cfg.get('inner_outward_offset_mm', 0.0)}mm, "
            f"outer_sides=-{sampling_cfg.get('outer_inward_offset_mm', 0.0)}mm, "
            f"outer_bottom=-{sampling_cfg.get('outer_bottom_offset_mm', 0.0)}mm"
            f"<br><sup>Click legend entries to toggle; double-click to isolate. Use the slider to scale marker sizes.</sup>"
        ),
        scene={"xaxis_title": "x (mm)", "yaxis_title": "y (mm)", "zaxis_title": "z (mm)", "aspectmode": "data"},
        legend={"itemsizing": "constant"},
        sliders=[{"active": 3, "currentvalue": {"prefix": "Marker size x"}, "pad": {"t": 40}, "steps": slider_steps}],
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(_DEFAULT_CONFIG_PATH))
    parser.add_argument("--assembly_id", type=int, default=None)
    parser.add_argument("--inner_outward_offset_mm", type=float, default=None)
    parser.add_argument("--outer_inward_offset_mm", type=float, default=None)
    parser.add_argument("--outer_bottom_offset_mm", type=float, default=None)
    parser.add_argument("--mask_downward_normal_threshold", type=float, default=None)
    parser.add_argument("--cap_z_tolerance_mm", type=float, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    with Path(args.config).open("r") as f:
        losses_config: dict = yaml.safe_load(f) or {}

    bb_cfg = losses_config.setdefault("bb_config", {})
    if args.assembly_id is not None:
        bb_cfg["assembly_id"] = args.assembly_id
    sampling_cfg = bb_cfg.setdefault("surface_sampling", {})
    for key, val in [
        ("inner_outward_offset_mm", args.inner_outward_offset_mm),
        ("outer_inward_offset_mm", args.outer_inward_offset_mm),
        ("outer_bottom_offset_mm", args.outer_bottom_offset_mm),
        ("mask_downward_normal_threshold", args.mask_downward_normal_threshold),
        ("cap_z_tolerance_mm", args.cap_z_tolerance_mm),
    ]:
        if val is not None:
            sampling_cfg[key] = val

    # The V1 classes only run _setup() if their weight is non-zero (ParentLoss.__init__ guard).
    # Force a non-zero weight here so the offset/sampling logic actually executes.
    bb_inner = BbInnerV1(device="cpu", losses_config=losses_config, weight=[1.0, 1.0], loss_name="bb_inner_v1")
    bb_outer = BbOuterV1(device="cpu", losses_config=losses_config, weight=[1.0, 1.0], loss_name="bb_outer_v1")

    print(
        f"[bb_offset_visualizer] assembly_id={bb_cfg.get('assembly_id')}, "
        f"inner_outward={sampling_cfg.get('inner_outward_offset_mm', 0.0)}mm, "
        f"outer_sides={sampling_cfg.get('outer_inward_offset_mm', 0.0)}mm, "
        f"outer_bottom={sampling_cfg.get('outer_bottom_offset_mm', 0.0)}mm"
    )

    fig = build_figure(bb_inner, bb_outer)

    if args.out is None:
        out_dir = Path(__file__).parent / "bb_offset_visualizer_out"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "bb_offset_check.html"
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"[bb_offset_visualizer] wrote {out_path}")


if __name__ == "__main__":
    main()
