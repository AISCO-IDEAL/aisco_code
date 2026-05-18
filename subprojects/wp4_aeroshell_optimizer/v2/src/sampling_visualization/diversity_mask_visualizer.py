"""Visual sanity check for the constraint-aware diversity masks.

Renders, in a single interactive 3D plotly scene:
- the wheel-contact mask points (red),
- the inner-prism (seat / bb-inner) mask sample points (orange, subsampled),
- a batch of raw diversity samples drawn from the outer prism (grey, small),
- the same batch coloured by mask outcome:
    green   = kept,
    blue    = dropped because too close to a wheel-contact point,
    magenta = dropped because too close to an inner-prism point.

Interactive controls in the HTML:
- Click any legend entry to toggle that trace; double-click a legend entry to
  isolate it (plotly defaults).
- The "Group toggles" buttons at the top toggle whole categories at once
  (masks, samples).
- The size slider scales every trace's marker size uniformly.

Run from the repo root:

    python -m subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.diversity_mask_visualizer \\
        --assembly_id 42 --wheel_cutoff_mm 40 --bb_inner_cutoff_mm 15 --num_points 4000

Output HTML is written next to this script in
``./diversity_mask_visualizer_out/`` by default; pass ``--out path/to/file.html``
to override.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import plotly.graph_objects as go
import torch
import yaml

from subprojects.wp2_assembler.version4.utils.reg_imp import get_outer_prism_bounds_mm
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.diversity_loss.field_diversity_masked_v0 import (
    FieldDiversityMaskedV0,
)

_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "losses_v0_config.yaml"


# Each trace declares its base size; the size slider multiplies these.
_TRACE_SPECS: list[dict] = [
    {"key": "raw", "name": "raw samples", "color": "lightgrey", "size": 1.5, "opacity": 0.5, "group": "samples"},
    {"key": "kept", "name": "kept (free region)", "color": "green", "size": 2.0, "opacity": 1.0, "group": "samples"},
    {
        "key": "kept_topped_up",
        "name": "kept (after top-up)",
        "color": "limegreen",
        "size": 1.5,
        "opacity": 0.4,
        "group": "samples",
    },
    {"key": "dropped_wheel", "name": "dropped @ wheel", "color": "blue", "size": 2.5, "opacity": 1.0, "group": "samples"},
    {"key": "dropped_bb", "name": "dropped @ bb_inner", "color": "magenta", "size": 2.0, "opacity": 1.0, "group": "samples"},
    {"key": "wheel_pts", "name": "wheel mask points", "color": "red", "size": 3.5, "opacity": 1.0, "group": "masks"},
    {"key": "bb_pts", "name": "bb_inner mask points", "color": "orange", "size": 1.5, "opacity": 0.7, "group": "masks"},
]


def _outer_prism_box() -> go.Scatter3d:
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
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    zs = [c[2] for c in corners]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name="outer prism",
        legendgroup="frame",
        showlegend=True,
        line={"color": "black", "width": 2},
    )


def build_figure(loss: FieldDiversityMaskedV0, num_points: int, bb_inner_render_n: int) -> go.Figure:
    debug = loss.debug_sample(num_points=num_points, device=torch.device("cpu"))
    raw = debug["raw"]
    tag = debug["tag"]
    kept_mask = debug["keep_mask"]
    kept_topped_up = debug["kept_topped_up"]

    src_by_name = {src.name: src for src in loss.sources}
    wheel_pts = src_by_name["wheel"].points if "wheel" in src_by_name else torch.zeros((0, 3))
    bb_pts = src_by_name["bb_inner"].points if "bb_inner" in src_by_name else torch.zeros((0, 3))
    if bb_pts.shape[0] > bb_inner_render_n:
        idx = torch.randperm(bb_pts.shape[0])[:bb_inner_render_n]
        bb_pts = bb_pts[idx]

    data: dict[str, torch.Tensor] = {
        "raw": raw,
        "kept": raw[kept_mask],
        "kept_topped_up": kept_topped_up,
        "dropped_wheel": raw[tag == 1],
        "dropped_bb": raw[tag == 2],
        "wheel_pts": wheel_pts,
        "bb_pts": bb_pts,
    }

    fig = go.Figure()
    fig.add_trace(_outer_prism_box())
    for spec in _TRACE_SPECS:
        pts = data[spec["key"]].detach().cpu().numpy()
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name=spec["name"],
                legendgroup=spec["group"],
                legendgrouptitle_text=spec["group"],
                marker={"size": spec["size"], "color": spec["color"], "opacity": spec["opacity"]},
            )
        )

    # Indices into fig.data: 0 = prism wireframe, then the _TRACE_SPECS in order.
    trace_indices = list(range(1, 1 + len(_TRACE_SPECS)))

    # Continuous size slider — multiplies every trace's base size.
    base_sizes = [s["size"] for s in _TRACE_SPECS]
    slider_steps = []
    for mult in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
        slider_steps.append(  # noqa: PERF401 weird ruff rule about comprehension
            {
                "label": f"{mult:g}x",
                "method": "restyle",
                "args": [
                    {"marker.size": [b * mult for b in base_sizes]},
                    trace_indices,
                ],
            }
        )

    # Group-toggle buttons. Each button sets `visible` for the affected indices.
    def vis_button(label: str, indices: list[int], visible) -> dict:
        return {
            "label": label,
            "method": "restyle",
            "args": [{"visible": visible}, indices],
        }

    fig.update_layout(
        title=(
            f"Diversity mask check — assembly_id={loss.losses_config.get('bb_config', {}).get('assembly_id')}, "
            f"wheel_cutoff={loss._wheel_cutoff_mm} mm, bb_inner_cutoff={loss._bb_inner_cutoff_mm} mm | "
            f"raw={raw.shape[0]}, kept={int(kept_mask.sum())}, "
            f"dropped_wheel={int((tag == 1).sum())}, dropped_bb={int((tag == 2).sum())} "
            f"<br><sup>Click a legend entry to toggle; double-click to isolate. Use the bottom slider to scale all marker sizes.</sup>"
        ),
        scene={"xaxis_title": "x (mm)", "yaxis_title": "y (mm)", "zaxis_title": "z (mm)", "aspectmode": "data"},
        legend={"itemsizing": "constant"},
        sliders=[
            {
                "active": 3,  # 1.0x
                "currentvalue": {"prefix": "Marker size x"},
                "pad": {"t": 40},
                "steps": slider_steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "showactive": False,
                "x": 0.0,
                "y": 1.12,
                "xanchor": "left",
                "buttons": [
                    vis_button("Show all", trace_indices, True),
                    vis_button("Hide all", trace_indices, "legendonly"),
                ],
            },
            {
                "type": "buttons",
                "direction": "right",
                "showactive": False,
                "x": 0.0,
                "y": 1.06,
                "xanchor": "left",
                "buttons": [
                    vis_button(
                        f"Only {spec['name']}",
                        trace_indices,
                        [True if i == (1 + idx) else "legendonly" for i in trace_indices],
                    )
                    for idx, spec in enumerate(_TRACE_SPECS)
                ],
            },
        ],
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG_PATH),
        help=(
            "Path to the training losses YAML to source cutoffs/assembly_id from. "
            "Defaults to the same file the trainer uses. CLI overrides below take precedence."
        ),
    )
    parser.add_argument("--assembly_id", type=int, default=None, help="Override bb_config.assembly_id from the config.")
    parser.add_argument(
        "--wheel_cutoff_mm", type=float, default=None, help="Override field_diversity_masked_v0_config.wheel_cutoff_mm."
    )
    parser.add_argument(
        "--bb_inner_cutoff_mm", type=float, default=None, help="Override field_diversity_masked_v0_config.bb_inner_cutoff_mm."
    )
    parser.add_argument("--wheel_sampling_density", type=int, default=None)
    parser.add_argument("--bb_inner_target_point_count", type=int, default=None)
    parser.add_argument("--num_points", type=int, default=4000, help="Raw diversity samples to draw for the check.")
    parser.add_argument("--bb_inner_render_n", type=int, default=4000, help="Downsample bb_inner cloud for the plot.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output HTML path. Default: <script_dir>/diversity_mask_visualizer_out/mask_check.html",
    )
    args = parser.parse_args()

    with Path(args.config).open("r") as f:
        losses_config = yaml.safe_load(f) or {}
    # Apply CLI overrides on top of the YAML — so the YAML is the single source of truth by default.
    if args.assembly_id is not None:
        losses_config.setdefault("bb_config", {})["assembly_id"] = args.assembly_id
    masked_cfg = losses_config.setdefault("field_diversity_masked_v0_config", {})
    for key, val in [
        ("wheel_cutoff_mm", args.wheel_cutoff_mm),
        ("bb_inner_cutoff_mm", args.bb_inner_cutoff_mm),
        ("wheel_sampling_density", args.wheel_sampling_density),
        ("bb_inner_target_point_count", args.bb_inner_target_point_count),
    ]:
        if val is not None:
            masked_cfg[key] = val

    print(
        "[diversity_mask_visualizer] using config:",
        f"assembly_id={losses_config.get('bb_config', {}).get('assembly_id')},",
        f"wheel_cutoff_mm={masked_cfg.get('wheel_cutoff_mm')},",
        f"bb_inner_cutoff_mm={masked_cfg.get('bb_inner_cutoff_mm')}",
    )
    loss = FieldDiversityMaskedV0(device="cpu", losses_config=losses_config)
    fig = build_figure(loss, num_points=args.num_points, bb_inner_render_n=args.bb_inner_render_n)

    if args.out is None:
        out_dir = Path(__file__).parent / "diversity_mask_visualizer_out"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "mask_check.html"
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"[diversity_mask_visualizer] wrote {out_path}")


if __name__ == "__main__":
    main()
