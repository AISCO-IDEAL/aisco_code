from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Path as MplPath  # type: ignore[attr-defined]


def plot_chord_diagram(  # noqa: PLR0915
    cos_sim_matrix: np.ndarray,
    grad_norms: np.ndarray,
    loss_names: list[str],
    epoch: int,
    output_path: Path,
) -> None:
    """
    Renders a circular chord diagram showing gradient conflict relationships.

    Args:
        cos_sim_matrix: (N, N) symmetric cosine similarity matrix, range [-1, 1]
        grad_norms: (N,) array of gradient norms
        loss_names: list of N loss names
        epoch: epoch number for title
        output_path: file path to save PNG

    Design:
        - Outer ring: fixed-width arc segments with gradient norm labels
        - Chords: tapered ribbons, endpoints placed along each arc using angular offsets
          - Color: red (conflict, cos_sim ≈ -1) → yellow (neutral, ≈ 0) → green (aligned, ≈ +1)
          - Width at endpoints: proportional to grad_norm (angular extent along arc)
          - Opacity: proportional to |cos_sim| (stronger conflicts/alignments more visible)
        - Labels: loss names around the ring, numeric cos_sim values on chords
        - Legend: explains width and opacity encoding
    """
    n = len(loss_names)
    if n < 2:
        print(f"Skipping chord diagram: need at least 2 losses, got {n}")
        return

    _fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Color map: red (-1) → yellow (0) → green (+1)
    colors_diverging = ["#970800", "#ffce49", "#00481F"]
    cmap = LinearSegmentedColormap.from_list("conflict", colors_diverging, N=256)
    norm_color = Normalize(vmin=-1, vmax=1)

    norm_for_width = Normalize(vmin=0, vmax=np.max(grad_norms) if np.max(grad_norms) > 0 else 1)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Ring geometry
    ring_radius = 1.0
    arc_width = 0.12
    chord_r = ring_radius - arc_width / 2  # chords attach at inner face of arc

    # ── Outer ring arcs and labels ──
    for i, (angle, name) in enumerate(zip(angles, loss_names, strict=True)):
        start_angle = np.degrees(angle - np.pi / (2 * n))
        end_angle = np.degrees(angle + np.pi / (2 * n))
        wedge = patches.Wedge(
            (0, 0),
            ring_radius + arc_width / 2,
            start_angle,
            end_angle,
            width=arc_width,
            facecolor="lightgray",
            edgecolor="darkgray",
            linewidth=1,
        )
        ax.add_patch(wedge)

        # Gradient norm label inside arc
        norm_label_r = ring_radius
        norm_rotation = np.degrees(angle) + 180 if np.cos(angle) < 0 else np.degrees(angle)
        ax.text(
            norm_label_r * np.cos(angle),
            norm_label_r * np.sin(angle),
            f"{grad_norms[i]:.2e}",
            ha="center",
            va="center",
            fontsize=8,
            color="dimgray",
            rotation=norm_rotation,
        )

        # Loss name label outside ring
        label_r = ring_radius + 0.32
        rotation = np.degrees(angle) + 180 if np.cos(angle) < 0 else np.degrees(angle)
        ax.text(
            label_r * np.cos(angle),
            label_r * np.sin(angle),
            name,
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            rotation=rotation,
        )

    # ── Pre-compute chord slot positions along each arc ──
    # Each node's chords are laid out side by side within its arc span.
    # Chords at node i are sorted by target angle (clockwise from i) for clean ordering.
    chord_slots: dict[tuple[int, int], tuple[float, float]] = {}  # (i, j) -> (left_angle, right_angle) at node i

    arc_span = 2 * np.pi / n
    margin = arc_span * 0.15  # leave 15% margin on each side of the arc

    for i in range(n):
        chord_width_i = 0.02 + norm_for_width(grad_norms[i]) * 0.08  # angular width of each chord slot at node i
        others = sorted(range(n), key=lambda j: (angles[j] - angles[i]) % (2 * np.pi), reverse=True)
        others = [j for j in others if j != i]

        n_chords = len(others)
        gap = chord_width_i * 0.15  # small gap between adjacent chords
        total_needed = n_chords * chord_width_i + (n_chords - 1) * gap
        available = arc_span - 2 * margin

        # Scale down uniformly if chords don't fit within the arc span
        if total_needed > available:
            scale = available / total_needed
            chord_width_i *= scale
            gap *= scale
            total_needed = available

        start_angle = angles[i] - total_needed / 2
        for k, j in enumerate(others):
            left = start_angle + k * (chord_width_i + gap)
            right = left + chord_width_i
            chord_slots[(i, j)] = (left, right)

    # ── Draw tapered chord ribbons using pre-computed slots ──
    ribbon_codes = [
        MplPath.MOVETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
    ]

    for i in range(n):
        for j in range(i + 1, n):
            cos_sim = cos_sim_matrix[i, j]
            angle_i = angles[i]
            angle_j = angles[j]

            left_i, right_i = chord_slots[(i, j)]
            left_j, right_j = chord_slots[(j, i)]

            # Four arc-face points: two at each end of the ribbon
            p_il = (chord_r * np.cos(left_i), chord_r * np.sin(left_i))
            p_ir = (chord_r * np.cos(right_i), chord_r * np.sin(right_i))
            p_jl = (chord_r * np.cos(left_j), chord_r * np.sin(left_j))
            p_jr = (chord_r * np.cos(right_j), chord_r * np.sin(right_j))

            # Bezier control point — bows inward, flatter for strong conflict/alignment
            mid_angle = (angle_i + angle_j) / 2
            ctrl_r = chord_r * 0.4 * (1 - abs(cos_sim))
            ctrl = (ctrl_r * np.cos(mid_angle), ctrl_r * np.sin(mid_angle))

            color = cmap(norm_color(cos_sim))
            alpha = 0.3 + abs(cos_sim) * 0.7

            # p_ir -> ctrl -> p_jl (one curved edge), then p_jr -> ctrl -> p_il (other edge)
            ribbon_verts = [p_ir, ctrl, p_jl, p_jr, ctrl, p_il]
            ax.add_patch(
                patches.PathPatch(
                    MplPath(ribbon_verts, ribbon_codes),
                    facecolor=color,
                    edgecolor="none",
                    alpha=alpha,
                    zorder=1,
                )
            )

            # Numeric cos_sim label at chord midpoint
            mid_x = (p_ir[0] + p_jl[0]) / 2 * 0.7 + ctrl[0] * 0.3
            mid_y = (p_ir[1] + p_jl[1]) / 2 * 0.7 + ctrl[1] * 0.3
            ax.text(
                mid_x,
                mid_y,
                f"{cos_sim:.2f}",
                fontsize=8,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
                zorder=2,
            )

    # ── Colorbar ──
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_color)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, shrink=0.55, aspect=20)
    cbar.set_label("Cosine Similarity", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # ── Legend for visual encoding ──
    legend_text = "Chord width  →  gradient magnitude\nChord opacity  →  |cosine similarity|"
    ax.text(
        -1.55,
        -1.5,
        legend_text,
        fontsize=9,
        va="bottom",
        ha="left",
        color="dimgray",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.8, "edgecolor": "lightgray"},
    )

    ax.set_title(
        f"Gradient Conflict Chord Diagram — Epoch {epoch}",
        fontsize=15,
        weight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
