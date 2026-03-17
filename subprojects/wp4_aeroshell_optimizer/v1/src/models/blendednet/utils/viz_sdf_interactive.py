"""Interactive BWB SDF visualization via TorchCAD neural3d pipeline.

Loads bwb_wing.yaml via CADProgram, meshes with skimage, displays with
TorchCAD's render_mesh_result (grey mesh + bbox + origin planes). 9
design-param sliders + resolution slider for real-time re-meshing.

Usage:
    python utils/viz_sdf_interactive.py

    # backwards-compat flags (ignored by TorchCAD path):
    python utils/viz_sdf_interactive.py --checkpoint training/adamw_2_march.pt
"""

import argparse
import os
import sys

import numpy as np
import pyvista as pv
import torch

# ensure blendednet package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ensure TorchCAD is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "TorchCAD"))

from torchcad.program import CADProgram
from torchcad.render import _add_sdf_slice_widget, render_mesh_result

PARAM_NAMES = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3"]

YAML_PATH = os.path.join(os.path.dirname(__file__), "..", "bwb_wing.yaml")


def _mesh_to_pv(mesh_result) -> pv.PolyData | None:
    """Convert TorchCAD MeshResult (batch idx 0) to pyvista PolyData."""
    verts = mesh_result.vertices[0]
    faces = mesh_result.faces[0]
    if verts is None or faces is None:
        return None
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy().astype(np.int64)
    n_faces = faces_np.shape[0]
    pv_faces = np.hstack([
        np.full((n_faces, 1), 3, dtype=np.int64), faces_np
    ]).flatten()
    mesh = pv.PolyData(verts_np, pv_faces)
    mesh.compute_normals(inplace=True)
    return mesh


def main():
    p = argparse.ArgumentParser(description="interactive BWB SDF visualizer (TorchCAD)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="(ignored) checkpoint path for backwards compat")
    p.add_argument("--wandb-run", type=str, default=None,
                   help="(ignored) wandb run path for backwards compat")
    p.add_argument("--resolution", type=int, default=None,
                   help="override YAML resolution")
    args = p.parse_args()

    # load program
    device = "cuda" if torch.cuda.is_available() else None
    prog = CADProgram.load_from_yaml(YAML_PATH, device=device)

    if args.resolution:
        prog.resolution = args.resolution

    # extract param ranges from checkpoint for slider bounds
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "training", "adamw_2_march.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    param_min = ckpt["param_min"]
    param_max = ckpt["param_max"]

    # current param values (from YAML defaults = midpoints)
    current_params = {name: float(prog.params[name]) for name in PARAM_NAMES}
    state = {"resolution": prog.resolution}

    # initial mesh
    print("computing initial mesh...")
    mesh = prog.mesh(backend="skimage")

    # render with show=False so we can add sliders before showing
    # semi-transparent mesh so xz slice is visible through it
    plotter = render_mesh_result(mesh, show=False, show_edges=False, opacity=0.3)

    # get the mesh actor for later updates
    actor_key = list(plotter.actors.keys())[0]
    actor = plotter.actors[actor_key]

    # add xz SDF slice plane — use wrapper so slice tracks current prog
    prog._build_shapes()

    def _live_sdf(points):
        """Delegate to current prog's shape (updated by sliders)."""
        return prog._shape(points)

    _add_sdf_slice_widget(
        plotter, _live_sdf, prog.xyz_min, prog.xyz_max,
        plane='xz', slice_resolution=5 * prog.resolution,
    )

    def _update_display():
        """Re-mesh with current params and update actor."""
        try:
            nonlocal prog
            prog = prog.with_params(**current_params)
            if state["resolution"] != prog.resolution:
                prog.resolution = state["resolution"]

            # rebuild shapes so _live_sdf sees updated SDF
            prog._build_shapes()

            new_mesh = prog.mesh(backend="skimage")
            pv_mesh = _mesh_to_pv(new_mesh)
            if pv_mesh is not None:
                actor.mapper.SetInputData(pv_mesh)
                plotter.render()
            else:
                print("no surface at current params")
        except Exception as e:
            print(f"mesh failed: {e}")

    def _on_param(idx, val):
        current_params[PARAM_NAMES[idx]] = val
        _update_display()

    def _on_resolution(val):
        state["resolution"] = int(val)
        _update_display()

    # add 9 parameter sliders (left column)
    for i in range(9):
        lo = param_min[i].item()
        hi = param_max[i].item()
        mid = current_params[PARAM_NAMES[i]]
        margin = (hi - lo) * 0.1
        y = 0.95 - i * 0.1
        plotter.add_slider_widget(
            callback=lambda val, idx=i: _on_param(idx, val),
            rng=(lo - margin, hi + margin),
            value=mid,
            title=PARAM_NAMES[i],
            pointa=(0.02, y),
            pointb=(0.35, y),
            style="modern",
        )

    # resolution slider (right side, bottom)
    plotter.add_slider_widget(
        callback=_on_resolution,
        rng=(32, 256),
        value=state["resolution"],
        title="resolution",
        pointa=(0.65, 0.1),
        pointb=(0.98, 0.1),
        style="modern",
    )

    plotter.camera_position = "xz"
    plotter.show()


if __name__ == "__main__":
    main()
