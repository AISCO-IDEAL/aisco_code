"""Solar loss computation for mesh SDF models.
Functions in this script:
- solar_loss: Computes the solar loss, calls the following two functions:
    - solar_mesh_loss: Calculates the solar exposure loss using the dot-product-Approach.
    - meshsdf_backward_step: performs a backward Step using the MeshSDF-Approach.
"""

import contextlib
import importlib
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import trimesh

from subprojects.wp3_solar_estimator.v2.src.geometric_utils import compute_face_normals, compute_triangle_areas
from subprojects.wp3_solar_estimator.v2.src.kaolin_utils import get_diff_mesh
from subprojects.wp3_solar_estimator.v2.src.visualizer import CADVisualizer


def meshsdf_backward_step(
    sdf_model,
    verts: torch.Tensor,
    faces: torch.Tensor,
    loss_fn: Callable[..., torch.Tensor],
    area_budget: float = 6e6,
    device="cpu",
) -> torch.Tensor:
    v = verts.detach().clone().requires_grad_(True)
    loss = loss_fn(v, faces, area_budget, device)
    (dldv,) = torch.autograd.grad(loss, v, create_graph=False, retain_graph=True)
    s = sdf_model(v).view(-1)
    (grad_xyz,) = torch.autograd.grad(
        s,
        v,
        grad_outputs=torch.ones_like(s),
        create_graph=True,
        retain_graph=True,
    )
    eps = 1e-12
    n = grad_xyz / (grad_xyz.norm(dim=-1, keepdim=True) + eps)
    dlds = -(dldv * n).sum(dim=-1)
    surrogate = (dlds.detach() * s).sum()
    surrogate.backward()
    return loss


def visualize_and_log_selected_faces(
    verts: torch.Tensor,
    faces: torch.Tensor,
    selected_face_indices: torch.Tensor,
    prefix: str = "solar_mesh_vis_",
) -> None:
    """Save the full mesh and the selected-face mesh as PLYs, render them, and log to W&B.

    - Exports two temporary PLYs (original and selected faces)
    - Renders with CADVisualizer.render_until_blue_detected
    - Logs the rendered image and metadata to Weights & Biases
    - Cleans up temporary files
    """

    # Debug helper
    def _dbg(msg: str, **info) -> None:
        try:
            info_str = " ".join(f"{k}={v}" for k, v in info.items())
            print(f"[solar_vis_debug] {msg} {info_str}".strip(), flush=True)
        except Exception:
            print(f"[solar_vis_debug] {msg}", flush=True)

    _dbg("enter visualize_and_log_selected_faces")
    try:
        _dbg(
            "inputs",
            verts_shape=tuple(verts.shape),
            verts_dtype=str(verts.dtype),
            verts_device=str(getattr(verts, "device", "n/a")),
            faces_shape=tuple(faces.shape),
            faces_dtype=str(faces.dtype),
            faces_device=str(getattr(faces, "device", "n/a")),
            selected_shape=tuple(selected_face_indices.shape),
            selected_dtype=str(selected_face_indices.dtype),
            selected_device=str(getattr(selected_face_indices, "device", "n/a")),
        )
        with contextlib.suppress(Exception):
            _dbg(
                "verts stats",
                nan=bool(torch.isnan(verts).any().item()),
                inf=bool(torch.isinf(verts).any().item()),
                vmin=float(torch.nanmin(verts).item()),
                vmax=float(torch.nanmax(verts).item()),
            )
        with contextlib.suppress(Exception):
            _dbg(
                "faces stats",
                fmin=int(torch.min(faces).item()),
                fmax=int(torch.max(faces).item()),
            )
        with contextlib.suppress(Exception):
            fcount = int(faces.shape[0])
            uniq = torch.unique(selected_face_indices)
            oob_low = int((selected_face_indices < 0).sum().item())
            oob_high = int((selected_face_indices >= fcount).sum().item())
            _dbg(
                "selected indices",
                count=int(selected_face_indices.numel()),
                unique=int(uniq.numel()),
                min=int(selected_face_indices.min().item()),
                max=int(selected_face_indices.max().item()),
                fcount=fcount,
                oob_low=oob_low,
                oob_high=oob_high,
            )
    except Exception as e:
        _dbg("input debug failed", error=str(e))

    tmp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    orig_ply = tmp_dir / "mesh_original.ply"
    sel_ply = tmp_dir / "mesh_selected.ply"
    out_img = tmp_dir / "render.png"
    _dbg(
        "paths",
        tmp_dir=str(tmp_dir),
        orig_ply=str(orig_ply),
        sel_ply=str(sel_ply),
        out_img=str(out_img),
    )

    # Convert tensors to numpy arrays on CPU
    v_np = verts.detach().cpu().numpy().astype(np.float32)
    f_np = faces.detach().cpu().numpy().astype(np.int64)
    f_sel_np = f_np[selected_face_indices.detach().cpu().numpy()]
    _dbg(
        "numpy arrays",
        v_np_shape=tuple(v_np.shape),
        f_np_shape=tuple(f_np.shape),
        f_sel_np_shape=tuple(f_sel_np.shape),
    )

    # Build and export meshes
    full_mesh = trimesh.Trimesh(vertices=v_np, faces=f_np, process=False)
    sel_mesh = trimesh.Trimesh(vertices=v_np, faces=f_sel_np, process=False)
    with contextlib.suppress(Exception):
        _dbg(
            "full mesh built",
            verts=len(full_mesh.vertices),
            faces=len(full_mesh.faces),
            watertight=bool(getattr(full_mesh, "is_watertight", False)),
        )
        _dbg(
            "sel mesh built",
            verts=len(sel_mesh.vertices),
            faces=len(sel_mesh.faces),
            watertight=bool(getattr(sel_mesh, "is_watertight", False)),
        )
        _dbg(
            "full bounds",
            min=list(np.asarray(full_mesh.bounds[0]).astype(float)),
            max=list(np.asarray(full_mesh.bounds[1]).astype(float)),
        )
        _dbg(
            "sel bounds",
            min=list(np.asarray(sel_mesh.bounds[0]).astype(float)),
            max=list(np.asarray(sel_mesh.bounds[1]).astype(float)),
        )

    # Ensure clean topology for export (remove unreferenced vertices)
    try:
        full_mesh.remove_unreferenced_vertices()
        sel_mesh.remove_unreferenced_vertices()
        _dbg(
            "post-clean",
            full_verts=len(full_mesh.vertices),
            full_faces=len(full_mesh.faces),
            sel_verts=len(sel_mesh.vertices),
            sel_faces=len(sel_mesh.faces),
        )
    except Exception as e:
        _dbg("clean topology failed", error=str(e))

    try:
        full_mesh.export(orig_ply.as_posix())
        sel_mesh.export(sel_ply.as_posix())
        _dbg(
            "exported",
            orig_exists=orig_ply.exists(),
            sel_exists=sel_ply.exists(),
        )
        with contextlib.suppress(Exception):
            _dbg(
                "export sizes",
                orig_size=int(orig_ply.stat().st_size),
                sel_size=int(sel_ply.stat().st_size),
            )
    except Exception as e:
        _dbg("export failed", error=str(e))

    # Render overlay (A=original in red, B=selected in semi-transparent blue)
    _dbg("render start")
    visualizer = CADVisualizer(
        ply_path_a=orig_ply.as_posix(),
        ply_path_b=sel_ply.as_posix(),
        out_image_path=out_img.as_posix(),
    )
    image_path, ok, attempts = visualizer.render_until_blue_detected()
    _dbg("render result", image=str(image_path), ok=ok, attempts=attempts)
    with contextlib.suppress(Exception):
        _dbg("render image file", exists=Path(image_path).exists(), size=int(Path(image_path).stat().st_size))

    # Log to Weights & Biases if available (do not fail the training if not present)
    try:
        wandb = importlib.import_module("wandb")
        _dbg("wandb import ok")
        wandb.log(
            {
                "solar_mesh_visualization": wandb.Image(image_path),
                "solar_vis_ok": ok,
                "solar_vis_attempts": attempts,
            }
        )
        _dbg("wandb log ok")
    except Exception as e:
        _dbg("wandb logging skipped", error=str(e))

    # Best-effort cleanup of temporary files
    for p in (orig_ply, sel_ply, Path(image_path)):
        try:
            if p.exists():
                _dbg("cleanup unlink", path=str(p))
                p.unlink()
            else:
                _dbg("cleanup missing", path=str(p))
        except Exception as e:
            _dbg("cleanup unlink failed", path=str(p), error=str(e))
    try:
        tmp_dir.rmdir()
        _dbg("cleanup rmdir ok", dir=str(tmp_dir))
    except Exception as e:
        _dbg("cleanup rmdir skipped", dir=str(tmp_dir), error=str(e))


def solar_mesh_loss(verts: torch.Tensor, faces: torch.Tensor, area_budget: float = 6e6, device="cpu") -> torch.Tensor:
    face_normals = compute_face_normals(verts, faces)

    s = torch.tensor([-1.0, 0.0, 1.0], dtype=face_normals.dtype, device=device)
    s = s / (torch.linalg.norm(s) + 1e-12)

    dot_products = torch.clamp((face_normals * s).sum(dim=1), min=0.0)

    triangle_areas = compute_triangle_areas(verts, faces)

    sorted_vals, sorted_indices = torch.sort(dot_products, descending=True)
    sorted_areas = triangle_areas[sorted_indices]
    cum_area = torch.cumsum(sorted_areas, dim=0)
    select_mask = cum_area <= area_budget
    selected_vals = sorted_vals[select_mask]
    selected_areas = sorted_areas[select_mask]
    total_power = torch.sum(selected_vals * selected_areas)

    # Visualization & logging moved to helper to keep loss clean
    try:
        with torch.no_grad():
            selected_face_indices = sorted_indices[select_mask]
            if selected_face_indices.numel() > 0:
                visualize_and_log_selected_faces(verts, faces, selected_face_indices)
    except Exception:
        with contextlib.suppress(Exception):
            wandb = importlib.import_module("wandb")
            wandb.log({"solar_mesh_visualization_error": True})

    return total_power * 1e-6


def solar_loss(model, area_budget: float = 6e6, device="cpu") -> torch.Tensor:
    verts, faces = get_diff_mesh(model)
    model.zero_grad(set_to_none=True)

    loss = meshsdf_backward_step(model, verts, faces, solar_mesh_loss, area_budget=area_budget, device=device)

    return loss
