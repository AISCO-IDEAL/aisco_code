"""
Solar loss computation for mesh SDF models.
Functions in this script:
- solar_loss: Computes the solar loss, calls the following two functions:
    - _solar_mesh_loss: Calculates the solar exposure loss using the dot-product-Approach.
    - _meshsdf_backward_step: performs a backward Step using the MeshSDF-Approach.
"""

import contextlib
import tempfile
from pathlib import Path

import numpy as np
import torch
import trimesh
import wandb

from subprojects.wp3_solar_estimator.v2.src.geometric_utils import (
    compute_face_normals,
    compute_triangle_areas,
)
from subprojects.wp3_solar_estimator.v2.src.kaolin_utils import get_diff_mesh
from subprojects.wp3_solar_estimator.v2.src.visualizer import CADVisualizer


class SolarLossEstimator:
    def __init__(self, device: str = "cpu", area_budget: float = 6e6):
        self.device = device
        self.area_budget = area_budget

    def solar_loss(self, model) -> torch.Tensor:
        # Store state for subsequent private methods
        self.model = model
        self.verts, self.faces = get_diff_mesh(model)
        model.zero_grad(set_to_none=True)
        loss = self._meshsdf_backward_step()
        return loss

    def _solar_mesh_loss(self) -> torch.Tensor:
        assert self._working_verts is not None and self.faces is not None
        face_normals = compute_face_normals(self._working_verts, self.faces)
        s = torch.tensor([-1.0, 0.0, 1.0], dtype=face_normals.dtype, device=self.device)
        s = s / (torch.linalg.norm(s) + 1e-12)
        dot_products = torch.clamp((face_normals * s).sum(dim=1), min=0.0)

        triangle_areas = compute_triangle_areas(self._working_verts, self.faces)
        sorted_vals, sorted_indices = torch.sort(dot_products, descending=True)
        sorted_areas = triangle_areas[sorted_indices]
        cum_area = torch.cumsum(sorted_areas, dim=0)
        select_mask = (cum_area <= self.area_budget) & (sorted_vals > 0)

        selected_vals = sorted_vals[select_mask]
        selected_areas = sorted_areas[select_mask]
        total_power = torch.sum(selected_vals * selected_areas)

        try:
            with torch.no_grad():
                selected_face_indices = sorted_indices[select_mask]
                if selected_face_indices.numel() > 0:
                    self._selected_face_indices = selected_face_indices
                    self._visualize_and_log_selected_faces()
        except Exception:
            pass
        return total_power * 1e-6

    def _meshsdf_backward_step(self) -> torch.Tensor:
        assert self.verts is not None and self.faces is not None and self.model is not None
        v = self.verts.detach().clone().requires_grad_(True)
        self._working_verts = v
        loss = self._solar_mesh_loss()
        (dldv,) = torch.autograd.grad(loss, v, create_graph=False, retain_graph=True)

        s = self.model(v).view(-1)
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

    def _visualize_and_log_selected_faces(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="solar_mesh_vis_"))
        self._tmp_dir = tmp_dir
        self._orig_ply = tmp_dir / "mesh_original.ply"
        self._sel_ply = tmp_dir / "mesh_selected.ply"
        self._out_img = tmp_dir / "render.png"

        assert self._working_verts is not None and self.faces is not None and self._selected_face_indices is not None
        v_np = self._working_verts.detach().cpu().numpy().astype(np.float32)
        f_np = self.faces.detach().cpu().numpy().astype(np.int64)
        f_sel_np = f_np[self._selected_face_indices.detach().cpu().numpy()]
        self._v_np, self._f_np, self._f_sel_np = v_np, f_np, f_sel_np

        self._full_mesh = trimesh.Trimesh(vertices=self._v_np, faces=self._f_np, process=False)
        self._sel_mesh = trimesh.Trimesh(vertices=self._v_np, faces=self._f_sel_np, process=False)

        try:
            if self._full_mesh is not None:
                self._full_mesh.remove_unreferenced_vertices()
            if self._sel_mesh is not None:
                self._sel_mesh.remove_unreferenced_vertices()
        except Exception:
            pass

        try:
            if self._full_mesh is not None and self._orig_ply is not None:
                self._full_mesh.export(self._orig_ply.as_posix())
            if self._sel_mesh is not None and self._sel_ply is not None:
                self._sel_mesh.export(self._sel_ply.as_posix())
        except Exception:
            pass

        assert self._orig_ply is not None and self._sel_ply is not None and self._out_img is not None
        visualizer = CADVisualizer(
            ply_path_a=self._orig_ply.as_posix(),
            ply_path_b=self._sel_ply.as_posix(),
            out_image_path=self._out_img.as_posix(),
        )
        image_path, ok, attempts = visualizer.render_until_blue_detected()
        self._image_path = Path(image_path)
        self._render_ok = ok
        self._render_attempts = attempts

        if self._image_path is not None:
            wandb.log({"solar_mesh_visualization": wandb.Image(str(self._image_path))})

        for p in (self._orig_ply, self._sel_ply, self._image_path):
            try:
                if p is not None and Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass
        with contextlib.suppress(Exception):
            if self._tmp_dir is not None:
                self._tmp_dir.rmdir()
