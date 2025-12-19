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
    def __init__(self, device: str = "cpu", losses_config: dict | None = None):
        self.device = device
        self.losses_config = losses_config or {}
        solar_cfg = self.losses_config.get("solar_config", {})
        self.wandb_logging = solar_cfg.get("wandb_logging", False)
        self.rendering = solar_cfg.get("rendering", False)
        self.rendering_frequency = solar_cfg.get("rendering_frequency", 1)
        self.area_budget = int(solar_cfg.get("area_budget", 0))

    def solar_loss(self, model, epoch_name) -> torch.Tensor:
        self.epoch_name = epoch_name
        self.model = model
        self.verts, self.faces = get_diff_mesh(model)
        model.zero_grad(set_to_none=True)
        loss = self._meshsdf_backward_step()
        return loss

    def _solar_mesh_loss(self) -> torch.Tensor:
        assert self._working_verts is not None and self.faces is not None
        face_normals = compute_face_normals(self._working_verts, self.faces)
        # Ensure the sunlight direction is created on the same device as face_normals to avoid device mismatch
        s = torch.tensor([-1.0, 0.0, 1.0], dtype=face_normals.dtype, device=face_normals.device)
        s = s / (torch.linalg.norm(s) + 1e-12)
        # Dot products per face (exposure); keep this differentiable
        dot_products = torch.clamp((face_normals * s).sum(dim=1), min=0.0)

        triangle_areas = compute_triangle_areas(self._working_verts, self.faces)

        # Compute hard selection (top faces under area budget) without tracking gradients
        with torch.no_grad():
            sorted_vals, sorted_indices = torch.sort(dot_products, descending=True)
            sorted_areas = triangle_areas[sorted_indices]
            cum_area = torch.cumsum(sorted_areas, dim=0)
            select_mask_sorted = (cum_area <= self.area_budget) & (sorted_vals > 0)
            selected_face_indices = sorted_indices[select_mask_sorted]

        # Build a constant mask over original face indices; gradients only flow through selected faces
        selection_mask = torch.zeros_like(dot_products)
        if selected_face_indices.numel() > 0:
            selection_mask[selected_face_indices] = 1.0
        selection_mask = selection_mask.detach()  # treat selection as constant

        # Differentiable total power but gated by constant selection
        per_face_power = dot_products * triangle_areas
        total_power = torch.sum(selection_mask * per_face_power)

        # Rendering and logging
        if self.rendering and int(self.epoch_name) % self.rendering_frequency == 0:
            with torch.no_grad():
                if selected_face_indices.numel() > 0:
                    self._selected_face_indices = selected_face_indices
                    self._visualize_and_log_selected_faces()

        if self.wandb_logging:
            with torch.no_grad():
                selected_areas_sum = (
                    triangle_areas[selected_face_indices].sum().item() if selected_face_indices.numel() > 0 else 0.0
                )
                total_area_sum = triangle_areas.sum().item()
            wandb.log(
                {
                    "solar_selected_area_m2": selected_areas_sum * 1e-6,
                    "total_area_of_mesh_m2": total_area_sum * 1e-6,
                },
                step=self.epoch_name,
            )

        return total_power * 1e-6

    def _meshsdf_backward_step(self) -> torch.Tensor:
        assert self.verts is not None and self.faces is not None and self.model is not None
        v = self.verts.detach().clone().requires_grad_(True)
        self._working_verts = v
        loss_vertices = self._solar_mesh_loss()
        (dldv,) = torch.autograd.grad(loss_vertices, v, create_graph=False, retain_graph=True)

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

        # Ensure every trainable parameter is referenced so torch.autograd.grad(loss, params)
        # does not fail on "unused parameters" during gradient logging.
        try:
            params = list(self.model.parameters()) if hasattr(self.model, "parameters") else []
            if params:
                aux = None
                for p in params:
                    if p.requires_grad and p.numel() > 0:
                        # Zero-weight reference to p keeps it in the graph but contributes nothing to the value
                        term = (p.reshape(-1)[:1] * 0.0).sum()
                        aux = term if aux is None else aux + term
                if aux is not None:
                    surrogate = surrogate + aux
        except Exception:
            # Never let logging-prep break training
            pass

        # Do not call backward here; return surrogate so the outer loop can backprop and log grads
        return surrogate

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

        if self._full_mesh is not None:
            self._full_mesh.remove_unreferenced_vertices()
        if self._sel_mesh is not None:
            self._sel_mesh.remove_unreferenced_vertices()

        if self._full_mesh is not None and self._orig_ply is not None:
            self._full_mesh.export(self._orig_ply.as_posix())
        if self._sel_mesh is not None and self._sel_ply is not None:
            self._sel_mesh.export(self._sel_ply.as_posix())

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
            wandb.log({"solar_mesh_visualization": wandb.Image(str(self._image_path))}, step=self.epoch_name)

        for p in (self._orig_ply, self._sel_ply, self._image_path):
            try:
                if p is not None and Path(p).exists():
                    Path(p).unlink()
            except Exception:
                print("try 4 failed")
        with contextlib.suppress(Exception):
            if self._tmp_dir is not None:
                self._tmp_dir.rmdir()
