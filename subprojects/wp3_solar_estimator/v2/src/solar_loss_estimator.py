"""
SolarLossEstimator: Overview and method-level functionality

This module defines the SolarLossEstimator class, which computes a solar exposure-based loss on a differentiable mesh, backpropagates gradients to the mesh vertices via a custom autograd function, and optionally visualizes and logs intermediate and final results to Weights & Biases (wandb).

The estimator is organized into three major sections:

1) Forward loss calculation
   - __init__: Initializes device, configuration, sun direction, and calls attribute setup.
   - _initialize_all_attributes: Resets internal state and caches used across computation and visualization.
   - solar_loss: Orchestrates the entire process for a single epoch: forward loss computation, selection logging, optional visualization, backward/surrogate composition, and BB visualization.
   - _compute_forward_loss: Builds a differentiable mesh from the model, prepares a clone of vertices with gradients enabled, and calls the custom autograd function to compute the solar-exposed area (in m^2) subject to an area budget of selected faces.
   - _compute_forward_selection_for_logging: Recomputes (without grad) the selection of faces under the same criterion for logging and visualization, and stores selected indices and the total selected area in mm^2.
   - _SolarMeshLossFn.forward: Given vertices, faces, a sun direction, and an area budget, it computes per-face areas and illumination quality via face normals aligned with the sun direction. Faces are greedily selected up to the budget, summing area * quality. Returns the selected-area-weighted loss in square meters.

2) Backward Gradient calculation
   - _SolarMeshLossFn.backward: Backpropagates gradients to vertex positions for the selected faces. It derives gradients from the area * quality term with respect to edges/vertices and accumulates per-vertex gradients, scaled appropriately.
   - _compute_backward_and_compose_loss: Pulls vertex gradients from the forward solar area term, computes SDF values and their spatial gradients via the model, and forms a surrogate loss coupling the solar gradient signal with the SDF. Composes the final loss as the bounded solar area term plus a detached surrogate to maintain proper gradients.

3) WandB-Logging
   - _maybe_visualize_and_log: Conditionally triggers rendering/visualization based on frequency and configuration; otherwise logs scalar area metrics.
   - _log_bb_visualization: Exports the current mesh to PLY, renders via a black-box CAD visualizer, and logs the final loss and image to wandb, then cleans temporary files.
   - _visualize_and_log_selected_faces: Creates temporary files, prepares original and selected-face meshes, renders them, logs the image, and cleans up.
   - _prepare_mesh_files: Converts tensors to numpy, builds trimesh objects for the full mesh and the selected faces, removes unreferenced vertices, and exports PLYs for visualization.
   - _render_and_log: Runs the CAD visualizer for A/B meshes and, if enabled, logs the resulting image to wandb. Falls back to logging area metrics if image logging fails.
   - _log_selected_area_metrics: Logs aggregate selected and total mesh area (in m^2) to wandb.
   - _cleanup_temp: Deletes any temporary artifacts (PLYs, images, directories) created during visualization.
"""

import contextlib
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
import wandb

from subprojects.wp3_solar_estimator.v2.src.kaolin_utils import get_diff_mesh
from subprojects.wp3_solar_estimator.v2.src.visualizer import CADVisualizer
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.visualizer import CADVisualizer as CADVisualizerBB


class SolarMeshLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, f, sun_dir, area_budget_tensor, eps: float):
        v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        a, b = v1 - v0, v2 - v0
        c = torch.cross(a, b, dim=-1)  # unnormalized face normal vector
        c_norm = torch.linalg.norm(c, dim=-1)  # ||c|| = 2 * area
        area = 0.5 * c_norm  # face area

        # Unit face normals and unit sun direction
        n = c / (c_norm.clamp_min(eps)).unsqueeze(-1)
        s = sun_dir / (torch.linalg.norm(sun_dir).clamp_min(eps))

        # Quality in [0,1] using dot(face_normal, sun_vector)
        quality = torch.clamp((n * s).sum(-1), 0.0, 1.0)

        # Select faces under area budget (greedy by highest quality)
        area_budget = float(area_budget_tensor.item())
        order = torch.argsort(quality, descending=True)
        selected, acc_area = [], 0.0
        for i in order.tolist():
            ai = float(area[i].detach())
            if ai <= 0.0:
                continue
            # quality <= 0 contributes nothing; break due to sorted order
            if float(quality[i].detach()) <= 0.0:
                break
            if acc_area + ai <= area_budget + 1e-12:
                selected.append(i)
                acc_area += ai

        sel_idx = torch.tensor(selected, device=v.device, dtype=torch.long)

        # Save tensors for backward (work only on selected faces)
        ctx.save_for_backward(
            v,
            f,
            sel_idx,
            a[sel_idx] if sel_idx.numel() else a.new_empty((0, 3)),
            b[sel_idx] if sel_idx.numel() else b.new_empty((0, 3)),
            sun_dir,
        )
        ctx.eps = float(eps)

        # Loss = sum(area * quality) over selected faces
        if sel_idx.numel() == 0:
            return torch.zeros((), device=v.device, dtype=v.dtype)

        area_sel = area[sel_idx]
        quality_sel = quality[sel_idx]
        loss_mm2 = (area_sel * quality_sel).sum()  # in mm^2
        return loss_mm2 * 1e-6  # convert to m^2

    @staticmethod
    def backward(ctx, g):
        v, f, sel_idx, a_sel, b_sel, sun_dir = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None

        gv = torch.zeros_like(v)
        if sel_idx.numel():
            fs = f[sel_idx]
            i0, i1, i2 = fs[:, 0], fs[:, 1], fs[:, 2]

            s_unit = sun_dir / (torch.linalg.norm(sun_dir).clamp_min(1e-12))
            dl_dc = 0.5 * s_unit * 1e-6  # convert to m^2

            g_a = torch.cross(dl_dc.expand_as(b_sel), b_sel, dim=-1)
            g_b = torch.cross(a_sel, dl_dc.expand_as(a_sel), dim=-1)

            g1 = g_a
            g2 = g_b
            g0 = -(g1 + g2)
            gv.index_add_(0, i1, g1)
            gv.index_add_(0, i2, g2)
            gv.index_add_(0, i0, g0)

        gv_out = gv * g * 1e-4
        return gv_out, None, None, None, None, None


class SolarLossEstimator:
    def __init__(self, device: str = "cpu", losses_config: dict = None):
        self.device = device
        self.losses_config = losses_config or {}
        solar_config = self.losses_config["solar_config"]
        self.enable_wandb_logging = bool(solar_config["wandb_logging"])
        self.enable_rendering = bool(solar_config["rendering"])
        self.rendering_frequency = int(solar_config["rendering_frequency"])
        self.max_selected_area_mm2 = int(solar_config["area_budget"])
        sqrt2 = math.sqrt(2.0)
        self.sun_dir = torch.tensor([-sqrt2, 0.0, sqrt2], device=self.device, dtype=torch.float32)

        self._loss_value: torch.Tensor = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self._initialize_all_attributes()

    def _initialize_all_attributes(self) -> None:
        self.clamp_backfaces: bool = True
        self.eps: float = 1e-12

        self._SolarMeshLossFn = SolarMeshLossFn()
        self.epoch_name: int | None = None
        self.mesh_vertices: torch.Tensor | None = None
        self.mesh_faces: torch.Tensor | None = None
        self._working_verts: torch.Tensor | None = None
        self._selected_face_indices: torch.Tensor | None = None
        self._selected_area_mm2: torch.Tensor | None = None
        self._solar_area_m2: torch.Tensor | None = None
        self._loss_value: torch.Tensor | None = None
        self._tmp_dir: Path | None = None
        self._orig_ply: Path | None = None
        self._sel_ply: Path | None = None
        self._out_img: Path | None = None
        self._image_path: Path | None = None
        self._render_ok: bool | None = None
        self._render_attempts: int = 0
        self._area_budget_tensor: torch.Tensor | None = None
        self._v_np: np.ndarray | None = None
        self._f_np: np.ndarray | None = None
        self._f_sel_np: np.ndarray | None = None
        self._full_mesh: trimesh.Trimesh | None = None
        self._sel_mesh: trimesh.Trimesh | None = None

    def solar_loss(self, model: Any, epoch_name: int) -> torch.Tensor:
        self.model = model
        self.epoch_name = int(epoch_name)

        # --- Forward loss calculation ---
        try:
            self._compute_forward_loss()
            self._compute_forward_selection_for_logging()
            # --- Backward Gradient calculation ---
            self._compute_backward_and_compose_loss()
            # --- WandB-Logging ---
            if self.enable_wandb_logging:
                self._wandb_logging()
        except Exception as e:
            print(f"Error in SolarLossEstimator.solar_loss: {e}")
            self._loss_value = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        return self._loss_value

    # --- 1) Forward loss calculation ---
    def _compute_forward_loss(self) -> None:
        self.mesh_vertices, self.mesh_faces = get_diff_mesh(self.model)
        self.mesh_vertices = self.mesh_vertices.to(self.device)
        self.mesh_faces = self.mesh_faces.to(self.device)
        if hasattr(self.model, "zero_grad"):
            self.model.zero_grad(set_to_none=True)

        working_vertices = self.mesh_vertices.detach().clone().to(self.device).requires_grad_(True)
        self._working_verts = working_vertices
        area_budget_mm2 = float(self.max_selected_area_mm2 or 0.0)
        self._area_budget_tensor = torch.tensor(area_budget_mm2, device=self.device, dtype=working_vertices.dtype)
        self._solar_area_m2 = self._SolarMeshLossFn.apply(
            self._working_verts, self.mesh_faces, self.sun_dir, self._area_budget_tensor, float(self.eps)
        )

    def _compute_forward_selection_for_logging(self) -> None:
        if self._working_verts is None or self.mesh_faces is None or self.sun_dir is None:
            return
        working_verts = self._working_verts
        faces = self.mesh_faces
        sun_dir = self.sun_dir
        with torch.no_grad():
            face_v0, face_v1, face_v2 = (
                working_verts[faces[:, 0]],
                working_verts[faces[:, 1]],
                working_verts[faces[:, 2]],
            )
            unit_normals = torch.cross(face_v1 - face_v0, face_v2 - face_v0, dim=-1)
            cross_norm = torch.linalg.norm(unit_normals, dim=-1)
            face_area_mm2 = 0.5 * cross_norm
            unit_normals = unit_normals / (cross_norm.clamp_min(self.eps)).unsqueeze(-1)
            unit_sun_dir = sun_dir / (torch.linalg.norm(sun_dir).clamp_min(self.eps))
            illumination_quality = (unit_normals * unit_sun_dir).sum(-1)
            sorted_idx = torch.argsort(illumination_quality, descending=True)
            selected_indices_list: list[int] = []
            accumulated_area = 0.0
            area_budget_mm2 = float(self._area_budget_tensor.item()) if self._area_budget_tensor is not None else 0.0
            for idx in sorted_idx.tolist():
                current_area = float(face_area_mm2[idx].cpu())
                if current_area <= 0.0:
                    continue
                if float(illumination_quality[idx].cpu()) <= 0.0:
                    break
                if accumulated_area + current_area <= area_budget_mm2 + 1e-12:
                    selected_indices_list.append(idx)
                    accumulated_area += current_area
            selected_indices = torch.tensor(selected_indices_list, device=self.device, dtype=torch.long)
            self._selected_face_indices = selected_indices
            self._selected_area_mm2 = (
                face_area_mm2[selected_indices].sum()
                if selected_indices.numel()
                else torch.tensor(0.0, device=working_verts.device)
            )

    def _compute_backward_and_compose_loss(self) -> None:
        if self._solar_area_m2 is None or self._working_verts is None:
            self._loss_value = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            return
        vertex_grad = torch.autograd.grad(self._solar_area_m2, self._working_verts, create_graph=False, retain_graph=True)[0]
        vertex_grad = torch.nan_to_num(vertex_grad, nan=0.0, posinf=0.0, neginf=0.0)
        sdf_values = self.model(self._working_verts).view(-1)
        grad_sdf_xyz = torch.autograd.grad(
            sdf_values, self._working_verts, grad_outputs=torch.ones_like(sdf_values), create_graph=True, retain_graph=True
        )[0]
        grad_sdf_xyz = torch.nan_to_num(grad_sdf_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        dlds = torch.tanh(((vertex_grad * grad_sdf_xyz).sum(dim=-1)) / 1e-6)
        surrogate = (dlds.detach() * sdf_values).sum()
        if not surrogate.requires_grad:
            surrogate = surrogate + self.model(self._working_verts).sum() * 0.0
        bounded_area_m2 = torch.clamp(self._solar_area_m2, 0.0, 6.0)
        self._loss_value = surrogate + (bounded_area_m2 - surrogate.detach())

    def _wandb_logging(self) -> None:
        verts = self._working_verts.to(self.device)
        faces = self.mesh_faces.to(self.device)
        sel = self._selected_face_indices.to(self.device)
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        a, b = v1 - v0, v2 - v0
        c = torch.cross(a, b, dim=-1)
        triangle_areas = 0.5 * torch.linalg.norm(c, dim=-1)
        selected_areas_sum = triangle_areas[sel].sum().item()
        total_area_sum = triangle_areas.sum().item()
        print("got here")
        wandb.log(
            {
                "solar_selected_area_m2": selected_areas_sum * 1e-6,
                "total_area_of_mesh_m2": total_area_sum * 1e-6,
            },
            step=int(self.epoch_name),
        )

        if self.rendering_frequency > 0 and int(self.epoch_name) % self.rendering_frequency == 0:
            temp_image_path = Path(tempfile.mktemp(suffix=".png", prefix="solar_loss_vis_"))
            temp_ply_path = Path(tempfile.mktemp(suffix=".ply", prefix="solar_loss_mesh_"))
            mesh_vertices_np = self.mesh_vertices.detach().cpu().numpy().astype(np.float32)
            mesh_faces_np = self.mesh_faces.detach().cpu().numpy().astype(np.int64)
            mesh = trimesh.Trimesh(vertices=mesh_vertices_np, faces=mesh_faces_np, process=False)
            mesh.export(temp_ply_path.as_posix())
            visualizer_a = CADVisualizerBB(temp_ply_path, out_image_path=temp_image_path)
            try:    
                visualizer_a.render_until_blue_detected()
            except Exception as e:
                print("BB Visualisation Error in Epoch", self.epoch_name)
            wandb.log({"solar_final_loss": float(self._loss_value.detach().cpu().item())}, step=int(self.epoch_name))
            wandb.log({"bb-visualisazion": wandb.Image(str(temp_image_path))}, step=int(self.epoch_name))
            temp_ply_path.unlink(missing_ok=True)
            temp_image_path.unlink(missing_ok=True)

            tmp_dir = Path(tempfile.mkdtemp(prefix="solar_mesh_vis_"))
            self._tmp_dir = tmp_dir
            self._orig_ply = tmp_dir / "mesh_original.ply"
            self._sel_ply = tmp_dir / "mesh_selected.ply"
            self._out_img = tmp_dir / "render.png"

            verts_np = self._working_verts.detach().cpu().numpy().astype(np.float32)
            faces_np = self.mesh_faces.detach().cpu().numpy().astype(np.int64)
            selected_faces_np = faces_np[self._selected_face_indices.detach().cpu().numpy()]
            self._v_np, self._f_np, self._f_sel_np = verts_np, faces_np, selected_faces_np
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

            visualizerb = CADVisualizer(
                ply_path_a=self._orig_ply.as_posix(),
                ply_path_b=self._sel_ply.as_posix(),
                out_image_path=self._out_img.as_posix(),
            )
            image_path, ok, attempts = visualizerb.render_until_blue_detected()
            self._image_path = Path(image_path) if isinstance(image_path, (str, Path)) else None
            self._render_ok = None if ok is None else bool(ok)
            self._render_attempts = int(attempts)
            if self.enable_wandb_logging and self._image_path is not None and self.epoch_name is not None:
                try:
                    wandb.log({"solar_mesh_visualization": wandb.Image(str(self._image_path))}, step=int(self.epoch_name))
                except Exception as e:
                    print(f"failed to log solar mesh visulisation in epoch {self.epoch_name}: {e}")
                
            for p in (self._orig_ply, self._sel_ply, self._image_path):
                if p is not None and Path(p).exists():
                    Path(p).unlink()
            with contextlib.suppress(Exception):
                if self._tmp_dir is not None:
                    self._tmp_dir.rmdir()
