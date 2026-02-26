"""
SolarLossEstimator: Overview

Computes a solar-exposure loss on a differentiable mesh and (a) returns the selected solar area in m^2 under an area budget and (b) couples its gradient with the model SDF via a surrogate term. Optionally logs areas and renders images to Weights & Biases (wandb).

Structure:
1) Forward loss
   - __init__: sets device/config, sun direction, initializes state.
   - _initialize_all_attributes: resets caches/temporary state.
   - solar_loss: top-level orchestration per epoch.
   - _compute_forward_loss: builds mesh from model, clones vertices with grad, calls SolarMeshLossFn.apply to compute selected solar area (m^2) under an area budget.
   - _compute_forward_selection_for_logging: recomputes the same greedy selection without grad for logging; stores selected face indices and the total selected area in mm^2.

2) Backward/Surrogate
   - SolarMeshLossFn.forward/backward: forward computes per-face areas and illumination quality (dot of unit face normal with unit sun direction), greedily selects faces until the area budget, and sums area * quality; backward distributes gradients to the vertices of the selected faces.
   - _compute_backward_and_compose_loss: retrieves vertex grads from the solar term, computes SDF and its spatial gradients, forms a detached surrogate (dlds * SDF), and composes the final loss as a bounded solar area term plus the surrogate with correct gradient flow.

3) Logging/Visualization
   - _wandb_logging: logs selected and total mesh area (in m^2); on configured epochs exports temporary PLYs, renders A/B meshes via CADVisualizer and CADVisualizerBB, logs images to wandb, and cleans up temporary files.

Units: per-face areas are computed in mm^2; the returned loss is converted to m^2.
"""

import contextlib
import math
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import trimesh

import wandb
from subprojects.wp3_solar_estimator.v2.src.kaolin_utils import get_diff_mesh
from subprojects.wp3_solar_estimator.v2.src.visualizer import CADVisualizer
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.visualizer import CADVisualizer as CADVisualizerBB


class SolarMeshLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertices, faces, sun_direction, area_budget_mm2_tensor, epsilon: float):
        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        edge_a, edge_b = v1 - v0, v2 - v0
        face_cross = torch.cross(edge_a, edge_b, dim=-1)
        face_cross_norm = torch.linalg.norm(face_cross, dim=-1)
        face_area_mm2 = 0.5 * face_cross_norm

        face_normals_unit = face_cross / (face_cross_norm.clamp_min(epsilon)).unsqueeze(-1)
        sun_direction_unit = sun_direction / (torch.linalg.norm(sun_direction).clamp_min(epsilon))

        illumination_quality = torch.clamp((face_normals_unit * sun_direction_unit).sum(-1), 0.0, 1.0)

        area_budget_mm2 = float(area_budget_mm2_tensor.item())
        quality_order_desc = torch.argsort(illumination_quality, descending=True)
        selected_face_indices_list, accumulated_area_mm2 = [], 0.0
        for face_idx in quality_order_desc.tolist():
            area_i = float(face_area_mm2[face_idx].detach())
            if area_i <= 0.0:
                continue

            if float(illumination_quality[face_idx].detach()) <= 0.0:
                break
            if accumulated_area_mm2 + area_i <= area_budget_mm2 + 1e-12:
                selected_face_indices_list.append(face_idx)
                accumulated_area_mm2 += area_i

        selected_face_indices = torch.tensor(selected_face_indices_list, device=vertices.device, dtype=torch.long)

        ctx.save_for_backward(
            vertices,
            faces,
            selected_face_indices,
            edge_a[selected_face_indices] if selected_face_indices.numel() else edge_a.new_empty((0, 3)),
            edge_b[selected_face_indices] if selected_face_indices.numel() else edge_b.new_empty((0, 3)),
            sun_direction,
        )
        ctx.epsilon = float(epsilon)

        if selected_face_indices.numel() == 0:
            return torch.zeros((), device=vertices.device, dtype=vertices.dtype)

        selected_areas_mm2 = face_area_mm2[selected_face_indices]
        selected_quality = illumination_quality[selected_face_indices]
        loss_mm2 = (selected_areas_mm2 * selected_quality).sum()  # in mm^2
        return loss_mm2 * 1e-6  # convert to m^2

    @staticmethod
    def backward(ctx, grad_output):
        vertices, faces, selected_face_indices, selected_edge_a, selected_edge_b, sun_direction = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None

        grad_vertices = torch.zeros_like(vertices)
        if selected_face_indices.numel():
            faces_selected = faces[selected_face_indices]
            idx_v0, idx_v1, idx_v2 = faces_selected[:, 0], faces_selected[:, 1], faces_selected[:, 2]

            sun_direction_unit = sun_direction / (torch.linalg.norm(sun_direction).clamp_min(1e-12))
            dloss_dcross = 0.5 * sun_direction_unit * 1e-6  # convert to m^2

            grad_edge_a = torch.cross(dloss_dcross.expand_as(selected_edge_b), selected_edge_b, dim=-1)
            grad_edge_b = torch.cross(selected_edge_a, dloss_dcross.expand_as(selected_edge_a), dim=-1)

            grad_v1 = grad_edge_a
            grad_v2 = grad_edge_b
            grad_v0 = -(grad_v1 + grad_v2)
            grad_vertices.index_add_(0, idx_v1, grad_v1)
            grad_vertices.index_add_(0, idx_v2, grad_v2)
            grad_vertices.index_add_(0, idx_v0, grad_v0)

        grad_vertices_out = grad_vertices * grad_output * 1e-6 # * 1e2
        return grad_vertices_out, None, None, None, None, None


class SolarLossEstimator:
    def __init__(self, device: str = "cpu", losses_config: Optional[dict] = None):
        self.device = device
        self.losses_config = losses_config or {}
        solar_config = self.losses_config["solar_config"]
        self.enable_wandb_logging = bool(solar_config["wandb_logging"])
        self.enable_rendering = bool(solar_config["rendering"])
        self.rendering_frequency = int(solar_config["rendering_frequency"])
        self.max_selected_area_mm2 = int(solar_config["area_budget"])
        self.normalise_loss = bool(solar_config["normalise_loss"])

        self.sun_direction_solid_angle = solar_config["solar_direction_solid_angle"]
        self.sun_direction_solid_angle = [torch.tensor(self.sun_direction_solid_angle[0], device=self.device), torch.tensor(self.sun_direction_solid_angle[1], device=self.device)]
        sin_theta = torch.sin(self.sun_direction_solid_angle[0])
        x = sin_theta * torch.cos(self.sun_direction_solid_angle[1])
        y = sin_theta * torch.sin(self.sun_direction_solid_angle[1])
        z = torch.cos(self.sun_direction_solid_angle[0])
        self.sun_dir = torch.stack([x, y, z]).to(device=self.device, dtype=torch.float32)

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

        normalisation_factor = 1.0
        if self.normalise_loss:
            normalisation_factor = 1e6/self.max_selected_area_mm2
        return self._loss_value*normalisation_factor

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
    
    def _to_device(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return tensor.to(self.device) if tensor is not None else None

    def _safe_detach(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Helper to safely detach and move to CPU if the tensor exists."""
        if tensor is not None:
            return tensor.detach().cpu()
        return None

    def _wandb_logging(self) -> None:

        # Then your call becomes very clean:
        verts = self._to_device(self._working_verts)
        faces = self._to_device(self.mesh_faces)
        sel   = self._to_device(self._selected_face_indices)

        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        a, b = v1 - v0, v2 - v0
        c = torch.cross(a, b, dim=-1)
        triangle_areas = 0.5 * torch.linalg.norm(c, dim=-1)
        selected_areas_sum = triangle_areas[sel].sum().item()
        total_area_sum = triangle_areas.sum().item()
        wandb.log(
            {
                "solar_selected_area_m2": selected_areas_sum * 1e-6,
                "total_area_of_mesh_m2": total_area_sum * 1e-6,
            },
            step=int(self.epoch_name),
        )

        if self.rendering_frequency > 0 and self.epoch_name is not None and int(self.epoch_name) % self.rendering_frequency == 0:
            epoch_val = int(self.epoch_name) # Extract once to satisfy mypy
            temp_image_path = Path(tempfile.mktemp(suffix=".png", prefix="solar_loss_vis_"))
            temp_ply_path = Path(tempfile.mktemp(suffix=".ply", prefix="solar_loss_mesh_"))
            
            # Use local variables to allow Type Narrowing
            d_verts = self._safe_detach(self._working_verts)
            d_faces = self._safe_detach(self.mesh_faces)

            # Check if both exist before rendering
            if d_verts is not None and d_faces is not None:
                mesh_vertices_np = d_verts.numpy().astype(np.float32)
                mesh_faces_np = d_faces.numpy().astype(np.int64)
                
                mesh = trimesh.Trimesh(vertices=mesh_vertices_np, faces=mesh_faces_np, process=False)
                mesh.export(temp_ply_path.as_posix())
                
                visualizer_a = CADVisualizerBB(temp_ply_path, out_image_path=temp_image_path)
                try:
                    visualizer_a.render_until_blue_detected()
                except Exception:
                    print(f"BB Visualisation Error in Epoch {self.epoch_name}")
                
                # Use _safe_detach or check _loss_value
                d_loss = self._safe_detach(self._loss_value)
                if d_loss is not None:
                    wandb.log({"solar_final_loss": float(d_loss.item())}, step=epoch_val)
                
                if temp_image_path.exists():
                    wandb.log({"bb-visualisazion": wandb.Image(str(temp_image_path))}, step=epoch_val)
            
            temp_ply_path.unlink(missing_ok=True)
            temp_image_path.unlink(missing_ok=True)

            # --- Second Mesh Visualization Logic ---
            tmp_dir = Path(tempfile.mkdtemp(prefix="solar_mesh_vis_"))
            self._tmp_dir = tmp_dir
            self._orig_ply = tmp_dir / "mesh_original.ply"
            self._sel_ply = tmp_dir / "mesh_selected.ply"
            self._out_img = tmp_dir / "render.png"

            # Re-verify and use Narrowing
            v_tens = self._safe_detach(self._working_verts)
            f_tens = self._safe_detach(self.mesh_faces)
            s_tens = self._safe_detach(self._selected_face_indices)

            if v_tens is not None and f_tens is not None:
                verts_np = v_tens.numpy().astype(np.float32)
                faces_np = f_tens.numpy().astype(np.int64)
                
                selected_faces_np = None
                if s_tens is not None:
                    # Narrowed: s_tens is now definitely a Tensor
                    selected_faces_np = faces_np[s_tens.numpy().astype(np.int64)]
                
                self._v_np, self._f_np, self._f_sel_np = verts_np, faces_np, selected_faces_np
                self._full_mesh = trimesh.Trimesh(vertices=self._v_np, faces=self._f_np, process=False)
                
                if self._full_mesh:
                    self._full_mesh.remove_unreferenced_vertices()
                    self._full_mesh.export(self._orig_ply.as_posix())

                if self._f_sel_np is not None:
                    self._sel_mesh = trimesh.Trimesh(vertices=self._v_np, faces=self._f_sel_np, process=False)
                    self._sel_mesh.remove_unreferenced_vertices()
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
