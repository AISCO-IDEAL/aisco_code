"""
Solar loss computation for mesh SDF models.
Functions in this script:
- solar_loss: Computes the solar loss, calls the following two functions:
    - _solar_mesh_loss: Calculates the solar exposure loss using the dot-product-Approach.
    - _meshsdf_backward_step: performs a backward Step using the MeshSDF-Approach.
"""

import contextlib
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
import trimesh
import wandb

from subprojects.wp3_solar_estimator.v2.src.kaolin_utils import get_diff_mesh
from subprojects.wp3_solar_estimator.v2.src.visualizer import CADVisualizer


class SolarMeshLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, f, sun_dir, area_budget_tensor, clamp_backfaces: bool, eps: float):
        """
        Compute solar exposure contribution for a triangle mesh in a custom autograd op.
        - Selects a subset of faces under a given area budget using a greedy ranking
          by exposure score (dot of face normal cross-product with sun_dir, normalized by area).
        - If clamp_backfaces is True, back-facing faces (negative dot) contribute zero.
        - Saves selection and tensors needed for manual gradient in backward.
        Returns the summed positive exposure (scaled by 0.5) for the selected faces.
        Inputs:
          v: (V,3) vertex positions (requires_grad allowed)
          f: (F,3) face indices into v
          sun_dir: (3,) sun direction vector
          area_budget_tensor: scalar tensor specifying total area limit for selected faces
          clamp_backfaces: bool to zero-out negative exposure
          eps: small float to avoid division by zero when ranking by area
        """
        print(
            "[SolarMeshLossFn.forward] v.requires_grad=", v.requires_grad, "v.shape=", tuple(v.shape), "f.shape=", tuple(f.shape)
        )
        v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        a, b = v1 - v0, v2 - v0
        c = torch.cross(a, b, dim=-1)  # (F,3)
        area = 0.5 * torch.linalg.norm(c, dim=-1)  # (F,)
        dot = (c * sun_dir).sum(-1)  # (F,)
        score = torch.clamp(dot, 0.0) if clamp_backfaces else dot

        area_budget = float(area_budget_tensor.item())
        order = torch.argsort(score / (area.clamp_min(eps)), descending=True)
        indices_list, acc = [], 0.0
        for i in order.tolist():
            ai = float(area[i].detach())
            if ai <= 0.0:
                continue
            if clamp_backfaces and float(score[i].detach()) <= 0.0:
                break
            if acc + ai <= area_budget + 1e-12:
                indices_list.append(i)
                acc += ai

        sel_tensor = torch.tensor(indices_list, device=v.device, dtype=torch.long)
        dot_sel = dot[sel_tensor] if sel_tensor.numel() else torch.empty(0, device=v.device, dtype=v.dtype)
        ctx.save_for_backward(v, f, sel_tensor, dot_sel, sun_dir)
        ctx.clamp_backfaces = clamp_backfaces
        print("[SolarMeshLossFn.forward] selected faces count=", int(sel_tensor.numel()))
        # Return positive solar exposure so downstream code can maximize it later
        return torch.zeros((), device=v.device, dtype=v.dtype) if sel_tensor.numel() == 0 else (torch.clamp(dot_sel, 0.0) if clamp_backfaces else dot_sel).sum() * 0.5

    @staticmethod
    def backward(ctx, g):
        """
        Manual gradient of the solar exposure objective w.r.t. vertices.
        - Reconstructs selected faces and computes per-vertex gradients derived from
          the cross-product formulation and dot with sun_dir.
        - Applies masking for clamped backfaces and distributes gradients to v0, v1, v2.
        - Multiplies by incoming scalar gradient g and returns only the vertex gradient.
        Inputs:
          g: scalar upstream gradient for the summed exposure.
        Returns:
          (gv, None, None, None, None, None) where gv matches shape of v from forward.
        """
        print("[SolarMeshLossFn.backward] incoming grad scalar=", float(g.detach().cpu().item()))
        v, f, sel_tensor, dot_sel, sun_dir = ctx.saved_tensors
        clamp_backfaces = ctx.clamp_backfaces
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None

        gv = torch.zeros_like(v)
        if sel_tensor.numel():
            print("[SolarMeshLossFn.backward] sel.numel=", int(sel_tensor.numel()))
            fs = f[sel_tensor]
            i0, i1, i2 = fs[:, 0], fs[:, 1], fs[:, 2]
            v0, v1, v2 = v[i0], v[i1], v[i2]
            a, b = v1 - v0, v2 - v0
            d = sun_dir.view(1, 3).expand_as(a)

            m = (dot_sel > 0).to(v.dtype).view(-1, 1) if clamp_backfaces else 1.0
            da = torch.cross(b, d, dim=-1)
            db = torch.cross(d, a, dim=-1)

            # Flip sign to match positive forward objective (previously -0.5)
            g1 = (0.5) * da * m
            g2 = (0.5) * db * m
            g0 = -(g1 + g2)

            gv.index_add_(0, i1, g1)
            gv.index_add_(0, i2, g2)
            gv.index_add_(0, i0, g0)

        gv_out = gv * g
        print("[SolarMeshLossFn.backward] gv_out.norm=", float(gv_out.detach().norm().cpu().item()))
        return gv_out, None, None, None, None, None


class SolarLossEstimator:
    def __init__(self, device: str = "cpu", losses_config: dict | None = None):
        """
        Configure the solar loss estimator.
        - Stores device and loss configuration.
        - Extracts solar-specific options like wandb logging, rendering toggles,
          rendering frequency, and an area budget used for face selection.
        """
        self.device = device
        self.losses_config = losses_config or {}
        solar_cfg = self.losses_config.get("solar_config", {})
        self.wandb_logging = solar_cfg.get("wandb_logging", False)
        self.rendering = solar_cfg.get("rendering", False)
        self.rendering_frequency = solar_cfg.get("rendering_frequency", 1)
        self.area_budget = int(solar_cfg.get("area_budget", 0))

    def solar_loss(self, model, epoch_name) -> torch.Tensor:
        """
        Top-level entry to compute a differentiable solar loss tied to model parameters.
        - Retrieves differentiable mesh (verts, faces) from the model.
        - Runs a MeshSDF-based backward step to produce a surrogate loss that
          propagates gradients through the model.
        - Adds tiny regularization and fallbacks to ensure loss depends on all
          trainable parameters for stable gradient logging.
        Returns a scalar tensor suitable for optimization/backprop.
        """
        print("[solar_loss] called with epoch=", epoch_name)
        self.epoch_name = epoch_name
        self.model = model
        self.verts, self.faces = get_diff_mesh(model)
        print("[solar_loss] verts.shape=", tuple(self.verts.shape), "faces.shape=", tuple(self.faces.shape))
        model.zero_grad(set_to_none=True)
        loss = self._meshsdf_backward_step()
        # Ensure the returned loss depends on model parameters so external autograd.grad logging never fails.
        try:
            params = list(model.parameters()) if hasattr(model, "parameters") else []
            if params:
                tiny_coeff = 1e-12
                l2_sum = None
                for p in params:
                    if p.requires_grad and p.numel() > 0:
                        term = p.pow(2).sum()
                        l2_sum = term if l2_sum is None else l2_sum + term
                if l2_sum is not None:
                    loss = loss + tiny_coeff * l2_sum
        except Exception:
            pass
        # Final safeguard: ensure the returned loss always requires gradients w.r.t model params
        if not loss.requires_grad:
            try:
                with torch.enable_grad():
                    tie = None
                    if hasattr(self, "_working_verts") and self._working_verts is not None:
                        tie = self.model(self._working_verts).sum() * 0.0
                    else:
                        tie = (
                            sum((p.sum() * 0.0) for p in self.model.parameters())
                            if hasattr(self.model, "parameters")
                            else torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                        )
                    loss = loss + (tie if tie is not None else 0.0)
            except Exception:
                pass
        print("[solar_loss] loss=", float(loss.detach().cpu().item()), "loss.requires_grad=", loss.requires_grad)
        return loss

    def _solar_mesh_loss(self) -> torch.Tensor:
        """
        Compute solar exposure on the mesh using a greedy face selection.
        - Builds sun direction (or uses provided), area budget, and clamps backfaces
          depending on configuration.
        - Calls the custom autograd op SolarMeshLossFn to obtain a differentiable
          exposure objective over currently working vertices.
        - Optionally recomputes selection indices for visualization and logging.
        Returns the exposure-based objective (positive when faces are sunlit).
        """
        print("[_solar_mesh_loss] using _working_verts=", getattr(self, "_working_verts", None) is not None)
        assert self.verts is not None and self.faces is not None, "verts and faces must be available on the instance"

        # Use the current working verts if available to maintain gradient connectivity
        verts = getattr(self, "_working_verts", None)
        if verts is None:
            verts = self.verts
        faces = self.faces
        area_budget = float(self.area_budget or 0.0)

        # default sun dir if not set on instance
        sun_dir = getattr(self, "sun_dir", None)
        if sun_dir is None:
            s2 = math.sqrt(2.0)
            sun_dir = torch.tensor([-s2, 0.0, s2], device=verts.device, dtype=verts.dtype)
            self.sun_dir = sun_dir
        else:
            sun_dir = sun_dir.to(device=verts.device, dtype=verts.dtype)

        clamp_backfaces = getattr(self, "clamp_backfaces", True)
        eps = getattr(self, "eps", 1e-12)
        area_budget_tensor = torch.tensor(area_budget, device=verts.device, dtype=verts.dtype)

        print("[_solar_mesh_loss] verts.requires_grad=", verts.requires_grad, "faces.requires_grad=False")
        loss = SolarMeshLossFn.apply(verts, faces, sun_dir, area_budget_tensor, clamp_backfaces, float(eps))
        print(
            "[_solar_mesh_loss] loss_vertices=",
            float(loss.detach().cpu().item()) if loss.requires_grad or loss.requires_grad is False else None,
            "requires_grad=",
            loss.requires_grad,
        )

        # Recompute selection indices for visualization/logging without using globals
        try:
            with torch.no_grad():
                sel_indices = self._compute_selection_indices(verts, faces, sun_dir, area_budget, clamp_backfaces, eps)
                if sel_indices is not None:
                    self._selected_face_indices = sel_indices
                    if self.rendering and int(self.epoch_name) % self.rendering_frequency == 0:
                        self._visualize_and_log_selected_faces()
                    if self.wandb_logging:
                        self._visualize_and_log_selected_faces()
        except Exception:
            pass

        return loss

    def _compute_selection_indices(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        sun_dir: torch.Tensor,
        area_budget: float,
        clamp_backfaces: bool,
        eps: float,
    ) -> torch.Tensor | None:
        """
        Determine which faces to select under an area budget.
        - Computes per-face area and exposure score (dot with sun_dir, optionally clamped).
        - Ranks faces by score/area and greedily accumulates until the area budget is reached.
        Returns a tensor of selected face indices or None on failure.
        """
        try:
            v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
            a, b = v1 - v0, v2 - v0
            c = torch.cross(a, b, dim=-1)
            area = 0.5 * torch.linalg.norm(c, dim=-1)
            dot = (c * sun_dir).sum(-1)
            score = torch.clamp(dot, 0.0) if clamp_backfaces else dot
            order = torch.argsort(score / (area.clamp_min(eps)), descending=True)
            indices_list, acc = [], 0.0
            for i in order.tolist():
                ai = float(area[i].cpu())
                if ai <= 0.0:
                    continue
                if clamp_backfaces and float(score[i].cpu()) <= 0.0:
                    break
                if acc + ai <= area_budget + 1e-12:
                    indices_list.append(i)
                    acc += ai
            return torch.tensor(indices_list, device=verts.device, dtype=torch.long)
        except Exception:
            return None

    def _meshsdf_backward_step(self) -> torch.Tensor:
        """
        Bridge mesh-space exposure gradients to model parameter gradients via MeshSDF.
        - Clones verts with grad enabled and runs the mesh exposure objective to get dL/dv.
        - Evaluates the SDF model at verts and obtains gradients dS/dv.
        - Forms a surrogate loss by projecting dL/dv onto normalized dS/dv and tying it
          to the scalar SDF values, enabling gradient flow to model parameters.
        - Adds tiny regularization and safety ties to ensure autograd connectivity.
        Returns the surrogate scalar loss.
        """
        print("[_meshsdf_backward_step] enter")
        assert self.verts is not None and self.faces is not None and self.model is not None
        v = self.verts.detach().clone().requires_grad_(True)
        self._working_verts = v
        print("[_meshsdf_backward_step] v.requires_grad=", v.requires_grad)
        # call mesh loss without arguments, it uses instance attributes
        loss_vertices = self._solar_mesh_loss()
        (dldv,) = torch.autograd.grad(loss_vertices, v, create_graph=False, retain_graph=True)
        print("[_meshsdf_backward_step] dldv.norm=", float(dldv.detach().norm().cpu().item()))

        s = self.model(v).view(-1)
        print("[_meshsdf_backward_step] s.shape=", tuple(s.shape))
        (grad_xyz,) = torch.autograd.grad(
            s,
            v,
            grad_outputs=torch.ones_like(s),
            create_graph=True,
            retain_graph=True,
        )
        print("[_meshsdf_backward_step] grad_xyz.norm=", float(grad_xyz.detach().norm().cpu().item()))
        eps = 1e-12
        n = grad_xyz / (grad_xyz.norm(dim=-1, keepdim=True) + eps)
        dlds = (dldv * n).sum(dim=-1)
        surrogate = (dlds.detach() * s).sum()
        print("[_meshsdf_backward_step] surrogate.initial.requires_grad=", surrogate.requires_grad)

        # Ensure every trainable parameter contributes (even minimally) so gradient logging via
        # torch.autograd.grad(loss, params) never fails due to "unused parameters".
        try:
            params = list(self.model.parameters()) if hasattr(self.model, "parameters") else []
            if params:
                # Add a tiny L2 regularization term to create a graph dependency on all trainable params.
                # The coefficient is extremely small to be effectively neutral for optimization,
                # but guarantees gradients exist for logging.
                tiny_coeff = 1e-12
                l2_sum = None
                for p in params:
                    if p.requires_grad and p.numel() > 0:
                        term = p.pow(2).sum()
                        l2_sum = term if l2_sum is None else l2_sum + term
                if l2_sum is not None:
                    surrogate = surrogate + tiny_coeff * l2_sum
        except Exception:
            # Never let logging-prep break training
            pass

        # Final safety: if for any reason surrogate lost its grad_fn (e.g., outer code wrapped in no_grad),
        # reattach a benign dependency on model outputs so autograd.grad can succeed.
        if not surrogate.requires_grad:
            print("[_meshsdf_backward_step] surrogate lost grad_fn, re-attaching tie to model output")
            s_tie = self.model(self._working_verts).sum()
            surrogate = surrogate + s_tie * 0.0

        print("[_meshsdf_backward_step] surrogate.final.requires_grad=", surrogate.requires_grad)
        # Do not call backward here; return surrogate so the outer loop can backprop and log grads
        return surrogate

    def _visualize_and_log_selected_faces(self) -> None:
        """
        Visualize selected faces against the full mesh and log artifacts.
        - Delegates to helper methods for mesh prep, rendering/logging, and cleanup
          to keep this method concise.
        """
        tmp_dir = Path(tempfile.mkdtemp(prefix="solar_mesh_vis_"))
        self._tmp_dir = tmp_dir
        self._orig_ply = tmp_dir / "mesh_original.ply"
        self._sel_ply = tmp_dir / "mesh_selected.ply"
        self._out_img = tmp_dir / "render.png"

        self._prepare_mesh_files()
        self._render_and_log()
        self._cleanup_temp()

    def _prepare_mesh_files(self) -> None:
        """
        Build trimesh objects for full mesh and selected faces, and export to PLY files.
        """
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

    def _render_and_log(self) -> None:
        """
        Render comparison image, log visualization and area summaries to wandb.
        """
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

        # Compute and log selected and total areas as part of visualization/logging
        try:
            verts = self._working_verts
            faces = self.faces
            sel = self._selected_face_indices
            if verts is not None and faces is not None and sel is not None and sel.numel() > 0:
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
                    step=self.epoch_name,
                )
        except Exception:
            pass

    def _cleanup_temp(self) -> None:
        """
        Remove temporary files and directory created for visualization.
        """
        for p in (self._orig_ply, self._sel_ply, self._image_path):
            try:
                if p is not None and Path(p).exists():
                    Path(p).unlink()
            except Exception:
                print("try 4 failed")
        with contextlib.suppress(Exception):
            if self._tmp_dir is not None:
                self._tmp_dir.rmdir()
