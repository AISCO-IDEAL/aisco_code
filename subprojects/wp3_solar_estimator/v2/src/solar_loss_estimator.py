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


class SolarLossEstimator:
    class _SolarMeshLossFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, v, f, sun_dir, area_budget_tensor, eps: float):
            v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
            a, b = v1 - v0, v2 - v0
            c = torch.cross(a, b, dim=-1)  # (F,3)
            area = 0.5 * torch.linalg.norm(c, dim=-1)  # (F,)
            dot = (c * sun_dir).sum(-1)  # (F,)
            score = dot

            area_budget = float(area_budget_tensor.item())
            order = torch.argsort(score / (area.clamp_min(eps)), descending=True)
            indices_list, acc = [], 0.0
            for i in order.tolist():
                ai = float(area[i].detach())
                if ai <= 0.0:
                    continue
                if acc + ai <= area_budget + 1e-12:
                    indices_list.append(i)
                    acc += ai

            sel_tensor = torch.tensor(indices_list, device=v.device, dtype=torch.long)
            area_sel = area[sel_tensor] if sel_tensor.numel() else torch.empty(0, device=v.device, dtype=v.dtype)
            ctx.save_for_backward(
                v,
                f,
                sel_tensor,
                c[sel_tensor] if sel_tensor.numel() else c.new_empty((0, 3)),
                a[sel_tensor] if sel_tensor.numel() else a.new_empty((0, 3)),
                b[sel_tensor] if sel_tensor.numel() else b.new_empty((0, 3)),
            )
            ctx.eps = float(eps)
            return torch.zeros((), device=v.device, dtype=v.dtype) if sel_tensor.numel() == 0 else area_sel.sum()

        @staticmethod
        def backward(ctx, g):
            v, f, sel_tensor, c_sel, a_sel, b_sel = ctx.saved_tensors
            if not ctx.needs_input_grad[0]:
                return None, None, None, None, None, None

            gv = torch.zeros_like(v)
            if sel_tensor.numel():
                fs = f[sel_tensor]
                i0, i1, i2 = fs[:, 0], fs[:, 1], fs[:, 2]
                nc = torch.linalg.norm(c_sel, dim=-1, keepdim=True)
                u = c_sel / (nc + ctx.eps)
                da = 0.5 * torch.cross(u, b_sel, dim=-1)
                db = 0.5 * torch.cross(a_sel, u, dim=-1)
                g1 = da
                g2 = db
                g0 = -(g1 + g2)
                gv.index_add_(0, i1, g1)
                gv.index_add_(0, i2, g2)
                gv.index_add_(0, i0, g0)

            gv_out = gv * g
            return gv_out, None, None, None, None, None

    def __init__(self, device: str = "cpu", losses_config: dict | None = None):
        self.device = device
        self.losses_config = losses_config or {}
        solar_cfg = self.losses_config.get("solar_config", {})
        self.wandb_logging = solar_cfg.get("wandb_logging", False)
        self.rendering = solar_cfg.get("rendering", False)
        self.rendering_frequency = solar_cfg.get("rendering_frequency", 1)
        self.area_budget = int(solar_cfg.get("area_budget", 0))

        self.clamp_backfaces = True
        self.eps = 1e-12

    def solar_loss(self, model, epoch_name) -> torch.Tensor:
        self.epoch_name = epoch_name
        self.model = model
        self.verts, self.faces = get_diff_mesh(model)
        model.zero_grad(set_to_none=True)
        loss = self._meshsdf_backward_step()
        if not loss.requires_grad and hasattr(model, "parameters"):
            with contextlib.suppress(Exception):
                loss = loss + sum((p.sum() * 0.0) for p in model.parameters())
        return loss

    def _solar_mesh_loss(self) -> torch.Tensor:
        working = getattr(self, "_working_verts", None)
        verts = working if working is not None else self.verts
        faces = self.faces
        area_budget = float(self.area_budget or 0.0)

        s2 = math.sqrt(2.0)
        sun_dir = torch.tensor([-s2, 0.0, s2], device=verts.device, dtype=verts.dtype)
        self.sun_dir = sun_dir

        area_budget_tensor = torch.tensor(area_budget, device=verts.device, dtype=verts.dtype)

        loss = self._SolarMeshLossFn.apply(verts, faces, self.sun_dir, area_budget_tensor, float(self.eps))

        with torch.no_grad():
            sel_indices = self._compute_selection_indices(verts, faces, sun_dir, area_budget, self.clamp_backfaces, self.eps)
            if sel_indices is not None:
                self._selected_face_indices = sel_indices
                if self.rendering and int(self.epoch_name) % self.rendering_frequency == 0:
                    self._visualize_and_log_selected_faces()
                if self.wandb_logging:
                    self._visualize_and_log_selected_faces()

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

    def _meshsdf_backward_step(self) -> torch.Tensor:
        v = self.verts.detach().clone().requires_grad_(True)
        self._working_verts = v

        loss_vertices = self._solar_mesh_loss()
        (dldv,) = torch.autograd.grad(loss_vertices, v, create_graph=False, retain_graph=True)
        dldv = torch.nan_to_num(dldv, nan=0.0, posinf=0.0, neginf=0.0)

        s = self.model(v).view(-1)
        (grad_xyz,) = torch.autograd.grad(s, v, grad_outputs=torch.ones_like(s), create_graph=True, retain_graph=True)
        grad_xyz = torch.nan_to_num(grad_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        dlds = torch.tanh(((dldv * grad_xyz).sum(dim=-1)) / 1e-6)
        s_scaled = s
        surrogate = (dlds.detach() * s_scaled).sum()
        if not surrogate.requires_grad:
            surrogate = surrogate + self.model(self._working_verts).sum() * 0.0

        area_m2 = loss_vertices * 1e-6
        bounded_area = torch.clamp(area_m2, 0.0, 6.0)
        return surrogate + (bounded_area - surrogate.detach())

    def _visualize_and_log_selected_faces(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="solar_mesh_vis_"))
        self._tmp_dir = tmp_dir
        self._orig_ply = tmp_dir / "mesh_original.ply"
        self._sel_ply = tmp_dir / "mesh_selected.ply"
        self._out_img = tmp_dir / "render.png"

        self._prepare_mesh_files()
        self._render_and_log()
        self._cleanup_temp()

    def _prepare_mesh_files(self) -> None:
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
        assert self._orig_ply is not None and self._sel_ply is not None and self._out_img is not None
        visualizer = CADVisualizer(
            ply_path_a=self._orig_ply.as_posix(),
            ply_path_b=self._sel_ply.as_posix(),
            out_image_path=self._out_img.as_posix(),
        )
        image_path, ok, attempts = visualizer.render_until_blue_detected()
        # Be robust in typing conversions without strict annotations
        self._image_path = Path(image_path) if isinstance(image_path, (str, Path)) else None
        self._render_ok = None if ok is None else bool(ok)
        self._render_attempts = int(attempts)

        if self._image_path is not None:
            wandb.log({"solar_mesh_visualization": wandb.Image(str(self._image_path))}, step=self.epoch_name)

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
        for p in (self._orig_ply, self._sel_ply, self._image_path):
            if p is not None and Path(p).exists():
                Path(p).unlink()
        with contextlib.suppress(Exception):
            if self._tmp_dir is not None:
                self._tmp_dir.rmdir()
