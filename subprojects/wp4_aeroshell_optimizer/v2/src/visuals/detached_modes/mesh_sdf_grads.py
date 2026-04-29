"""Render the surface mesh coloured by per-vertex `dL/dv . n` for one loss.

The visualised quantity is the dot product between each vertex's outward normal
and the gradient of a single loss (or the weighted total) with respect to that
vertex. Positive values (loss wants to push the surface outward) are rendered
green, negative values (push inward) are rendered red.

Per-loss data flow:
    ParentLoss.last_vertice_grads          (set by losses that use MeshSDFExtractor)
        -> TotalLossV2.collect_vertex_grads_for_visualisation()
        -> diversity_manager._render injects into optimization_state
        -> MeshSDFGrads.update reads optimization_state["vertice_grads_per_loss"][loss_key]
"""

from __future__ import annotations

from typing import Any

import torch

import wandb
from subprojects.wp4_aeroshell_optimizer.v2.logging_lib.image_uploads import upload_neural_renderer_gif
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.rendering_mode_base_class import RenderingMode
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.visualiser import Visualiser

# Sentinel used to select the weighted-sum gradient instead of a single loss.
TOTAL_KEY = "__total__"


class MeshSDFGrads(RenderingMode):
    """Per-vertex gradient visualiser for one loss key.

    Required config keys (in addition to RenderingMode's camera_position / look_at / render_mode):
        loss_key:  loss name to visualise, or "__total__" for the weighted sum.
        intensity: colour-saturation scale (default 1.0).
    """

    def __init__(self, visualiser: Visualiser, config: dict[str, Any]) -> None:
        super().__init__(visualiser, config)
        self.loss_key: str = config.get("loss_key", TOTAL_KEY)
        self.intensity: float = float(config.get("intensity", 1.0))

    def _select_vertice_grads(self, optimization_state: dict[str, Any]) -> torch.Tensor | None:
        """Return the (V, 3) grad tensor for `self.loss_key`, or None if missing."""
        if self.loss_key == TOTAL_KEY:
            return optimization_state.get("vertice_grads_total")
        per_loss = optimization_state.get("vertice_grads_per_loss") or {}
        return per_loss.get(self.loss_key)

    def update(self, optimization_state: dict[str, Any]) -> None:
        if not torch.cuda.is_available():
            return

        vertice_grads = self._select_vertice_grads(optimization_state)
        if vertice_grads is None:
            if not getattr(self, "_diag_logged_miss", False):
                self._diag_logged_miss = True
                has_total = "vertice_grads_total" in optimization_state
                per_loss_keys = list((optimization_state.get("vertice_grads_per_loss") or {}).keys())
                print(
                    f"[VisuDiag MeshSDFGrads/{self.loss_key}] no grads available. "
                    f"has_total={has_total} per_loss_keys={per_loss_keys}"
                )
            return  # No grad available this step; silently skip a frame.

        if not getattr(self, "_diag_logged_hit", False):
            self._diag_logged_hit = True
            print(f"[VisuDiag MeshSDFGrads/{self.loss_key}] first render OK, grads shape={tuple(vertice_grads.shape)}")

        vertices = optimization_state["vertices"].detach()
        faces = optimization_state["faces"].detach()
        if faces.numel() == 0:
            return

        # Vertex normals = average of incident face normals.
        e1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        e2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
        face_normals = torch.cross(e1, e2, dim=-1)
        face_normals = face_normals / (face_normals.norm(dim=-1, keepdim=True) + 1e-8)
        vertex_normals = torch.zeros_like(vertices)
        vertex_normals.index_add_(0, faces[:, 0], face_normals)
        vertex_normals.index_add_(0, faces[:, 1], face_normals)
        vertex_normals.index_add_(0, faces[:, 2], face_normals)
        vertex_normals = vertex_normals / (vertex_normals.norm(dim=-1, keepdim=True) + 1e-8)

        # Per-vertex scalar: positive => grad pushes surface outward.
        dot = torch.einsum("ij,ij->i", vertex_normals, vertice_grads.detach())
        norm = torch.clamp(dot / (dot.abs().max() + 1e-8) * self.intensity, -1.0, 1.0)

        base = torch.tensor([0.5, 0.5, 0.5], device=faces.device)
        out_color = torch.tensor([0.25, 1.0, 0.25], device=faces.device)  # green: outward push
        in_color = torch.tensor([1.0, 0.0, 0.0], device=faces.device)     # red:   inward push
        colors = base.unsqueeze(0) + norm.unsqueeze(-1) * (out_color - in_color).unsqueeze(0)

        textures = self.visualiser.vertex_colors_to_textures(vertex_colors=colors, faces=faces)
        image = self._render(
            vertices=vertices,
            faces=faces,
            textures=textures,
            mode="rgb",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )
        if hasattr(image, "__len__") and len(image) > 0:
            self.frame_storage.append(image[0])
        else:
            self.frame_storage.append(image)

    def finalize(self, output_dir, file_name: str | None = None, delete: bool = True) -> None:
        """Save GIF + first/last PNG, upload to wandb, optionally clear frames."""
        if not self.frame_storage:
            return
        file_name = file_name or f"MeshSDF_grads_{self.loss_key}.gif"
        gif_path = output_dir / file_name
        self._frames_to_gif(gif_path, fps=self.fps)
        try:
            upload_neural_renderer_gif(gif_path, key=f"animations/MeshSDFGrads/{self.loss_key}", step=0)
        except Exception as e:
            print(f"  MeshSDFGrads[{self.loss_key}]: wandb gif upload failed: {e}")
        self._save_image(self.frame_storage[0], output_dir / f"{file_name}_first_frame.png")
        self._save_image(self.frame_storage[-1], output_dir / f"{file_name}_final_frame.png")
        if delete:
            self.clear_frames()

    def log_single_image_to_wandb(self, optimization_state: dict[str, Any], epoch_name: int) -> None:
        """Render one frame and log it as a wandb image for this loss key."""
        try:
            self.update(optimization_state)
            if not self.frame_storage:
                return
            wandb.log(
                {"epoch": epoch_name, f"Aero-Gradients/{self.loss_key}": wandb.Image(self.frame_storage[-1])},
                step=int(epoch_name),
            )
        except Exception as e:
            print(f"MeshSDFGrads[{self.loss_key}]: per-step wandb log failed at epoch {epoch_name}: {e}")
