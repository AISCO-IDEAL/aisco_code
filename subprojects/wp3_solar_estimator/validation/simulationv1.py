"""
For validation purposes, runs a basic race simulation to predict the total solar output during the race
Assumes a constant speed given in the australia.yaml config file, and a constant solar output per unit area given in the same config file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from subprojects.wp3_solar_estimator.australia.route_v2 import RouteV2
from subprojects.wp3_solar_estimator.utils.solar_utils import averaging_solar_quality
from subprojects.wp4_aeroshell_optimizer.v1.logging_lib.image_uploads import upload_neural_renderer_image


class SimulationV1:
    def __init__(
        self,
        cfg_path: str = "solar_configs/australia.yaml",
        area_budget_mm2: float = 3500000,
        mesh_render_cameras: list[dict] | None = None,
    ) -> None:
        self.route = RouteV2(cfg_path=cfg_path)
        self.area_budget_mm2 = area_budget_mm2
        # Each dict: {"position": [x,y,z] | None, "direction": [x,y,z] | None, "suffix": str}
        # None position/direction → auto-computed from mesh bounds at render time
        self.mesh_render_cameras: list[dict] = mesh_render_cameras if mesh_render_cameras is not None else [
            {"position": None, "direction": None, "suffix": "side"},
        ]
        self._visualiser = None  # lazy-initialised on first mesh render

    def _get_visualiser(self):
        """Lazy-initialise the Visualiser on first use so it doesn't load when not needed."""
        if self._visualiser is None:
            from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser import Visualiser  # noqa: PLC0415

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._visualiser = Visualiser(image_size=512, background_color=[0.1, 0.1, 0.1], device=device)
        return self._visualiser

    def run(
        self,
        vertices: torch.Tensor,
        mesh_faces: torch.Tensor,
        output_dir: Path | str | None = None,
        epoch_name: str | int | None = None,
    ):
        # disable gradient tracking since this is just a validation simulation
        verts, faces = vertices.detach().clone(), mesh_faces.detach().clone()
        verts.requires_grad_(False)
        faces.requires_grad_(False)

        solar_cells_mask, normals, areas_mm2 = self.allocate_solar_cells(verts, faces, self.route)
        # cull faces, normals and areas that dont get solar cells allocated

        allocated_normals = normals[solar_cells_mask]
        allocated_areas_mm2 = areas_mm2[solar_cells_mask]

        # calculate the solar output for each point
        solar_output_area = np.zeros(
            self.route.num_points
        )  # array to store total solar projected area at each point along the route
        positions = np.zeros((self.route.num_points, 3))  # array to store positions along the route for debugging/analysis
        for point in range(self.route.num_points):
            sun_vec = self.route.get_route_solar_vecs(point).to(verts.device)
            position = self.route.get_route_points(point)

            # dot product of all allocated normals with single solar vector to get solar quality, then multiply by area to get total solar projected area at this point
            solar_qualities = allocated_normals * sun_vec
            solar_projected_area = (solar_qualities * allocated_areas_mm2.unsqueeze(-1)).sum()

            positions[point] = position.tolist()
            solar_output_area[point] = solar_projected_area.item()

        self.plot_results(positions, solar_output_area, output_dir, epoch_name, verts=verts, faces=faces, solar_cells_mask=solar_cells_mask)

    def allocate_solar_cells(
        self, verts: torch.Tensor, faces: torch.Tensor, route: RouteV2
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each face, compute the average solar quality across its vertices

        Returns:
            solar_cells_allocated: (M,) boolean tensor mask indicating which faces get solar cells allocated
            normals: (M, 3) tensor of unit face normals, for later use in computing solar quality
            areas_mm2: (M,) tensor of face areas in mm^2, for later use in computing total solar output
        """
        solar_vectors = route.get_route_solar_vecs().to(verts.device)

        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        edge_a, edge_b = v1 - v0, v2 - v0
        face_cross = torch.cross(edge_a, edge_b, dim=-1)
        face_cross_norm = torch.linalg.norm(face_cross, dim=-1)
        face_area_mm2 = 0.5 * face_cross_norm

        # Compute illumination quality (cosine of angle with sun direction, clamped to [0, 1])
        face_normals_unit = face_cross / (face_cross_norm.clamp_min(1e-6)).unsqueeze(-1)  # (M, 3) unit normals of faces
        sun_directions_unit = solar_vectors / (torch.linalg.norm(solar_vectors, dim=-1, keepdim=True).clamp_min(1e-6))

        agg_face_solar_quality = averaging_solar_quality(face_normals_unit, sun_directions_unit)

        # Sort solar cells based on quality and select top N based on area budget
        quality_order_desc = torch.argsort(agg_face_solar_quality, descending=True)
        sorted_areas = face_area_mm2[quality_order_desc]

        # Deterministic-safe cumulative area
        cumulative_area = torch.cumsum(sorted_areas.cpu(), dim=0).to(verts.device)
        # Find which faces we can allocate (better quality first) before exceeding the area budget
        allocated_mask_sorted = cumulative_area <= self.area_budget_mm2
        # Map back to original face order
        solar_cells_allocated = torch.zeros_like(allocated_mask_sorted, dtype=torch.bool)
        solar_cells_allocated[quality_order_desc] = allocated_mask_sorted

        return solar_cells_allocated, face_normals_unit, face_area_mm2

    def render_mesh_selection(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        solar_cells_mask: torch.Tensor,
        out_path: Path,
        epoch_name: str | int | None = None,
    ) -> None:
        """Render the mesh with selected solar faces highlighted in gold, rest in dark gray.

        Each entry in self.mesh_render_cameras produces a separate image file.
        position/direction of None in a camera config falls back to an auto-computed value
        derived from the mesh bounding box, so adding new camera angles only requires
        appending a dict to mesh_render_cameras – no other changes needed.
        """
        try:
            vis = self._get_visualiser()

            # Build per-face color tensor: gold for selected, dark gray for the rest
            gold = torch.tensor([1.0, 0.75, 0.0], device=verts.device)
            gray = torch.tensor([0.3, 0.3, 0.3], device=verts.device)
            face_colors = torch.where(solar_cells_mask.unsqueeze(-1), gold, gray)  # (F, 3)

            textures = vis.face_colors_to_textures(face_colors)

            # Auto-compute a sensible default camera from bounding box
            vmin = verts.min(dim=0).values.cpu()
            vmax = verts.max(dim=0).values.cpu()
            center = ((vmin + vmax) / 2.0).tolist()
            extent = (vmax - vmin).max().item()
            auto_pos = [center[0] + extent * 1.5, center[1], center[2] + extent * 0.3]
            auto_dir = [-1.0, 0.0, -0.3]  # points roughly toward mesh center

            out_path.mkdir(parents=True, exist_ok=True)

            for cam in self.mesh_render_cameras:
                pos = cam.get("position") or auto_pos
                direction = cam.get("direction") or auto_dir
                suffix = cam.get("suffix", "view")

                image = vis.render(verts, faces, textures, mode="rgb", camera_position=pos, camera_direction=direction)

                if epoch_name is not None:
                    fname = f"mesh_selection_{suffix}_{epoch_name}.png"
                    key_name = f"mesh_selection_{suffix}_{epoch_name}"
                else:
                    fname = f"mesh_selection_{suffix}.png"
                    key_name = f"mesh_selection_{suffix}"

                file_path = str(out_path / fname)
                vis.save_image(image[0], file_path)
                upload_neural_renderer_image(file_path, key_name)

        except Exception as e:
            print(f"Warning: mesh selection render failed (epoch={epoch_name}): {e}")

    def plot_results(
        self,
        positions: np.ndarray,
        solar_output_area: np.ndarray,
        output_dir: Path | str | None = None,
        epoch_name: str | int | None = None,
        verts: torch.Tensor | None = None,
        faces: torch.Tensor | None = None,
        solar_cells_mask: torch.Tensor | None = None,
    ):
        # goal here is to make a gif of the australia route map with points colored by solar output, and a plot of solar output and lat/lon over the route points.
        # note: normalises mm2 to m2 with 1e-6 calculation
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(
            positions[:, 1],
            positions[:, 0],
            c=solar_output_area * 1e-6,
            cmap="inferno",
            vmin=0,
            vmax=self.area_budget_mm2 * 1e-6,
        )
        plt.colorbar(label="Solar Output (m^2)")
        title_str = "Simulated Solar Output and Route Positions"
        if epoch_name is not None:
            title_str += f" (Epoch: {epoch_name})"
        plt.title(title_str)

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        # ensure equal aspect ratio for lat/lon
        plt.gca().set_aspect("equal", adjustable="box")

        file_path = "simulation_results.png"
        key_name = "simulation_results"
        if output_dir is not None:
            out_path = Path(output_dir) / "sim_renders"
            out_path.mkdir(parents=True, exist_ok=True)
            if epoch_name is not None:
                file_path = str(out_path / f"simulation_results_{epoch_name}.png")
                key_name = f"simulation_results_{epoch_name}"
            else:
                file_path = str(out_path / "simulation_results.png")

        plt.savefig(file_path, bbox_inches="tight")
        plt.close()

        # log to wandb as image
        upload_neural_renderer_image(file_path, key_name)

        # render 3-D mesh with selected faces highlighted
        if verts is not None and faces is not None and solar_cells_mask is not None:
            render_out = Path(output_dir) / "sim_renders" if output_dir is not None else Path("sim_renders")
            self.render_mesh_selection(verts, faces, solar_cells_mask, render_out, epoch_name)
