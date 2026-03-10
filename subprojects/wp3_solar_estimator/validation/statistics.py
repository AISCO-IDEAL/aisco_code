import torch

from subprojects.wp3_solar_estimator.geo_mesh_utils.mesh_properties import (
    compute_face_normals,
    compute_mesh_surface_area,
    compute_triangle_areas,
)
from subprojects.wp4_aeroshell_optimizer.v1.logging_lib.basic_plots import (
    basic_multi_log,
    basic_single_log,  # noqa: F401
    log_matplotlib_fig,  # noqa: F401
    log_renderer_image,  # noqa: F401
    multi_log_in_one,
)
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config


class MeshStatistics:
    """
    This class should log relevant mesh data to the wandb project to help with solar/misc. debugging
    """

    def __init__(self, device: str = "cpu", use_wandb: bool = True):
        self.device = device
        self.use_wandb = use_wandb  # Set to True if you want to log to wandb, make sure to initialize wandb in your main file and set up a project

    def log_basic_mesh_stats(self, vertices: torch.Tensor, faces: torch.Tensor, epoch: int) -> None:
        """
        Logs basic mesh statistics such as number of vertices, number of faces
        """
        if not self.use_wandb:
            return
        num_vertices = vertices.shape[0]
        num_faces = faces.shape[0]
        basic_multi_log({"num_vertices": num_vertices, "num_faces": num_faces}, step=epoch)

    def log_solar_area_stats(self, vertices: torch.Tensor, faces: torch.Tensor, epoch: int) -> None:
        """
        Logs solar area related statistics such as total surface area and solar performance area
        """
        if not self.use_wandb:
            return
        solar_config = load_config(v="v1", cfg_name="solar_loss_config.yaml")["SolarLossConfigV3"]
        solar_angle = torch.tensor(solar_config["sun_dir"], device=vertices.device)
        self.solar_angle = solar_angle / torch.norm(solar_angle)  # Normalize to unit vector so dot products give cos(θ)
        self.area_budget = solar_config["area_budget_mm2"]
        # Whole selection logic here:

        face_normals = compute_face_normals(vertices, faces)
        face_areas = compute_triangle_areas(vertices, faces)

        # On the side: log how many face normals point away from the origin (0,0,0)
        origin_dot_prods = (face_normals * vertices[faces].mean(dim=1)).sum(dim=1)
        away_from_origin = (origin_dot_prods > 0).sum().item()
        total_faces = faces.shape[0]
        multi_log_in_one(
            "Inside out stats",
            {
                "Faces Pointing Away from Origin": away_from_origin,
                "Percentage pointing inside": (1 - away_from_origin / total_faces) * 100
                if total_faces > 0
                else 0,  # Division by 0 handling, weird ruff-autosave formatting here?
                "Total Faces": total_faces,
            },
            step=epoch,
        )

        # Calculate dot products (cosine of angle with sun vector)
        solar_dot_prods = (face_normals * self.solar_angle).sum(dim=1)
        # Calculate score (energy) and cost (area)
        # We maximize total energy (score) subject to area (cost) <= budget
        score = solar_dot_prods * face_areas
        cost = face_areas
        budget = self.area_budget
        # Ratio for greedy selection (value density) = energy / area = dot_product
        ratio = solar_dot_prods
        # Sort by ratio descending
        order = torch.argsort(ratio, descending=True)
        sorted_cost = cost[order]
        # sorted_score = score[order] # Not strictly needed for logic but good for debugging if needed  # noqa: ERA001
        torch.use_deterministic_algorithms(
            False
        )  # Cumsum can be nondeterministic on GPU, but we need it for the selection logic, so we disable deterministic algorithms here
        cum_cost = torch.cumsum(sorted_cost, dim=0)
        torch.use_deterministic_algorithms(True)
        # fully taken items
        full = cum_cost <= budget
        mask = torch.zeros_like(score, dtype=torch.float32)
        # assign full items
        if full.any():
            mask[order[full]] = 1.0
        # Compare how much are was selected by taking the sum of face_areas where mask > 0.5 (i.e. selected) and log this along with the total surface area for comparison
        selected_solar_area = cost[mask > 0.5].sum()

        total_surface_area = compute_mesh_surface_area(vertices, faces)

        # Compute and/or log the projected solar area of the selected faces, the projected solar area of the whole mesh (filter out negative dot prods),
        # the total area as a comparison and make a line with the budget area for reference
        solar_performance_selected = (solar_dot_prods[mask > 0.5] * cost[mask > 0.5]).sum()

        # Total projected area of the mesh (only considering faces facing the sun)
        pos_mask = solar_dot_prods > 0
        total_projected_area = (solar_dot_prods[pos_mask] * cost[pos_mask]).sum()

        multi_log_in_one(
            "Solar Area Stats",
            {
                "Total Surface Area": total_surface_area / 1e6,
                "Selected Surface Area": selected_solar_area / 1e6,
                "Selected Projected Area": solar_performance_selected / 1e6,
                "Total Projected Area": total_projected_area / 1e6,
                "Area Budget": self.area_budget / 1e6,
            },
            step=epoch,
        )
