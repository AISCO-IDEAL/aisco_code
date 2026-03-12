from pathlib import Path

import torch
import trimesh

import wandb
from subprojects.wp3_solar_estimator.mesh_extractors.non_diff_algos.marching_cubes import MeshExtractor
from subprojects.wp3_solar_estimator.validation.statistics import MeshStatistics
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.cruising_speed import cruising_speed
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.weights import weights_alm, weights_manual
from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.dynamic_loading import load_class_from_path
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.visualizer import CADVisualizer
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visual_wrapper import VisualisationManager


class TotalLoss:
    def __init__(self, dev: str = "cuda", renderer_manager: VisualisationManager | None = None) -> None:
        self._dev = dev
        self._renderer_manager = renderer_manager
        self._project_root = Path.cwd()

        # 1. Configs
        self.__general_config = load_config()
        self.losses_config = load_config(
            v=self.__general_config["general_info"]["version"],
            cfg_name=self.__general_config["active_configs"]["losses"],
        )
        _log_cfg = load_config(v="v1", cfg_name="logging.yaml")

        # 2. State & Flags
        self.__active_losses: dict = {}
        self._last_weighted_losses: dict[str, torch.Tensor] = {}
        self._last_unweighted_losses: dict[str, torch.Tensor] = {}

        gi = self.__general_config.get("wandb_general", {})
        self._use_wandb = gi.get("use_wandb", True)
        self._use_renderer = gi.get("use_renderer", True)
        self._use_neural_renderer = gi.get("use_neural_renderer", False)
        self._rendering_freq = gi.get("rendering_frequency")

        self._mem_diag = _log_cfg.get("memory_diagnostics", False)
        self._mesh_stats_enabled = _log_cfg.get("mesh_statistics", False)

        # 3. Initialization
        self._init_mesh_extractor()
        self._init_alm_params()
        self._init_active_losses()
        self._mesh_statistics = MeshStatistics(device=dev, use_wandb=self._use_wandb)

    def _init_mesh_extractor(self):
        r_cfg = load_config(v="v1", cfg_name="neural_renderer_v0.yaml")
        m_cfg = r_cfg.get("mesh_extraction", {})
        self._mesh_extraction_config = m_cfg
        self._mesh_extractor = MeshExtractor(
            algorithm=m_cfg.get("algorithm", "mc"),
            device=m_cfg.get("device", self._dev),
            verbose=m_cfg.get("verbose", False),
        )
        self._mesh_extractor.set_grid(
            bbox=m_cfg.get("bbox", [[-5000, 5000], [-5000, 5000], [-5000, 5000]]),
            resolution=m_cfg.get("resolution", 128),
        )

    def _init_alm_params(self):
        if self.losses_config["general_info"].get("ALM") == "True":
            alm = self.losses_config["general_info"]
            self.lamda = alm.get("initial_lambda", 0.01)
            self.lamdas = torch.full((3, 1), self.lamda, device=self._dev)
            self.mus = torch.ones((3, 1), device=self._dev)
            self.nus = torch.ones((3, 1), device=self._dev)
            self.alphas = torch.full((3, 1), alm.get("alpha", 0.99), device=self._dev)
            self.gamma = alm.get("gamma", 0.01)

    def _init_active_losses(self):
        for category, lossnames in self.losses_config.get("losses", {}).items():
            if category == "total":
                continue
            for name, (compute, _, path) in lossnames.items():
                if compute:
                    abs_path = self._project_root / Path(path)
                    family = Path(path).stem.title().replace("_", "")
                    loss_cls = load_class_from_path(str(abs_path), family)
                    self.__active_losses[name] = loss_cls(device=self._dev, loss_name=name, losses_config=self.losses_config)

    def get_total_loss(self, model: BaseModel, epoch_name: int = -1) -> torch.Tensor:
        device = next(iter(model.parameters())).device

        # 1. Compute & Weight
        loss_names_map = self._compute_all_losses(model, epoch_name)

        if self.losses_config["general_info"].get("ALM") == "True":
            total_loss = self._apply_alm_weighting(device)
        else:
            total_loss = self._apply_manual_weighting(loss_names_map, device)

        # 2. Finalize & Log
        loss_names_map["total_loss"] = total_loss.item()
        self._log_everything(loss_names_map, epoch_name, device)

        if self._use_renderer and self._renderer_manager:
            self._handle_visuals(model, total_loss, epoch_name)

        return total_loss

    def _compute_all_losses(self, model, epoch):
        names_map = {}
        for name, instance in self.__active_losses.items():
            tensor = instance.get_loss(model, epoch)
            self._last_unweighted_losses[name] = tensor
            names_map[name] = tensor.item()
        return names_map

    def _apply_alm_weighting(self, device, loss_names_map) -> torch.Tensor:
        # Mapping names to your logic (formerly index [0] and [4])
        drag: torch.Tensor = self._last_unweighted_losses.get("null_loss_aero_v1_")
        solar: torch.Tensor = self._last_unweighted_losses.get("solar_v4")

        if self.losses_config["general_info"].get("CSE") == "True":
            v_cruise = cruising_speed(
                solar_power=solar,
                drag_force_at_25ms=drag,
                mass=210,
                c_losses=20,
                rolling_friction_coefficient=0.006,
                max_iter=12,
                tol=1e-4,
            )
            loss_names_map["metrics/cruising_speed_ms"] = v_cruise.item()
            o = -v_cruise  # minimize the negative speed
        else:
            o = torch.as_tensor(-1.0 * (solar - drag), device=device)

        # Constraints (indices 1, 2, 3)
        c_names = ["bb_v3", "eikonal_v0", "smoothness_v0"]
        constraints = [self._last_unweighted_losses[n] for n in c_names]

        # Multiplier Update
        c_det = torch.stack([c.detach() for c in constraints])
        thresh = self.losses_config["general_info"].get("alm_thresholds")
        t_tensor = torch.tensor(thresh, device=device) if thresh else None

        self.lamdas, self.mus, self.nus = weights_alm(self.lamdas, self.mus, self.nus, self.alphas, self.gamma, c_det, t_tensor)

        self._last_weighted_losses = {"alm_objective": o}
        for i, name in enumerate(c_names):
            ci = constraints[i]

            weighted_constraint = self.lamdas[i].detach() * ci + 0.5 * self.mus[i].detach() * ci**2
            self._last_weighted_losses[f"alm_{name}"] = weighted_constraint

            loss_names_map[f"alm_weighted_constraint_{name}"] = weighted_constraint.item()
        return sum(self._last_weighted_losses.values())

    def _apply_manual_weighting(self, loss_names_map, device) -> torch.Tensor:
        weights = weights_manual(loss_names_map, self.losses_config, list(self.__active_losses.keys()), device)
        w_tensor = torch.as_tensor(weights, device=device, dtype=torch.float32)

        self._last_weighted_losses = {}
        for i, name in enumerate(self.__active_losses.keys()):
            weighted_val = self._last_unweighted_losses[name] * w_tensor[i]
            self._last_weighted_losses[name] = weighted_val

            # ADDED: Log the contribution (e.g., weighted/eikonal_v0)
            loss_names_map[f"weighted/{name}"] = weighted_val.item()

        return sum(self._last_weighted_losses.values())

    def _handle_visuals(self, model, total_loss, epoch):
        verts, faces = self._mesh_extractor.extract_mesh(
            model=model,
            isovalue=self._mesh_extraction_config.get("isovalue", 0.0),
            batch_size=self._mesh_extraction_config.get("batch_size", 8192),
        )

        if self._mesh_stats_enabled:
            self._mesh_statistics.log_basic_mesh_stats(verts, faces, epoch)
            self._mesh_statistics.log_solar_area_stats(verts, faces, epoch)

        if self._rendering_freq and epoch % self._rendering_freq == 0:
            if self._use_neural_renderer:
                try:
                    self._renderer_manager.update(
                        {"model": model, "iteration": epoch, "loss": total_loss.item(), "vertices": verts, "faces": faces}
                    )
                except Exception as e:
                    print(f"Renderer failed: {e}")
            else:
                self._run_cad_vis(verts, faces, epoch)

    def _run_cad_vis(self, verts, faces, epoch):
        v_np = verts.detach().cpu().numpy() if torch.is_tensor(verts) else verts
        f_np = faces.detach().cpu().numpy() if torch.is_tensor(faces) else faces

        mesh = trimesh.Trimesh(vertices=v_np, faces=f_np)
        stl = Path(f"subprojects/wp4_aeroshell_optimizer/v1/logs/mesh_epoch_{epoch}.stl")
        stl.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(stl))

        img = stl.with_suffix(".png")
        vis = CADVisualizer(stl_path=str(stl), out_image_path=str(img))
        vis.render_until_blue_detected()

        if self._use_wandb:
            wandb.log({"mesh_3d": wandb.Object3D(str(stl)), "render": wandb.Image(str(img))}, step=epoch)
        if stl.exists():
            stl.unlink()
        if img.exists():
            img.unlink()

    def _log_everything(self, loss_names, epoch, device):
        if not self._use_wandb:
            return
        wandb.log(loss_names, step=epoch)

        if self.losses_config["general_info"].get("ALM") == "True":
            wandb.log(
                {
                    "alm/lambda_bb": self.lamdas[0].item(),
                    "alm/mu_bb": self.mus[0].item(),
                    "alm/lambda_eikonal": self.lamdas[1].item(),
                    "alm/mu_eikonal": self.mus[1].item(),
                    "alm/lambda_smoothness": self.lamdas[2].item(),
                    "alm/mu_smoothness": self.mus[2].item(),
                },
                step=epoch,
            )

        for instance in self.__active_losses.values():
            instance.log_gradients(epoch)

        if self._mem_diag:
            self._log_memory_diagnostics(epoch, device)

    def _log_memory_diagnostics(self, epoch_name: int, device: torch.device) -> None:
        if device.type != "cuda":
            return
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        if self._use_wandb:
            wandb.log({"memory/allocated_mb": alloc, "memory/peak_mb": peak}, step=epoch_name)
