"""Modular total-loss pipeline for the aeroshell optimiser.

1. ``_compute_raw_losses``  - evaluate every active loss instance
2. ``_apply_alm_weighting`` - single objective by calling CSE + constraint weighting by ALM (adaptive or aggressive)
3. ``_handle_telemetry``    - WandB logging, mesh rendering, memory diagnostics

**Category convention** (instead of fragile index-based lookup):

* ``aero``, ``solar``       → *objective* losses  (fed to cruising-speed / ALM objective)
* ``bb``, ``eikonal``, ``smoothness`` → *constraint* losses (ALM penalty terms)
* everything else            → *auxiliary* losses  (logged only, no backward)
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import trimesh

import wandb
from subprojects.wp3_solar_estimator.mesh_extractors.non_diff_algos.marching_cubes import (
    MeshExtractor,
)
from subprojects.wp3_solar_estimator.validation.statistics import MeshStatistics
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.cruising_speed import cruising_speed
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.weights import weights_alm
from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.dynamic_loading import load_class_from_path
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.visualizer import CADVisualizer
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visual_wrapper import VisualisationManager

# ---------------------------------------------------------------------------
# Category → role mapping (used to classify losses from the YAML config)
# ---------------------------------------------------------------------------
_ALM_OBJECTIVE_CATEGORIES: frozenset[str] = frozenset({"aero", "solar"})
_ALM_CONSTRAINT_CATEGORIES: tuple[str, ...] = ("bb", "eikonal", "smoothness")  # order matters ALM


class TotalLoss:
    """Aggregates, weights (ALM), calls CSE, and logs all loss components.

    complies with main file, which:
    * Call ``get_total_loss(model, epoch_name)`` each epoch.
    * Iterate ``_last_weighted_losses`` for per-component ``.backward(retain_graph=False)``.
    * Access ``_gradient_memory_diagnostics`` for optional per-backward memory logging.
    """

    # ------------------------------------------------------------------
    # initilization
    # ------------------------------------------------------------------
    def __init__(
        self,
        dev: str | torch.device = "cuda",
        renderer_manager: VisualisationManager | None = None,
    ) -> None:
        self._device: torch.device = torch.device(dev) if isinstance(dev, str) else dev
        self._renderer_manager: VisualisationManager | None = renderer_manager

        # Populated each call to get_total_loss(); maps component name → weighted
        # live tensor.  The training loop calls .backward() per entry.
        self._last_weighted_losses: dict[str, torch.Tensor] = {}
        # Stores the last computed cruising speed (if CSE active) for telemetry.
        self._last_cruising_speed: torch.Tensor | None = None

        self._load_configs()
        self._init_alm_state()
        self._init_mesh_infrastructure()
        self._init_active_losses()

    def _load_configs(self) -> None:
        self._general_config: dict[str, Any] = load_config()
        self.losses_config: dict[str, Any] = load_config(
            v=self._general_config["general_info"]["version"],
            cfg_name=self._general_config["active_configs"]["losses"],
        )
        # NOTE: sweep overrides are applied automatically by the config
        # registry (see config_utils.py).  No manual wandb.config patching needed.

        wg: dict[str, Any] = self._general_config.get("wandb_general", {})
        self._use_wandb: bool = wg.get("use_wandb", True)
        self._use_renderer: bool = wg.get("use_renderer", True)
        self._use_neural_renderer: bool = wg.get("use_neural_renderer", False)
        self._rendering_freq: int | None = wg.get("rendering_frequency")

        _log_cfg: dict[str, Any] = load_config(v="v1", cfg_name="logging.yaml")
        self._memory_diagnostics: bool = _log_cfg.get("memory_diagnostics", False)
        self._mesh_statistics_enabled: bool = _log_cfg.get("mesh_statistics", False)
        self._gradient_memory_diagnostics: bool = _log_cfg.get("gradient_memory_diagnostics", False)

    def _init_alm_state(self) -> None:
        """Initialise λ, μ, ν tensors and scalar hyper-parameters for ALM."""  # noqa: RUF002
        gi: dict[str, Any] = self.losses_config["general_info"]
        if gi.get("ALM") != "True":
            raise ValueError(
                "TotalLoss requires ALM='True' in losses config. "
                "Manual weighting has been removed (see refactoring decision log)."
            )

        dev = self._device
        self.lamda: float = gi.get("initial_lambda", 0.01)
        self.mu: float = gi.get("initial_mu", 1.0)
        self.nu: float = gi.get("initial_nu", 1.0)
        self.alpha: float = gi.get("alpha", 0.99)
        self.gamma: float = gi.get("gamma", 0.01)
        self.gamma_aggressive: float = gi.get("gamma_aggressive", 1.1)

        # Shape [3, 1] to match weights_alm's internal broadcasting logic.
        self.lamdas: torch.Tensor = torch.full((3, 1), self.lamda, device=dev)
        self.mus: torch.Tensor = torch.full((3, 1), self.mu, device=dev)
        self.nus: torch.Tensor = torch.full((3, 1), self.nu, device=dev)
        self.alphas: torch.Tensor = torch.full((3, 1), self.alpha, device=dev)

    def _init_mesh_infrastructure(self) -> None:
        dev_str = str(self._device)
        renderer_config: dict[str, Any] = load_config(v="v1", cfg_name="neural_renderer_v0.yaml")
        self._mesh_extraction_config: dict[str, Any] = renderer_config.get("mesh_extraction", {})

        self._mesh_extractor: MeshExtractor = MeshExtractor(
            algorithm=self._mesh_extraction_config.get("algorithm", "mc"),
            device=self._mesh_extraction_config.get("device", dev_str),
            verbose=self._mesh_extraction_config.get("verbose", False),
        )
        self._mesh_extractor.set_grid(
            bbox=self._mesh_extraction_config.get("bbox", [[-5000, 5000], [-5000, 5000], [-5000, 5000]]),
            resolution=self._mesh_extraction_config.get("resolution", 128),
        )
        self._mesh_statistics: MeshStatistics = MeshStatistics(device=dev_str, use_wandb=self._use_wandb)

    def _init_active_losses(self) -> None:
        """Load every loss with ``compute=True`` and classify into objective or constraint."""
        project_root: Path = Path.cwd()
        losses_block: dict[str, Any] = self.losses_config.get("losses", {})

        # Flat ordered dict of *all* active losses (for _compute_raw_losses iteration)
        self._active_losses: dict[str, Any] = {}
        # Role-classified subsets (references into _active_losses values)
        self._objective_losses: dict[str, tuple[str, Any]] = {}  # category → (loss_name, instance)
        self._constraint_losses: OrderedDict[str, tuple[str, Any]] = OrderedDict()  # bb → eikonal → smoothness
        self._auxiliary_losses: dict[str, Any] = {}  # name → instance

        for category, lossnames in losses_block.items():
            if category == "total":
                continue
            for loss_name, (compute, _weight, path) in lossnames.items():
                if not compute:
                    continue
                abs_path: Path = project_root / Path(path)
                family: str = Path(path).stem.title().replace("_", "")
                loss_cls = load_class_from_path(str(abs_path), family)
                instance = loss_cls(
                    device=str(self._device),
                    loss_name=loss_name,
                    losses_config=self.losses_config,
                )
                self._active_losses[loss_name] = instance

                # Classify by category
                if category in _ALM_OBJECTIVE_CATEGORIES:
                    self._objective_losses[category] = (loss_name, instance)
                elif category in _ALM_CONSTRAINT_CATEGORIES:
                    self._constraint_losses[category] = (loss_name, instance)
                else:
                    self._auxiliary_losses[loss_name] = instance

        # Ensure constraint ordering matches the λ/μ/ν vector order #noqa: RUF003
        self._constraint_losses = OrderedDict(
            (cat, self._constraint_losses[cat]) for cat in _ALM_CONSTRAINT_CATEGORIES if cat in self._constraint_losses
        )

    # ------------------------------------------------------------------
    # loss computation
    # ------------------------------------------------------------------
    def _compute_raw_losses(self, model: BaseModel, epoch_name: int) -> dict[str, torch.Tensor]:
        """Evaluate every active loss instance and return raw (unweighted) tensors."""
        raw: dict[str, torch.Tensor] = {}
        for name, instance in self._active_losses.items():
            raw[name] = instance.get_loss(model, epoch_name)
        return raw

    def _apply_alm_weighting(
        self,
        raw_losses: dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Apply Augmented Lagrangian weighting on the constraints, calls the CSE, calculate and return the total loss.

        Populates ``self._last_weighted_losses`` with independent-graph tensors
        suitable for per-component ``.backward(retain_graph=False)``.
        """
        gi: dict[str, Any] = self.losses_config["general_info"]
        use_alm_aggressive: bool = gi.get("ALM_aggressive") == "True"

        # --- Objective: maximise cruising speed or (solar - drag) ----------
        drag_name: str = self._objective_losses["aero"][0]
        solar_name: str = self._objective_losses["solar"][0]
        drag: torch.Tensor = raw_losses[drag_name]
        solar: torch.Tensor = raw_losses[solar_name]

        o: torch.Tensor
        if gi.get("CSE") == "True":
            v_cruise: torch.Tensor = cruising_speed(
                solar_power=solar,
                drag_force_at_25ms=drag,
                mass=210,
                c_losses=20,
                rolling_friction_coefficient=0.006,
                max_iter=12,
                tol=1e-4,
            )
            self._last_cruising_speed = v_cruise
            o = -v_cruise  # minimise negative speed
        else:
            self._last_cruising_speed = None
            o = torch.as_tensor(-1.0 * (solar - drag), device=device)

        # --- Constraints: bb, eikonal, smoothness -------------------------
        # NOTE: we must NOT use torch.stack on live tensors — it creates a
        # shared StackBackward node that causes "backward through the graph a
        # second time" in the per-component loop.
        constraint_names: list[str] = [name for (name, _inst) in self._constraint_losses.values()]
        constraints: list[torch.Tensor] = [raw_losses[n] for n in constraint_names]

        # Detach for the multiplier update (no gradient through λ/μ/ν update) # noqa: RUF003
        c_for_alm: torch.Tensor = torch.stack([c.detach() for c in constraints])

        alm_thresholds = gi.get("alm_thresholds")
        thresholds_tensor: torch.Tensor | None = torch.tensor(alm_thresholds, device=device) if alm_thresholds else None

        self.lamdas, self.mus, self.nus = weights_alm(
            lamdas=self.lamdas,
            mus=self.mus,
            nus=self.nus,
            alphas=self.alphas,
            gamma=self.gamma,
            gamma_aggressive=self.gamma_aggressive,
            constrained_losses=c_for_alm,
            thresholds=thresholds_tensor,
            use_alm_aggressive=use_alm_aggressive,
        )

        # --- Build per-component weighted tensors -------------------------
        self._last_weighted_losses = {"alm_objective": o}
        for i, cname in enumerate(constraint_names):
            ci: torch.Tensor = constraints[i]  # direct reference, no StackBackward
            lam_i: torch.Tensor = self.lamdas[i].detach()
            mu_i: torch.Tensor = self.mus[i].detach()
            self._last_weighted_losses[f"alm_{cname}"] = lam_i * ci + 0.5 * mu_i * ci**2

        weighted_total: torch.Tensor = sum(self._last_weighted_losses.values())  # type: ignore[assignment]
        return weighted_total

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    def _handle_telemetry(
        self,
        loss_log: dict[str, float],
        epoch_name: int,
        device: torch.device,
        model: BaseModel,
        total_loss: torch.Tensor,
    ) -> None:
        """WandB logging, mesh rendering, gradient logging, and memory diagnostics."""
        if not self._use_wandb:
            # Still log gradients (they print to stdout when wandb is off)
            for inst in self._active_losses.values():
                inst.log_gradients(epoch_name)
            self._log_memory_diagnostics(epoch_name, device)
            return

        # --- unweighted losses + total loss metrics -------------------------------------
        wandb.log(loss_log, step=epoch_name)

        # --- individual weighted contributions ------------------
        weighted_log: dict[str, float] = {f"weighted/{name}": val.item() for name, val in self._last_weighted_losses.items()}
        wandb.log(weighted_log, step=epoch_name)

        # --- cruising speed ---------------------------
        if self._last_cruising_speed is not None:
            wandb.log(
                {"metrics/cruising_speed_ms": self._last_cruising_speed.item()},
                step=epoch_name,
            )

        # --- ALM multiplier -------------------------------------
        constraint_names: list[str] = [name for (name, _inst) in self._constraint_losses.values()]
        alm_params: dict[str, float] = {}
        for i, cname in enumerate(constraint_names):
            alm_params[f"alm/lambda_{cname}"] = self.lamdas[i].item()
            alm_params[f"alm/mu_{cname}"] = self.mus[i].item()
            alm_params[f"alm/nu_{cname}"] = self.nus[i].item()
        wandb.log(alm_params, step=epoch_name)

        # --- Mesh extraction & rendering ----------------------------------
        if self._use_renderer and self._renderer_manager is not None:
            verts, faces = self._mesh_extractor.extract_mesh(
                model=model,
                isovalue=self._mesh_extraction_config.get("isovalue", 0.0),
                batch_size=self._mesh_extraction_config.get("batch_size", 8192),
            )
            if self._mesh_statistics_enabled:
                self._mesh_statistics.log_basic_mesh_stats(verts, faces, epoch_name)
                self._mesh_statistics.log_solar_area_stats(verts, faces, epoch_name)

            if self._rendering_freq is not None and epoch_name % self._rendering_freq == 0:
                if self._use_neural_renderer and self._renderer_manager is not None:
                    try:
                        self._renderer_manager.update(
                            {
                                "model": model,
                                "iteration": epoch_name,
                                "loss": total_loss.item(),
                                "vertices": verts,
                                "faces": faces,
                            }
                        )
                    except Exception as e:
                        print(f"Renderer wrapper update failed: {e}")
                else:
                    self._run_cad_vis(verts, faces, epoch_name)
        else:
            print(f"Skipping rendering at epoch {epoch_name} due to rendering frequency setting.")

        # --- Gradient logging for each loss component ---------------------
        for inst in self._active_losses.values():
            inst.log_gradients(epoch_name)

        # --- GPU memory diagnostics ---------------------------------------
        self._log_memory_diagnostics(epoch_name, device)

    def _run_cad_vis(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        epoch_name: int,
    ) -> None:
        """Render mesh via CADVisualizer, log to WandB, and clean up local files."""
        verts_np = verts.detach().cpu().numpy() if torch.is_tensor(verts) else verts
        faces_np = faces.detach().cpu().numpy() if torch.is_tensor(faces) else faces

        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
        stl_path = Path(f"subprojects/wp4_aeroshell_optimizer/v1/logs/mesh_epoch_{epoch_name}.stl")
        stl_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(stl_path))

        image_path = Path(f"subprojects/wp4_aeroshell_optimizer/v1/logs/render_epoch_{epoch_name}.png")
        visualizer = CADVisualizer(stl_path=str(stl_path), out_image_path=str(image_path))
        visualizer.render_until_blue_detected()

        if stl_path.exists():
            wandb.log({"mesh_3d": wandb.Object3D(str(stl_path))}, step=epoch_name)
            stl_path.unlink()
        if image_path.exists():
            wandb.log({"render_image": wandb.Image(str(image_path))}, step=epoch_name)
            image_path.unlink()

    def _log_memory_diagnostics(self, epoch_name: int, device: torch.device) -> None:
        """Print and optionally log GPU memory stats (toggled by logging.yaml)."""
        if not self._memory_diagnostics:
            return
        if device.type != "cuda":
            return
        allocated_mb: float = torch.cuda.memory_allocated(device) / 1024**2
        reserved_mb: float = torch.cuda.memory_reserved(device) / 1024**2
        peak_mb: float = torch.cuda.max_memory_allocated(device) / 1024**2
        free_cached_mb: float = reserved_mb - allocated_mb
        print(
            f"[Epoch {epoch_name}] GPU Memory | "
            f"Allocated: {allocated_mb:.1f} MB | "
            f"Reserved: {reserved_mb:.1f} MB | "
            f"Peak: {peak_mb:.1f} MB | "
            f"Free (cached): {free_cached_mb:.1f} MB"
        )
        if self._use_wandb:
            wandb.log(
                {
                    "memory/allocated_mb": allocated_mb,
                    "memory/reserved_mb": reserved_mb,
                    "memory/peak_allocated_mb": peak_mb,
                    "memory/free_cached_mb": free_cached_mb,
                },
                step=epoch_name,
            )

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def get_total_loss(self, model: BaseModel, epoch_name: int = -1) -> torch.Tensor:
        """Initilizes unweighted losses computation, computes weighted total loss, logs to wandb, and returns the total loss tensor.

        After this call, ``self._last_weighted_losses`` contains per-component
        tensors with independent computation graphs suitable for individual
        ``.backward(retain_graph=False)`` calls.
        """
        device: torch.device = next(iter(model.parameters())).device

        # 1. Evaluate all active loss instances
        raw: dict[str, torch.Tensor] = self._compute_raw_losses(model, epoch_name)

        # 2. Apply ALM weighting (populates _last_weighted_losses) & CSE
        total: torch.Tensor = self._apply_alm_weighting(raw, device)

        # 3. Build logging dict and dispatch telemetry
        loss_log: dict[str, float] = {name: t.item() for name, t in raw.items()}
        loss_log["total_loss"] = total.item()
        self._handle_telemetry(loss_log, epoch_name, device, model, total)

        return total
