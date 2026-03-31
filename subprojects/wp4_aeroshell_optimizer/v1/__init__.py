"""v1 package for wp4_aeroshell_optimizer."""

from pathlib import Path

# --- logging (no circular deps) ---
from .logging_lib import image_uploads
from .logging_lib.basic_plots import basic_multi_log, basic_single_log, log_matplotlib_fig, log_renderer_image, multi_log_in_one
from .logging_lib.image_uploads import create_and_upload_gif_from_folder, upload_neural_renderer_image

# --- losses (no circular deps) ---
from .src.losses.cruising_speed import cruising_speed
from .src.losses.weights import weights_alm

# --- utils (kaolin_conversions → model_baseclass → visual_wrapper — no total_loss chain) ---
from .src.utils.config_utils import apply_sweep_overrides, get_master_config, load_config, register_all_configs
from .src.utils.dynamic_loading import dynamic_type_loading
from .src.utils.hashing import dict_hashing
from .src.utils.kaolin_conversions import get_diff_mesh
from .src.utils.storage_utils import allround_save_data, fetch_pretrained_model, reformat_wandb_name
from .src.utils.visualizer import CADVisualizer

# --- visuals (model_baseclass and utils are fully loaded by now) ---
from .src.visuals.detached_modes.mesh_sdf_grads import MeshSDFGrads
from .src.visuals.detached_modes.shading_modes import ShadingPenalty, ShadingVisibility
from .src.visuals.detached_modes.solar_v3_selection import SolarV3Selection
from .src.visuals.visual_wrapper import VisualisationManager, Wrapper
from .src.visuals.visualiser import Visualiser

# --- path constants ---
CONFIGS_DIR = Path(__file__).parent / "configs"
SLURM_SCRIPTS_DIR = Path(__file__).parent / "slurm_scripts"

__all__ = [  # noqa: RUF022
    # visuals
    "Visualiser",
    "VisualisationManager",
    "Wrapper",
    "MeshSDFGrads",
    "ShadingPenalty",
    "ShadingVisibility",
    "SolarV3Selection",
    # utils
    "load_config",
    "register_all_configs",
    "apply_sweep_overrides",
    "get_master_config",
    "dict_hashing",
    "reformat_wandb_name",
    "allround_save_data",
    "fetch_pretrained_model",
    "dynamic_type_loading",
    "get_diff_mesh",
    "CADVisualizer",
    # losses
    "weights_alm",
    "cruising_speed",
    # logging
    "multi_log_in_one",
    "basic_multi_log",
    "basic_single_log",
    "log_matplotlib_fig",
    "log_renderer_image",
    "image_uploads",
    "upload_neural_renderer_image",
    "create_and_upload_gif_from_folder",
    # paths
    "CONFIGS_DIR",
    "SLURM_SCRIPTS_DIR",
]
