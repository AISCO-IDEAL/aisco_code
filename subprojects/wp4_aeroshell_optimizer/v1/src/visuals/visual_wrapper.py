# The purpose of this file is to wrap the visualiser class and visualiser utilities into one nice thing that is compactly callable.
# What is in this file is supposed to be hassle free for the user! --> Config settable parameters only!
# If you want some custom stuff, dig yourself in visualiser.py or visualiser_utils.py, or even better implement your stuff in here!

from __future__ import annotations

import datetime
import importlib
import traceback as tb
from pathlib import Path
from typing import Any

import torch
import wandb

from subprojects.wp4_aeroshell_optimizer.v1.logging_lib.image_uploads import (
    upload_neural_renderer_gif,
    upload_neural_renderer_image,
)
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.hashing import str_hashing
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base_class import RenderingMode
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser import Visualiser


class VisualisationManager:
    """
    Visual Manager that manages multiple rendering modes using 1 Visualiser Object simultaneously.

    The Manager owns a Visualiser instance and manages a collection of RenderingMode
    instances that all share access to this single Visualiser. Each mode manages
    its own frame storage to enable clean separation and potential parallelization.
    I.e. if multiple ppl want to render different debugging things, they all can do so in their custom mode,
    the renderer then takes care of their needs during the optimization.

    Usage:
        manager = VisualisationManager(verbose=True, output_dir=Path("results/"))

        # During optimization loop:
        for iteration in range(num_iterations):
            manager.update(optimization_state={
                'model': model,
                'iteration': iteration,
                'loss': loss_val,
                'vertices': vertices,
                'faces': faces
            })

        # At the end:
        manager.finalize()
    """

    def __init__(
        self,
        verbose: bool = True,
        output_dir: Path | None = None,
        visualiser: Visualiser | None = None,
        key_for_modes: str = "rendering_modes",
    ):
        """
        Initialize manager, the shared Visualiser, and all rendering modes from config.

        Args:
            verbose: Print status information during initialization and finalization
            output_dir: Base directory for saving results. Defaults to ./rendering_results
                       Each mode gets its own subdirectory: output_dir/mode_name/
        """
        # Load configuration
        # The visualiser config is nested under "neural_renderer" in the general config.
        # It contains all the settings that define a neural renderer object, and all modes will run through this same neural renderer instance.
        general_config = load_config(cfg_name="general_config.yaml")
        neural_renderer_config_name = general_config["active_configs"]["neural_renderer"]
        self.config = load_config(v="v1", cfg_name=neural_renderer_config_name)
        self.store_wandb = self.config["general"].get("store_wandb", False)  # Whether to upload final GIFs to wandb

        # Initialize the shared Visualiser instance
        if visualiser is not None:
            self.visualiser = visualiser
        else:
            self.visualiser = Visualiser(**(self.config.get("neural_renderer", {})))

        # set output dir (WIP, not sure if keeping this way)

        self.verbose = verbose
        self.output_dir = output_dir or Path("/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/rendering_results")
        self.output_dir.mkdir(parents=False, exist_ok=True)

        # Initialize all rendering modes
        self.rendering_modes: dict[str, RenderingMode] = {}
        self._initialize_rendering_modes(key=key_for_modes)

        if verbose:
            print("Initialized Visualisation Manager")
            self.status()

    def _initialize_rendering_modes(self, key: str = "rendering_modes") -> None:
        """
        Dynamically load and initialize all rendering modes from config.

        Config structure expected:
            rendering_modes:
              mode_name:
                enabled: true
                class_path: "custom_modes.rendering_mode_example:ExampleRenderingMode"
                config:
                  camera_position: [4000, 0, 0]
                  look_at: [0, 0, 0]
                  render_mode: "rgb"
                  fps: 15

        Each rendering mode is instantiated with:
        - visualiser: self.visualiser (the shared Visualiser instance)
        - config: The mode-specific config dict from YAML
        """
        modes_config = self.config.get(key, {})
        if not modes_config["enabled"]:
            if self.verbose:
                print("Rendering Manager is disabled in config")
            return
        # get rid of "enabled" key, no longer needed at this point, we check it at mode level
        modes_config.pop("enabled", None)
        for mode_name, mode_config in modes_config.items():
            if not mode_config.get("enabled", False):
                if self.verbose:
                    print(f"  Skipping disabled mode: {mode_name}")
                continue

            try:
                # Import the rendering mode class dynamically
                class_path = mode_config.get("class_path")
                if not class_path:
                    print(f"Warning: No class_path for mode {mode_name}, skipping")
                    continue

                module_name, class_name = class_path.rsplit(":", 1)

                # Import from custom_modes package
                module = importlib.import_module(f"subprojects.wp4_aeroshell_optimizer.v1.src.visuals.{module_name}")
                rendering_mode_class = getattr(module, class_name)

                # Instantiate the rendering mode with shared Visualiser and config
                # key change: passing self.visualiser, not self
                mode_instance = rendering_mode_class(visualiser=self.visualiser, config=mode_config.get("config", {}))

                # Set enabled flag after instantiation (it's at YAML mode level, not inside config)
                mode_instance.enabled = mode_config.get("enabled", False)
                self.rendering_modes[mode_name] = mode_instance

                if self.verbose:
                    print(f"  Loaded rendering mode: {mode_name}")

            except Exception as e:
                print(f"Error loading rendering mode '{mode_name}': {e}")
                tb.print_exc()

    def status(self) -> None:
        """Print status of all active rendering modes."""
        active_modes = [name for name, mode in self.rendering_modes.items() if mode.enabled is not None and mode.enabled]
        total_modes = len(self.rendering_modes)
        print(f"Total modes: {total_modes} | Active modes ({len(active_modes)}): {active_modes}")
        print(f"Output directory: {self.output_dir}")

    def update(self, optimization_state: dict[str, Any]) -> None:
        """
        Call update on all active rendering modes.

        Each mode processes the optimization state independently and manages
        its own frame storage. Errors in one mode don't affect others.

        Args:
            optimization_state: Dictionary with optimization step data:
                - 'model': The current neural network model
                - 'iteration': Current iteration number (int)
                - 'loss': Current loss value (float)
                - 'vertices': Current mesh vertices (optional)
                - 'faces': Current mesh faces (optional)
                - Any custom keys specific to your modes
        """
        # Detach all tensors before dispatching to any mode so no rendering
        # operation can retain or extend the training computation graph.
        optimization_state = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in optimization_state.items()}
        step = optimization_state.get("iteration")

        for mode_name, mode in self.rendering_modes.items():
            if mode.enabled:
                try:
                    frames_before = len(mode.frame_storage)
                    mode.update(optimization_state)
                    # Stream each new frame to the wandb dashboard as a wandb.Image
                    if self.store_wandb and wandb.run and len(mode.frame_storage) > frames_before:
                        upload_neural_renderer_image(
                            mode.frame_storage[-1],
                            key=f"renders/{mode_name}",
                            step=step,
                        )
                except Exception as e:
                    print(f"Error in {mode_name}.update(): {e}")
                    tb.print_exc()

    def finalize(self, custom_name: str | None = None, delete_frames: bool = False) -> None:  # noqa: PLR0912
        """
        Call finalize on all rendering modes to save their results.

        if delete_frames: after rendering, delete frames so a new run can be started without old frames interfering. Use with caution, only if you are sure you don't need the frames anymore (e.g. after saving GIFs, or if you just want to clear the buffer for the next run).
        Each mode saves its accumulated frames, GIFs, images, etc. to its
        own subdirectory. Errors in one mode don't prevent others from finalizing.
        """
        if self.verbose:
            print(f"\nFinalizing rendering modes to {self.output_dir}")

        for mode_name, mode in self.rendering_modes.items():
            if mode.enabled:
                try:
                    # Create a subdirectory for this mode
                    mode_output_dir = self.output_dir / mode_name
                    mode_output_dir.mkdir(parents=True, exist_ok=True)
                    # Use a hash of the current time as filename to ensure uniqueness
                    current_time = datetime.datetime.now().isoformat()
                    canonical_time_bytes = str(current_time.encode("utf-8"))
                    file_name = str_hashing(canonical_time_bytes) + ".gif"
                    if custom_name:
                        file_name = custom_name + "_" + file_name
                    mode.finalize(mode_output_dir, file_name=file_name, delete=delete_frames)

                    if self.verbose:
                        print(f"  Finalized {mode_name} → {mode_output_dir}")

                except Exception as e:
                    print(f"Error finalizing {mode_name}: {e}")
                    tb.print_exc()
        if self.store_wandb:
            print("All modes finalized. Uploading to wandb...")
            # Check the output dir and subfolders for the generated GIFs and upload them to wandb
            for mode_name, mode in self.rendering_modes.items():
                if mode.enabled:
                    mode_output_dir = self.output_dir / mode_name
                    if mode_output_dir.exists() and mode_output_dir.is_dir():
                        for file in mode_output_dir.iterdir():
                            suffix = file.suffix.lower()
                            if suffix == ".gif":
                                print(f"Uploading {file} to wandb under key animations/{mode_name}...")
                                upload_neural_renderer_gif(str(file), key=f"animations/{mode_name}", step=0)
                            elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                                print(f"Uploading {file} to wandb under key images/{mode_name}...")
                                upload_neural_renderer_image(str(file), key=f"images/{mode_name}", step=0)
                    else:
                        print(f"Output directory for mode {mode_name} does not exist: {mode_output_dir}")


# Deprecated alias for backward compatibility
Wrapper = VisualisationManager
