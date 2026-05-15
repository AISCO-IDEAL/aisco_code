import inspect
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from subprojects.wp3_solar_estimator.rollout_losses.MRRT_SolarLoss import solar_loss_mrrt
from subprojects.wp3_solar_estimator.rollout_losses.SRRT_SolarLoss import solar_loss_srrt
from subprojects.wp3_solar_estimator.testbench.testbench_main import SOLAR_LOSS_REGISTRY, get_device, load_ply_file
from subprojects.wp4_aeroshell_optimizer.v2.src.models.blendednet.gpu_sdf import mesh_sdf
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import NullVisManager


class MetricBenchModel:
    """A minimal wrapper to represent a mesh as an SDF function that supports differentiable vertices.
    This allows gradients to flow backwards into the vertices, simulating a realistic
    backward pass where the geometry itself receives gradients."""

    def __init__(self, verts: torch.Tensor, faces: torch.Tensor):
        self.verts = verts
        self.faces = faces

    def __call__(self, pts: torch.Tensor) -> torch.Tensor:

        return mesh_sdf(pts, self.verts, self.faces).view(-1)

    def get_model_info_dict(self) -> dict:
        return {"model_type": "dynamic_mesh_sdf", "num_vertices": self.verts.shape[0]}


def load_benchmark_config() -> dict:
    cfg_path = Path(__file__).parent / "benchmark_cfg.yaml"
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def run_benchmark():  # noqa: PLR0912, PLR0915
    cfg = load_benchmark_config()
    loss_class = cfg["solar_loss"]["class"]

    benchmark_cfg = cfg.get("benchmark", {})
    warmup_steps = int(benchmark_cfg.get("warmup_steps", 10))
    active_steps = int(benchmark_cfg.get("active_steps", 50))

    prof_cfg = cfg.get("profiler", {})
    prof_enabled = prof_cfg.get("enabled", True)
    trace_dir = Path(__file__).parent / prof_cfg.get("trace_dir", "traces")
    if prof_enabled:
        trace_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Solar Benchmarking ===")
    print(f"Loss class   : {loss_class}")
    print(f"Warmup steps : {warmup_steps}")
    print(f"Active steps : {active_steps}")
    print(f"Profiler     : {'enabled' if prof_enabled else 'disabled'}")

    device = get_device()
    print(f"Device       : {device}")

    ply_path = cfg["loading"]["ply_path"]
    ply_scaling = cfg["loading"]["ply_scaling_factor"]
    print(f"Loading PLY  : {ply_path}  (scale={ply_scaling})")

    verts, faces = load_ply_file(ply_path, ply_scaling)
    verts = verts.to(device)
    faces = faces.to(device)

    # We want gradients w.r.t the vertices
    verts.requires_grad_(True)

    print(f"Initialising {loss_class}...")
    loss_cls = SOLAR_LOSS_REGISTRY[loss_class]
    solar_loss: Any = loss_cls(device=device, weight=[1.0, 1.0])
    solar_loss._vis_manager = NullVisManager()

    # Provide model that computes exact mesh SDF every call so gradients flow directly to `verts`
    sdf_model = MetricBenchModel(verts, faces)

    print("\n--- Starting Warmup ---")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for i in range(warmup_steps):
        step_start = time.perf_counter()
        loss = solar_loss.get_loss(model=sdf_model, epoch_name=i, skip_telemetry=True)
        loss.backward()
        if verts.grad is not None:
            verts.grad.zero_()
        step_time_ms = (time.perf_counter() - step_start) * 1000
        print(f"Warmup step {i}: {step_time_ms:.2f} ms")

    print(f"--- Starting Active Tuning & Profiling ({active_steps} steps) ---")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.empty_cache()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    ctx = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=active_steps),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)) if prof_enabled else None,
            record_shapes=prof_cfg.get("record_shapes", True),
            profile_memory=prof_cfg.get("profile_memory", True),
            with_stack=prof_cfg.get("with_stack", True),
        )
        if prof_enabled
        else None
    )

    start_event.record()

    with ctx if ctx is not None else nullcontext() as p:
        for step in range(active_steps):
            # also record each iteration time for better insights into compile:
            step_start = time.perf_counter()

            loss = solar_loss.get_loss(model=sdf_model, epoch_name=step + 1000, skip_telemetry=True)
            loss.backward()

            # Print at the first step that gradients successfully reached vertices
            if step == 0 and verts.grad is not None and torch.any(verts.grad != 0):
                print(">> Verified: Gradients successfully backpropagated to inputs (verts).")

            if verts.grad is not None:
                verts.grad.zero_()

            if ctx is not None:
                cast(torch.profiler.profile, p).step()

            step_time_ms = (time.perf_counter() - step_start) * 1000
            print(f"Step {step}: {step_time_ms:.2f} ms")

    end_event.record()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / max(active_steps, 1)

    peak_vram = 0.0
    reserved_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(device=device) / (1024**2)
        reserved_vram = torch.cuda.max_memory_reserved(device=device) / (1024**2)

    print("\n=== Benchmark Results (Full get_loss: SDF + Visibility) ===")
    print(f"Avg forward+backward pass : {avg_time_ms:.2f} ms")
    print(f"Peak VRAM allocated       : {peak_vram:.2f} MB")
    print(f"Peak VRAM reserved        : {reserved_vram:.2f} MB")

    # Measure VISIBILITY ONLY (isolated from SDF extraction)
    print("\n--- Measuring Visibility Computation Alone (via raytracing functions) ---")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.empty_cache()

    # Pre-extract mesh once so we have static vertices/faces to measure against
    mesh_verts, mesh_faces, _, _, _ = solar_loss.extractor.extract_and_compute_loss(
        sdf_model=sdf_model,
        loss_fn=lambda *_: torch.zeros((), device=device),  # dummy loss
        enable_backward=False,
    )
    mesh_verts.requires_grad_(True)

    print(f"Extracted mesh: {mesh_verts.shape[0]} vertices, {mesh_faces.shape[0]} faces")

    # Get the actual loss function from the loss object (e.g., solar_loss_mrrt, solar_loss_srrt)
    loss_source = inspect.getsourcefile(solar_loss.__class__)
    print(f"Loss class: {solar_loss.__class__.__name__} from {loss_source}")

    vis_start_event = torch.cuda.Event(enable_timing=True)
    vis_end_event = torch.cuda.Event(enable_timing=True)

    # Warmup for visibility-only measurement
    print("Warming up visibility-only measurement...")
    for _ in range(5):
        # Call the actual raytracing loss function directly with fixed mesh
        # This bypasses the extractor and SDF eval entirely
        if hasattr(solar_loss, "raytracing_cfg"):  # SRRT/MRRT classes have this
            if "MRRT" in solar_loss.__class__.__name__:
                loss_rt, _ = solar_loss_mrrt(
                    mesh_verts,
                    mesh_faces,
                    solar_loss.solar_vectors.to(device),
                    area_budget_mm2=solar_loss.area_budget_mm2,
                    raytracing_cfg=solar_loss.raytracing_cfg,
                    effective_irradiance_W_m2=solar_loss.effective_irradiance_W_m2,
                    sigmoid_sharpness=solar_loss.sigmoid_sharpness,
                    soft_mask_selection_threshold=solar_loss.soft_mask_selection_threshold,
                    soft_mask_full_selection_threshold=solar_loss.soft_mask_full_selection_threshold,
                    shading_penalty_weight=solar_loss.shading_penalty_weight,
                    kappa=solar_loss.kappa,
                    _lambda=solar_loss._lambda,
                    epsilon=solar_loss.epsilon,
                    skip_backface_receivers=solar_loss.skip_backface_receivers,
                    vis_manager=NullVisManager(),
                    epoch_name=None,
                    mesh_visibility_fn=solar_loss._mesh_visibility_fn if hasattr(solar_loss, "_mesh_visibility_fn") else None,
                )
            elif "SRRT" in solar_loss.__class__.__name__:
                loss_rt, _ = solar_loss_srrt(
                    mesh_verts,
                    mesh_faces,
                    solar_loss.sun_dir.to(device),
                    area_budget_mm2=solar_loss.area_budget_mm2,
                    raytracing_cfg=solar_loss.raytracing_cfg,
                    effective_irradiance_W_m2=solar_loss.effective_irradiance_W_m2,
                    sigmoid_sharpness=solar_loss.sigmoid_sharpness,
                    soft_mask_selection_threshold=solar_loss.soft_mask_selection_threshold,
                    soft_mask_full_selection_threshold=solar_loss.soft_mask_full_selection_threshold,
                    shading_penalty_weight=solar_loss.shading_penalty_weight,
                    kappa=solar_loss.kappa,
                    _lambda=solar_loss._lambda,
                    epsilon=solar_loss.epsilon,
                    skip_backface_receivers=solar_loss.skip_backface_receivers,
                    vis_manager=NullVisManager(),
                    epoch_name=None,
                    mesh_visibility_fn=solar_loss._mesh_visibility_fn if hasattr(solar_loss, "_mesh_visibility_fn") else None,
                )
            loss_rt.backward()
            if mesh_verts.grad is not None:
                mesh_verts.grad.zero_()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)

    vis_start_event.record()
    for step in range(active_steps):
        step_start = time.perf_counter()

        if "MRRT" in solar_loss.__class__.__name__:
            loss_rt, _ = solar_loss_mrrt(
                mesh_verts,
                mesh_faces,
                solar_loss.solar_vectors.to(device),
                area_budget_mm2=solar_loss.area_budget_mm2,
                raytracing_cfg=solar_loss.raytracing_cfg,
                effective_irradiance_W_m2=solar_loss.effective_irradiance_W_m2,
                sigmoid_sharpness=solar_loss.sigmoid_sharpness,
                soft_mask_selection_threshold=solar_loss.soft_mask_selection_threshold,
                soft_mask_full_selection_threshold=solar_loss.soft_mask_full_selection_threshold,
                shading_penalty_weight=solar_loss.shading_penalty_weight,
                kappa=solar_loss.kappa,
                _lambda=solar_loss._lambda,
                epsilon=solar_loss.epsilon,
                skip_backface_receivers=solar_loss.skip_backface_receivers,
                vis_manager=NullVisManager(),
                epoch_name=None,
                mesh_visibility_fn=solar_loss._mesh_visibility_fn if hasattr(solar_loss, "_mesh_visibility_fn") else None,
            )
        elif "SRRT" in solar_loss.__class__.__name__:
            loss_rt, _ = solar_loss_srrt(
                mesh_verts,
                mesh_faces,
                solar_loss.sun_dir.to(device),
                area_budget_mm2=solar_loss.area_budget_mm2,
                raytracing_cfg=solar_loss.raytracing_cfg,
                effective_irradiance_W_m2=solar_loss.effective_irradiance_W_m2,
                sigmoid_sharpness=solar_loss.sigmoid_sharpness,
                soft_mask_selection_threshold=solar_loss.soft_mask_selection_threshold,
                soft_mask_full_selection_threshold=solar_loss.soft_mask_full_selection_threshold,
                shading_penalty_weight=solar_loss.shading_penalty_weight,
                kappa=solar_loss.kappa,
                _lambda=solar_loss._lambda,
                epsilon=solar_loss.epsilon,
                skip_backface_receivers=solar_loss.skip_backface_receivers,
                vis_manager=NullVisManager(),
                epoch_name=None,
                mesh_visibility_fn=solar_loss._mesh_visibility_fn if hasattr(solar_loss, "_mesh_visibility_fn") else None,
            )

        loss_rt.backward()
        if mesh_verts.grad is not None:
            mesh_verts.grad.zero_()

        step_time_ms = (time.perf_counter() - step_start) * 1000
        if step < 5:  # Print first 5 steps
            print(f"  Visibility step {step}: {step_time_ms:.2f} ms")

    vis_end_event.record()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    vis_total_time_ms = vis_start_event.elapsed_time(vis_end_event)
    vis_avg_time_ms = vis_total_time_ms / max(active_steps, 1)

    vis_peak_vram = 0.0
    vis_reserved_vram = 0.0
    if torch.cuda.is_available():
        vis_peak_vram = torch.cuda.max_memory_allocated(device=device) / (1024**2)
        vis_reserved_vram = torch.cuda.max_memory_reserved(device=device) / (1024**2)

    print("\n=== Benchmark Results (Raytracing Functions Only, NO SDF extraction) ===")
    print(f"Avg forward+backward pass : {vis_avg_time_ms:.2f} ms")
    print(f"Peak VRAM allocated       : {vis_peak_vram:.2f} MB")
    print(f"Peak VRAM reserved        : {vis_reserved_vram:.2f} MB")
    print(f"RATIO (raytracing / total): {vis_avg_time_ms / avg_time_ms * 100:.1f}% of full get_loss time")

    if prof_enabled:
        trace_files = list(trace_dir.glob("*.pt.trace.json"))
        if not trace_files:
            trace_files = list(trace_dir.glob("*.json"))
        if trace_files:
            newest_trace = max(trace_files, key=os.path.getctime)
            print(f"Profiler trace exported to: {newest_trace}")


if __name__ == "__main__":
    run_benchmark()
