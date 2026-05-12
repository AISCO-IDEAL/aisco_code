import time
import os
import hashlib
from datetime import datetime
from pathlib import Path

import torch
import yaml

from subprojects.wp3_solar_estimator.testbench.testbench_main import (
    SOLAR_LOSS_REGISTRY,
    load_ply_file,
    get_device
)
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import NullVisManager

class MetricBenchModel:
    """A minimal wrapper to represent a mesh as an SDF function that supports differentiable vertices.
    This allows gradients to flow backwards into the vertices, simulating a realistic
    backward pass where the geometry itself receives gradients."""
    def __init__(self, verts: torch.Tensor, faces: torch.Tensor):
        self.verts = verts
        self.faces = faces
        
    def __call__(self, pts: torch.Tensor) -> torch.Tensor:
        from subprojects.wp4_aeroshell_optimizer.v2.src.models.blendednet.gpu_sdf import mesh_sdf
        return mesh_sdf(pts, self.verts, self.faces).view(-1)
        
    def get_model_info_dict(self) -> dict:
        return {"model_type": "dynamic_mesh_sdf", "num_vertices": self.verts.shape[0]}


def load_benchmark_config() -> dict:
    cfg_path = Path(__file__).parent / "benchmark_cfg.yaml"
    with cfg_path.open() as f:
        return yaml.safe_load(f)

def run_benchmark():
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
    solar_loss = loss_cls(device=device, weight=[1.0, 1.0])
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
    
    ctx = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=active_steps),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)) if prof_enabled else None,
        record_shapes=prof_cfg.get("record_shapes", True),
        profile_memory=prof_cfg.get("profile_memory", True),
        with_stack=prof_cfg.get("with_stack", True),
    ) if prof_enabled else None

    from contextlib import nullcontext
    
    start_event.record()
    
    with ctx if ctx is not None else nullcontext() as p:
        for step in range(active_steps):
            # also record each iteration time for better insights into compile:
            step_start = time.perf_counter()
            
            loss = solar_loss.get_loss(model=sdf_model, epoch_name=step+1000, skip_telemetry=True)
            loss.backward()
            
            # Print at the first step that gradients successfully reached vertices
            if step == 0 and verts.grad is not None and torch.any(verts.grad != 0):
                print(">> Verified: Gradients successfully backpropagated to inputs (verts).")
                
            if verts.grad is not None:
                verts.grad.zero_()
                
            if ctx is not None:
                p.step()
            
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
        peak_vram = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
        reserved_vram = torch.cuda.max_memory_reserved(device=device) / (1024 ** 2)

    print("\n=== Benchmark Results ===")
    print(f"Avg forward+backward pass : {avg_time_ms:.2f} ms")
    print(f"Peak VRAM allocated       : {peak_vram:.2f} MB")
    print(f"Peak VRAM reserved        : {reserved_vram:.2f} MB")
    
    if prof_enabled:
        trace_files = list(trace_dir.glob("*.pt.trace.json"))
        if not trace_files:
            trace_files = list(trace_dir.glob("*.json"))
        if trace_files:
            newest_trace = max(trace_files, key=os.path.getctime)
            print(f"Profiler trace exported to: {newest_trace}")
            
if __name__ == "__main__":
    run_benchmark()
