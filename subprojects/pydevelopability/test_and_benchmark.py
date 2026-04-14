#!/usr/bin/env python3
"""
Test & benchmark script for pydevelopability.

1. Loads a PLY mesh
2. Computes energy + grad for all 6 energy types via the PyBind11 wrapper
3. Compares against the C++ binary output (if available)
4. Profiles with cProfile
5. Tests torch autograd wrapper

Usage:
    python test_and_benchmark.py [--mesh path/to/mesh.ply]
"""

import argparse
import cProfile
import pstats
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

# ── Import pydevelopability ─────────────────────────────────────────────────
try:
    import pydevelopability as pdev
except ImportError:
    sys.exit("ERROR: pydevelopability not installed. Run: pip install -e .")

# ── Optional: torch ─────────────────────────────────────────────────────────
torch: Any = None
try:
    import torch
    from pydevelopability.torch_wrapper import (
        CachedDevelopabilityEnergy,
        DevelopabilityEnergy,
    )
except ImportError:
    pass


def load_mesh(path):
    """Load a PLY or OBJ mesh, return (V, F) as numpy arrays."""
    mesh = trimesh.load(path, process=False)
    V = np.array(mesh.vertices, dtype=np.float64)
    F = np.array(mesh.faces, dtype=np.int32)
    return V, F


def test_all_energies(V, F):
    """Evaluate all 6 energy types and print results."""
    problem = pdev.build_problem(V, F)
    print(f"\nMesh: {V.shape[0]} vertices, {F.shape[0]} faces")
    print("-" * 60)

    results = {}
    for etype in range(6):
        name = pdev.ENERGY_NAMES[etype]
        t0 = time.perf_counter()
        energy, grad = pdev.evaluate_energy_and_grad(problem, V, etype)
        dt = time.perf_counter() - t0
        total = energy.sum()
        grad_norm = np.linalg.norm(grad)
        results[name] = {
            "total_energy": total,
            "grad_norm": grad_norm,
            "time_ms": dt * 1000,
        }
        print(f"  {name:25s}  E={total:12.6f}  |grad|={grad_norm:12.6f}  ({dt * 1000:.1f} ms)")

    return results


def benchmark_repeated(V, F, energy_type=0, n_iters=50):
    """Benchmark repeated energy evaluations (simulating training loop)."""
    problem = pdev.build_problem(V, F)
    name = pdev.ENERGY_NAMES[energy_type]

    # Warmup
    for _ in range(3):
        pdev.evaluate_energy_and_grad(problem, V, energy_type)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        pdev.evaluate_energy_and_grad(problem, V, energy_type)
    dt = time.perf_counter() - t0

    print(f"\nBenchmark: {name} x{n_iters} iters")
    print(f"  Total: {dt * 1000:.1f} ms, Per-iter: {dt / n_iters * 1000:.2f} ms")
    return dt / n_iters


def make_output_filename(
    mesh_path: Path,
    final_energy: float,
    energy_type: int,
    linesearch: int,
    step_type: int,
    num_steps: int,
    subdivisions: int = 0,
) -> Path:
    """
    Build output filename matching the original CC repo convention:
      {final_energy}_{basename}_energy{E}_linesearch{L}_steptype{S}_iterations{N}_subdivisions{D}.{ext}
    """
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    ext = mesh_path.suffix.lstrip(".")  # 'ply' or 'obj'
    basename = mesh_path.stem
    fname = (
        f"{final_energy:.6g}_{basename}"
        f"_energy{energy_type}"
        f"_linesearch{linesearch}"
        f"_steptype{step_type}"
        f"_iterations{num_steps}"
        f"_subdivisions{subdivisions}"
        f".{ext}"
    )
    return output_dir / fname


def save_mesh(path: Path, V, F) -> None:
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    mesh.export(str(path))
    print(f"  Saved: {path}")


def benchmark_optimization(V, F, mesh_path: Path, energy_type=0, linesearch=1, step_type=1, num_steps=100, subdivisions=0):
    """Benchmark the full C++ optimization loop and save the optimized mesh."""
    problem = pdev.build_problem(V, F)
    name = pdev.ENERGY_NAMES[energy_type]

    print(
        f"\nOptimization: {name}, linesearch={linesearch}, steptype={step_type}, {num_steps} steps, {subdivisions} subdivisions"
    )
    print(f"  Initial: {V.shape[0]} vertices, {F.shape[0]} faces")

    t0 = time.perf_counter()
    result = pdev.run_optimization(problem, V, energy_type, linesearch, step_type, num_steps, subdivisions)
    dt = time.perf_counter() - t0

    final_energy = result.energy_history[-1]
    print(f"\n  Time: {dt * 1000:.1f} ms ({dt / (num_steps * (subdivisions + 1)) * 1000:.2f} ms/step avg)")
    print(f"  Energy: {result.energy_history[0]:.6f} -> {final_energy:.6f}")
    print(f"  Improvement: {(result.energy_history[0] - final_energy) / result.energy_history[0] * 100:.1f}%")

    # Print subdivision stats
    if result.subdivision_stats:
        print("\n  Subdivision stats:")
        for i, (nv, nf) in enumerate(result.subdivision_stats):
            print(f"    Subdiv {i}: {nv} vertices, {nf} faces")

    # Print energy per step (sample every ~10 steps)
    print(f"\n  Energy history ({len(result.energy_history)} steps):")
    sample_interval = max(1, len(result.energy_history) // 10)
    steps_per_subdiv = num_steps
    subdiv_boundaries = [(i + 1) * steps_per_subdiv for i in range(subdivisions)]

    for i in range(0, len(result.energy_history), sample_interval):
        # Check if we're at or near a subdivision boundary
        marker = ""
        for j, boundary in enumerate(subdiv_boundaries):
            if abs(i - boundary) < sample_interval and i >= boundary:
                marker = f" [=>Subdiv {j + 1}]"
                break
        print(f"    Step {i:4d}: {result.energy_history[i]:12.6f}{marker}")

    # Print final step if not already printed
    if (len(result.energy_history) - 1) % sample_interval != 0:
        final_marker = ""
        if subdivisions > 0 and len(result.energy_history) - 1 >= (subdivisions) * steps_per_subdiv:
            final_marker = ""
        print(f"    Step {len(result.energy_history) - 1:4d}: {result.energy_history[-1]:12.6f}{final_marker}")

    out_path = make_output_filename(mesh_path, final_energy, energy_type, linesearch, step_type, num_steps, subdivisions)
    save_mesh(out_path, result.V, result.F)

    return result


def compare_with_cpp_binary(_V, _F, mesh_path, energy_type=0):
    """Run the C++ binary on the same mesh and compare energy values."""
    cpp_bin = Path(__file__).parent.parent / "DevelopabilityOfTriangleMeshes" / "build" / "developableflow"
    if not cpp_bin.exists():
        print(f"\n[SKIP] C++ binary not found at {cpp_bin}")
        return None

    # Run C++ binary with 0 steps to just get initial energy
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            str(cpp_bin),
            str(mesh_path),
            "-e",
            str(energy_type),
            "-n",
            "1",
            "-l",
            "0",
            "-s",
            "0",
            "-f",
            "ply",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=tmpdir, check=False)
            print(f"\n[C++ binary output]:\n{result.stdout[:500]}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"\n[SKIP] C++ binary failed: {e}")
            return None


def test_torch_wrapper(V, F):
    """Test the PyTorch autograd wrapper."""
    if torch is None:
        print("\n[SKIP] torch not available")
        return

    V_t = torch.from_numpy(V).double().requires_grad_(True)
    F_t = torch.from_numpy(F.astype(np.int64)).long()

    print("\n--- PyTorch autograd test ---")
    for etype in range(6):
        name = pdev.ENERGY_NAMES[etype]
        energy_fn = DevelopabilityEnergy(energy_type=etype)

        V_t_clone = V_t.clone().detach().requires_grad_(True)
        loss = energy_fn(V_t_clone, F_t)
        loss.backward()

        assert V_t_clone.grad is not None
        print(f"  {name:25s}  loss={loss.item():12.6f}  |V.grad|={V_t_clone.grad.norm().item():12.6f}")

    # Test cached version
    print("\n--- Cached wrapper (repeated calls) ---")
    cached_fn = CachedDevelopabilityEnergy(energy_type=0)
    times = []
    for _i in range(10):
        V_t_clone = V_t.clone().detach().requires_grad_(True)
        t0 = time.perf_counter()
        loss = cached_fn(V_t_clone, F_t)
        loss.backward()
        dt = time.perf_counter() - t0
        times.append(dt)
    avg = np.mean(times[2:]) * 1000  # skip warmup
    print(f"  HINGE cached avg: {avg:.2f} ms/iter")


def profile_energy_eval(V, F, energy_type=0, n_iters=100):
    """Run cProfile on energy evaluation."""
    problem = pdev.build_problem(V, F)
    name = pdev.ENERGY_NAMES[energy_type]
    print(f"\n{'=' * 60}")
    print(f"cProfile: {name} x{n_iters}")
    print(f"{'=' * 60}")

    def run():
        for _ in range(n_iters):
            pdev.evaluate_energy_and_grad(problem, V, energy_type)

    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)


def profile_torch_wrapper(V, F, energy_type=0, n_iters=50):
    """Run cProfile on the torch wrapper (forward + backward)."""
    if torch is None:
        return

    V_t = torch.from_numpy(V).double()
    F_t = torch.from_numpy(F.astype(np.int64)).long()
    cached_fn = CachedDevelopabilityEnergy(energy_type=energy_type)
    name = pdev.ENERGY_NAMES[energy_type]

    print(f"\n{'=' * 60}")
    print(f"cProfile (torch fwd+bwd): {name} x{n_iters}")
    print(f"{'=' * 60}")

    def run():
        for _ in range(n_iters):
            v = V_t.clone().requires_grad_(True)
            loss = cached_fn(v, F_t)
            loss.backward()

    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)


def main():
    parser = argparse.ArgumentParser(description="Test & benchmark pydevelopability")
    parser.add_argument("--mesh", type=str, default=None, help="Path to a PLY or OBJ mesh file")
    parser.add_argument("--energy", type=int, default=0, help="Energy type for benchmarks (0-5)")
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=100,
        help="Number of iterations for cProfile",
    )
    parser.add_argument("--opt-steps", type=int, default=100, help="Number of optimization steps")
    parser.add_argument("--linesearch", type=int, default=1, help="Linesearch: 0=NONE, 1=OVERTON, 2=BACKTRACK")
    parser.add_argument("--steptype", type=int, default=1, help="Step type: 0=GRADDESC, 1=LBFGS")
    parser.add_argument(
        "--optimize-only", action="store_true", help="Skip benchmarks/profiling, just run optimization and save result"
    )
    parser.add_argument("--subdivisions", type=int, default=0, help="Number of subdivisions (for output filename)")
    args = parser.parse_args()

    # Find a test mesh
    if args.mesh:
        mesh_path = Path(args.mesh)
    else:
        # Look for test meshes in the output directory
        output_dir = Path(__file__).parent.parent / "developability_output"
        if output_dir.exists():
            plys = sorted(output_dir.glob("*.ply"))
            if plys:
                mesh_path = plys[0]
            else:
                sys.exit("No PLY files found in developability_output/. Use --mesh.")
        else:
            sys.exit("No default mesh found. Use --mesh path/to/mesh.ply")

    print(f"Loading mesh: {mesh_path}")
    V, F = load_mesh(str(mesh_path))

    if args.optimize_only:
        print(f"\nMesh: {V.shape[0]} vertices, {F.shape[0]} faces")
        benchmark_optimization(
            V,
            F,
            mesh_path,
            args.energy,
            linesearch=args.linesearch,
            step_type=args.steptype,
            num_steps=args.opt_steps,
            subdivisions=args.subdivisions,
        )
        return

    # 1. Test all energies
    test_all_energies(V, F)

    # 2. Compare with C++ binary
    compare_with_cpp_binary(V, F, mesh_path, args.energy)

    # 3. Benchmark
    benchmark_repeated(V, F, args.energy)

    # 4. Optimization benchmark
    benchmark_optimization(
        V,
        F,
        mesh_path,
        args.energy,
        linesearch=args.linesearch,
        step_type=args.steptype,
        num_steps=args.opt_steps,
        subdivisions=args.subdivisions,
    )

    # 5. Torch wrapper test
    test_torch_wrapper(V, F)

    # 6. cProfile
    profile_energy_eval(V, F, args.energy, args.profile_iters)
    profile_torch_wrapper(V, F, args.energy, min(args.profile_iters, 50))


if __name__ == "__main__":
    main()
