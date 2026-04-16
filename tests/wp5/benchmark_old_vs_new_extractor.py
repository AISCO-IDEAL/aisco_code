"""Head-to-head benchmark: OLD uniform MeshSDF vs NEW adaptive WP5 extractor.
Measures extraction time, gradient-flow time, peak GPU memory, vertex count.
Run via SLURM on RTX 4090 or locally with:
    python tests/wp5/benchmark_old_vs_new_extractor.py [--resolution 128]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from subprojects.wp3_solar_estimator.mesh_extractors.mesh_sdf.mesh_sdf_extractor import MeshSDFExtractor as OldExtractor

try:
    from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor as NewExtractor
    from subprojects.wp5.unified_mesh_extraction import load_mesh_extraction_config

    _NEW_EXTRACTOR_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    NewExtractor = None
    load_mesh_extraction_config = None
    _NEW_EXTRACTOR_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Standalone WIRE-like SDF model (same arch as WIREV2, ~133k params, avoids
# importing the full WP4 stack with wandb / TotalLoss / VisualisationManager)
# ---------------------------------------------------------------------------
class _GaborLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, omega: float = 0.8, sigma: float = 10.0) -> None:
        super().__init__()
        self.omega_0 = nn.Parameter(torch.tensor(omega))
        self.scale_0 = sigma
        self.splitter = nn.Linear(in_f, 2 * out_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wave, decay = torch.chunk(self.splitter(x), 2, dim=-1)
        return torch.cos(self.omega_0 * wave) * torch.exp(-(self.scale_0 * decay**2))


class WireSDF(nn.Module):
    """Mimics WIREV2 architecture: 3→128 Gabor, 4*128→128 Gabor, 128→1 linear."""

    def __init__(self, hidden: int = 128, layers: int = 4) -> None:
        super().__init__()
        parts: list[nn.Module] = [_GaborLayer(3, hidden, omega=0.8, sigma=10.0)]
        parts.extend(_GaborLayer(hidden, hidden, omega=0.8, sigma=10.0) for _ in range(layers))
        parts.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*parts)
        # Bias final layer so the zero-level-set sits inside the bbox
        with torch.no_grad():
            self.net[-1].bias.fill_(0.05)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords / 1000.0) * 1000.0


# ---------------------------------------------------------------------------
# Synthetic loss function (stands in for solar / aero loss)
# ---------------------------------------------------------------------------
def _dummy_loss(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = torch.cross(v1 - v0, v2 - v0, dim=1)
    area = 0.5 * torch.norm(cross, dim=1).sum()
    return area  # differentiable surface-area loss


# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    label: str
    resolution: int
    num_vertices: int
    num_faces: int
    extract_time_s: float
    grad_flow_time_s: float
    total_time_s: float
    peak_gpu_alloc_mb: float
    peak_gpu_reserved_mb: float
    loss_value: float
    surrogate_value: float
    success: bool
    error: str


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------
def _reset_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def _raise_empty_mesh(extractor_label: str) -> None:
    raise RuntimeError(f"Empty mesh from {extractor_label} extractor")


def _run_old_extractor(model: nn.Module, resolution: int, bbox_flat: list[float], device: str) -> BenchResult:
    _reset_cuda()
    ext = OldExtractor(resolution=resolution, bbox=bbox_flat, isovalue=0.0, device=device)
    ext.set_grid(
        [[bbox_flat[0], bbox_flat[1]], [bbox_flat[2], bbox_flat[3]], [bbox_flat[4], bbox_flat[5]]],
        resolution,
    )
    error_msg = ""
    try:
        t0 = time.perf_counter()
        vertices, faces = ext.extract_mesh(model)
        torch.cuda.synchronize()
        t_extract = time.perf_counter() - t0
        if vertices.shape[0] == 0:
            _raise_empty_mesh("old")
        t1 = time.perf_counter()
        vertices_for_loss = vertices.detach().requires_grad_(True)
        loss = _dummy_loss(vertices_for_loss, faces)
        dL_dv = torch.autograd.grad(loss, vertices_for_loss, torch.ones_like(loss))[0]
        surrogate = ext.compute_gradient_flow(vertices, dL_dv, model)
        surrogate.backward()
        torch.cuda.synchronize()
        t_grad = time.perf_counter() - t1
        t_total = time.perf_counter() - t0
        peak_alloc = torch.cuda.max_memory_allocated() / 1e6
        peak_reserved = torch.cuda.max_memory_reserved() / 1e6
        return BenchResult(
            label="old_uniform",
            resolution=resolution,
            num_vertices=int(vertices.shape[0]),
            num_faces=int(faces.shape[0]),
            extract_time_s=round(t_extract, 4),
            grad_flow_time_s=round(t_grad, 4),
            total_time_s=round(t_total, 4),
            peak_gpu_alloc_mb=round(peak_alloc, 1),
            peak_gpu_reserved_mb=round(peak_reserved, 1),
            loss_value=round(float(loss.item()), 2),
            surrogate_value=round(float(surrogate.item()), 2),
            success=True,
            error="",
        )
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated() / 1e6
        peak_reserved = torch.cuda.max_memory_reserved() / 1e6
        return BenchResult(
            label="old_uniform",
            resolution=resolution,
            num_vertices=0,
            num_faces=0,
            extract_time_s=0.0,
            grad_flow_time_s=0.0,
            total_time_s=0.0,
            peak_gpu_alloc_mb=round(peak_alloc, 1),
            peak_gpu_reserved_mb=round(peak_reserved, 1),
            loss_value=0.0,
            surrogate_value=0.0,
            success=False,
            error=error_msg,
        )


def _run_new_extractor(
    model: nn.Module,
    base_resolution: int,
    max_refine_levels: int,
    leaf_cell_resolution: int,
    bbox_nested: list[list[float]],
    device: str,
) -> BenchResult:
    _reset_cuda()
    error_msg = ""
    if NewExtractor is None or load_mesh_extraction_config is None:
        return BenchResult(
            label="new_adaptive",
            resolution=base_resolution,
            num_vertices=0,
            num_faces=0,
            extract_time_s=0.0,
            grad_flow_time_s=0.0,
            total_time_s=0.0,
            peak_gpu_alloc_mb=0.0,
            peak_gpu_reserved_mb=0.0,
            loss_value=0.0,
            surrogate_value=0.0,
            success=False,
            error=f"ImportError: {_NEW_EXTRACTOR_IMPORT_ERROR}",
        )
    cfg = load_mesh_extraction_config()
    cfg = replace(
        cfg,
        device=device,
        bbox=bbox_nested,
        base_resolution=base_resolution,
        max_refine_levels=max_refine_levels,
        leaf_cell_resolution=leaf_cell_resolution,
    )
    ext = NewExtractor(config=cfg)
    try:
        t0 = time.perf_counter()
        vertices, faces = ext.extract_mesh(model)
        torch.cuda.synchronize()
        t_extract = time.perf_counter() - t0
        if vertices.shape[0] == 0:
            _raise_empty_mesh("new")
        t1 = time.perf_counter()
        vertices_for_loss = vertices.detach().requires_grad_(True)
        loss = _dummy_loss(vertices_for_loss, faces)
        dL_dv = torch.autograd.grad(loss, vertices_for_loss, torch.ones_like(loss))[0]
        surrogate = ext.compute_gradient_flow(vertices, dL_dv, model)
        surrogate.backward()
        torch.cuda.synchronize()
        t_grad = time.perf_counter() - t1
        t_total = time.perf_counter() - t0
        peak_alloc = torch.cuda.max_memory_allocated() / 1e6
        peak_reserved = torch.cuda.max_memory_reserved() / 1e6
        return BenchResult(
            label="new_adaptive",
            resolution=base_resolution,
            num_vertices=int(vertices.shape[0]),
            num_faces=int(faces.shape[0]),
            extract_time_s=round(t_extract, 4),
            grad_flow_time_s=round(t_grad, 4),
            total_time_s=round(t_total, 4),
            peak_gpu_alloc_mb=round(peak_alloc, 1),
            peak_gpu_reserved_mb=round(peak_reserved, 1),
            loss_value=round(float(loss.item()), 2),
            surrogate_value=round(float(surrogate.item()), 2),
            success=True,
            error="",
        )
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated() / 1e6
        peak_reserved = torch.cuda.max_memory_reserved() / 1e6
        return BenchResult(
            label="new_adaptive",
            resolution=base_resolution,
            num_vertices=0,
            num_faces=0,
            extract_time_s=0.0,
            grad_flow_time_s=0.0,
            total_time_s=0.0,
            peak_gpu_alloc_mb=round(peak_alloc, 1),
            peak_gpu_reserved_mb=round(peak_reserved, 1),
            loss_value=0.0,
            surrogate_value=0.0,
            success=False,
            error=error_msg,
        )


# ---------------------------------------------------------------------------
def _print_table(rows: list[BenchResult]) -> None:
    hdr = (
        f"{'Label':<16} {'Res':>5} {'Vertices':>10} {'Faces':>10} "
        f"{'Extract(s)':>11} {'GradFlow(s)':>12} {'Total(s)':>9} "
        f"{'PeakAlloc(MB)':>14} {'PeakRsrv(MB)':>14} {'OK':>4} {'Error'}"
    )
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.label:<16} {r.resolution:>5} {r.num_vertices:>10} {r.num_faces:>10} "
            f"{r.extract_time_s:>11.4f} {r.grad_flow_time_s:>12.4f} {r.total_time_s:>9.4f} "
            f"{r.peak_gpu_alloc_mb:>14.1f} {r.peak_gpu_reserved_mb:>14.1f} {'yes' if r.success else 'NO':>4} {r.error}"
        )
    print("=" * len(hdr))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark old uniform vs new adaptive mesh extractor")
    parser.add_argument(
        "--resolutions", nargs="+", type=int, default=[64, 128, 256], help="Grid resolutions to test for the old extractor"
    )
    parser.add_argument("--new-base-res", type=int, default=32, help="Base resolution for new adaptive extractor")
    parser.add_argument(
        "--new-refine-levels", nargs="+", type=int, default=[3, 4, 5], help="Refinement levels to test for the new extractor"
    )
    parser.add_argument("--new-leaf-res", type=int, default=3, help="Leaf cell resolution for new extractor")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--output-json", default=None, help="Path to write results JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    _ = np.random.default_rng(args.seed)
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1
    device = args.device
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e6
    print(f"GPU: {gpu_name}  ({gpu_mem:.0f} MB)")
    # ── Create SDF model ──────────────────────────────────────────────
    model = WireSDF(hidden=128, layers=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SDF model: WireSDF  params={n_params}")
    bbox_flat = [-5000.0, 5000.0, -5000.0, 5000.0, -5000.0, 5000.0]
    bbox_nested = [[-5000.0, 5000.0], [-5000.0, 5000.0], [-5000.0, 5000.0]]
    # ── Warm-up pass ──────────────────────────────────────────────────
    with torch.no_grad():
        _ = model(torch.randn(256, 3, device=device))
    torch.cuda.synchronize()
    results: list[BenchResult] = []
    # ── OLD extractor at each resolution ──────────────────────────────
    for res in args.resolutions:
        print(f"\n[OLD uniform] resolution={res}  ({res**3:,} voxels) ...")
        model.zero_grad(set_to_none=True)
        r = _run_old_extractor(model, res, bbox_flat, device)
        results.append(r)
        tag = "OK" if r.success else f"FAIL: {r.error}"
        print(
            f"  → V={r.num_vertices}  F={r.num_faces}  extract={r.extract_time_s:.3f}s  "
            f"grad_flow={r.grad_flow_time_s:.3f}s  peak_alloc={r.peak_gpu_alloc_mb:.0f}MB  {tag}"
        )
    # ── Fully recreate model to ensure no OOM residue from old runs ──
    del model
    _reset_cuda()
    model = WireSDF(hidden=128, layers=4).to(device)
    # ── NEW extractor at each refinement level ────────────────────────
    for levels in args.new_refine_levels:
        print(f"\n[NEW adaptive] base_res={args.new_base_res}  refine_levels={levels}  leaf_res={args.new_leaf_res} ...")
        model.zero_grad(set_to_none=True)
        r = _run_new_extractor(model, args.new_base_res, levels, args.new_leaf_res, bbox_nested, device)
        results.append(r)
        tag = "OK" if r.success else f"FAIL: {r.error}"
        print(
            f"  → V={r.num_vertices}  F={r.num_faces}  extract={r.extract_time_s:.3f}s  "
            f"grad_flow={r.grad_flow_time_s:.3f}s  peak_alloc={r.peak_gpu_alloc_mb:.0f}MB  {tag}"
        )
    # ── Summary table ─────────────────────────────────────────────────
    print("\n")
    _print_table(results)
    # ── Write JSON ────────────────────────────────────────────────────
    out_path = args.output_json
    if out_path is None:
        out_dir = Path("tests/wp5/benchmark_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "old_vs_new_extractor.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"gpu": gpu_name, "gpu_mem_mb": round(gpu_mem), "model_params": n_params, "results": [asdict(r) for r in results]},
            f,
            indent=2,
        )
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
