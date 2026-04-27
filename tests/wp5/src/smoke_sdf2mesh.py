"""Smoke benchmark for the rewritten sdf2mesh: differentiability, speed, target
vertex count, memory, and cross-loss mesh reuse."""

from __future__ import annotations

import time
from dataclasses import replace

import torch

from subprojects.wp5.unified_mesh_extraction import (
    MeshSDFExtractor,
    compute_surface_area,
    load_mesh_extraction_config,
)
from subprojects.wp5.unified_mesh_extraction.sdf2mesh import _MESH_CACHE, clear_mesh_cache


class SphereSDF(torch.nn.Module):
    def __init__(self, radius: float = 1500.0) -> None:
        super().__init__()
        self.r = torch.nn.Parameter(torch.tensor(radius))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(p, dim=-1) - self.r


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> None:  # noqa: PLR0915
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    cfg = load_mesh_extraction_config()
    cfg = replace(
        cfg,
        device=device,
        bbox=[[-5000.0, 5000.0]] * 3,
        isovalue=0.0,
        base_resolution=64,
        max_refine_levels=0,
        target_num_vertices=5000,
        vertex_count_tolerance=0.15,
    )
    model = SphereSDF(1500.0).to(device)
    ext = MeshSDFExtractor(config=cfg)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 1) Differentiability
    def area(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return compute_surface_area(v, f)

    _sync(device)
    t0 = time.perf_counter()
    v, f, _loss, surrogate, _ = ext.extract_and_compute_loss(model, area)
    _sync(device)
    t_first = time.perf_counter() - t0
    print(f"[1] extract+loss: {t_first * 1000:.1f}ms  verts={v.shape[0]}  faces={f.shape[0]}")
    assert int(0.85 * 5000) - 1 <= v.shape[0] <= int(1.15 * 5000) + 1

    model.r.grad = None
    surrogate.backward()
    expected = 8 * 3.14159265 * model.r.item()
    print(f"[1] dL/dr={model.r.grad.item():.1f} expected~{expected:.1f}")
    assert abs(model.r.grad.item() - expected) / expected < 0.05

    # 2) Reuse across losses (identical config)
    clear_mesh_cache()
    _sync(device)
    t0 = time.perf_counter()
    v1, f1, *_ = ext.extract_and_compute_loss(model, area)
    _sync(device)
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    v2, f2, *_ = ext.extract_and_compute_loss(model, lambda vv, _ff: vv.mean() ** 2)
    _sync(device)
    t_second = time.perf_counter() - t0
    print(f"[2] reuse: first={t_first * 1000:.1f}ms  second={t_second * 1000:.2f}ms  entries={len(_MESH_CACHE)}")
    assert v1.data_ptr() == v2.data_ptr() and f1.data_ptr() == f2.data_ptr()
    assert t_second < t_first

    # 3) Cache invalidation when params change
    with torch.no_grad():
        model.r.add_(10.0)
    v3, *_ = ext.extract_and_compute_loss(model, area)
    entries = sum(1 for k in _MESH_CACHE if k[0] == id(model))
    print(f"[3] post-update: verts={v3.shape[0]}  entries_for_model={entries}")
    assert v3.data_ptr() != v1.data_ptr() and entries == 1

    # 4) Cross-extractor reuse when configs match
    ext2 = MeshSDFExtractor(config=cfg)
    _sync(device)
    t0 = time.perf_counter()
    v4, *_ = ext2.extract_and_compute_loss(model, area)
    _sync(device)
    print(f"[4] cross-extractor: {1000 * (time.perf_counter() - t0):.2f}ms  shared={v4.data_ptr() == v3.data_ptr()}")
    assert v4.data_ptr() == v3.data_ptr()

    # 5) Larger target + memory
    cfg_big = replace(cfg, target_num_vertices=20000)
    ext_big = MeshSDFExtractor(config=cfg_big)
    clear_mesh_cache()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    _sync(device)
    t0 = time.perf_counter()
    vb, fb, *_ = ext_big.extract_and_compute_loss(model, area)
    _sync(device)
    print(f"[5] target=20000 -> verts={vb.shape[0]}  faces={fb.shape[0]}  time={1000 * (time.perf_counter() - t0):.1f}ms")
    if device == "cuda":
        print(f"[5] peak alloc {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

    # 6) High-resolution sparse marching cubes (no target)
    cfg_hi = replace(cfg, target_num_vertices=None, base_resolution=256, max_refine_levels=0)
    ext_hi = MeshSDFExtractor(config=cfg_hi)
    clear_mesh_cache()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    _sync(device)
    t0 = time.perf_counter()
    vh, fh = ext_hi.extract_mesh(model)
    _sync(device)
    print(f"[6] res=256 sparse -> verts={vh.shape[0]} faces={fh.shape[0]} time={1000 * (time.perf_counter() - t0):.0f}ms")
    if device == "cuda":
        print(f"[6] peak alloc {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
