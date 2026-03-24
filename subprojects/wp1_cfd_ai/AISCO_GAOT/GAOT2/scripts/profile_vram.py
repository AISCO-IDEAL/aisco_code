#!/usr/bin/env python3
"""
Profile GPU VRAM usage during one forward + backward pass of GAOT2.

Shows exactly how memory breaks down between:
  - Model parameters + buffers
  - Optimizer state (Adam momentums)
  - Data batch on GPU
  - Activations / intermediate tensors (forward pass)
  - Gradients (backward pass)
  - Peak memory

Usage:
    python scripts/profile_vram.py --config config/experimental_configs/euler_field_cp_full_oom.yaml
    python scripts/profile_vram.py --config config/main_config.yaml
"""

import argparse
import operator
import os
import sys
from functools import reduce

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.dataloader.pyg_datasets import VTKMeshDataset
from src.dataloader.pyg_transforms import RescalePosition
from src.model import get_model
from src.utils.file_parser import FileParser


def mb(nbytes):
    return nbytes / 1024**2


def print_memory_line(label, value_mb, total_mb=None):
    bar = ""
    if total_mb and total_mb > 0:
        pct = value_mb / total_mb * 100
        bar_len = int(pct / 2)
        bar = f"  [{'█' * bar_len}{'░' * (50 - bar_len)}] {pct:5.1f}%"
    print(f"  {label:<40s} {value_mb:>8.1f} MB{bar}")


def _profile_model(model_config, device, dtype, total_vram_mb):
    """Load model, return (model, param_mem, model_mem, nparams)."""
    print("=" * 70)
    print("Loading model...")
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)

    model = get_model(model_config).to(device).to(dtype)

    mem_after = torch.cuda.memory_allocated(device)
    model_mem = mem_after - mem_before

    param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_mem = sum(b.numel() * b.element_size() for b in model.buffers())
    nparams = sum(p.numel() for p in model.parameters())

    print(f"\nModel: {nparams:,} parameters")
    print_memory_line("Parameters", mb(param_mem), total_vram_mb)
    print_memory_line("Buffers", mb(buffer_mem), total_vram_mb)
    print_memory_line("Total model allocation", mb(model_mem), total_vram_mb)
    return model, param_mem, model_mem


def _profile_optimizer(model, param_mem, total_vram_mb):
    """Create optimizer, return (optimizer, estimated_state_mem)."""
    print("\n" + "=" * 70)
    print("Creating optimizer (AdamW)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    adam_state_est = 2 * param_mem
    print_memory_line("Optimizer state (estimated)", mb(adam_state_est), total_vram_mb)
    return optimizer


def _profile_data(dataset_config, device, total_vram_mb):
    """Load one batch, return (batch, data_mem)."""
    print("\n" + "=" * 70)
    print("Loading one training batch...")

    ds = VTKMeshDataset(
        root=dataset_config["base_path"],
        order_file=None,
        dataset_config=dataset_config,
        split="train",
        transform=RescalePosition(),
    )

    batch_size = dataset_config.get("batch_size", 4)
    loader = PyGDataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    print(f"\n  Batch size: {batch_size}")
    print(f"  Total nodes in batch: {batch.num_nodes:,}")
    print(f"  Total graphs: {batch.num_graphs}")
    batch_cpu_mem = 0
    for key in sorted(batch.keys()):
        val = batch[key]
        if isinstance(val, torch.Tensor):
            tensor_bytes = val.numel() * val.element_size()
            batch_cpu_mem += tensor_bytes
            print(f"    {key:<20s} {list(val.shape)!s:<25s} {mb(tensor_bytes):>8.1f} MB")

    print(f"\n  Batch total on CPU: {mb(batch_cpu_mem):.1f} MB")

    mem_before = torch.cuda.memory_allocated(device)
    batch = batch.to(device)
    mem_after = torch.cuda.memory_allocated(device)
    data_mem = mem_after - mem_before
    print_memory_line("Batch on GPU", mb(data_mem), total_vram_mb)
    return batch, data_mem


def _profile_forward_backward(model, batch, latent_tokens, optimizer, total_vram_mb):
    """Run forward + backward + optimizer step, return memory breakdown."""
    device = next(model.parameters()).device

    # Forward
    print("\n" + "=" * 70)
    print("Running forward pass...")
    mem_before_fwd = torch.cuda.memory_allocated(device)
    model.train()
    pred = model(batch=batch, tokens_pos=latent_tokens)
    mem_after_fwd = torch.cuda.memory_allocated(device)
    fwd_mem = mem_after_fwd - mem_before_fwd
    peak_after_fwd = torch.cuda.max_memory_allocated(device)
    print_memory_line("Activations (forward)", mb(fwd_mem), total_vram_mb)
    print_memory_line("Peak after forward", mb(peak_after_fwd), total_vram_mb)

    # Backward
    print("\n" + "=" * 70)
    print("Running backward pass...")
    mem_before_bwd = torch.cuda.memory_allocated(device)
    if isinstance(pred, dict):
        losses = [v.mean() for v in pred.values() if isinstance(v, torch.Tensor) and v.requires_grad]
        loss = torch.stack(losses).sum()
    else:
        loss = pred.mean()
    loss.backward()
    mem_after_bwd = torch.cuda.memory_allocated(device)
    bwd_mem = mem_after_bwd - mem_before_bwd
    peak_after_bwd = torch.cuda.max_memory_allocated(device)
    print_memory_line("Gradients (backward)", mb(bwd_mem), total_vram_mb)
    print_memory_line("Peak after backward", mb(peak_after_bwd), total_vram_mb)

    # Optimizer step
    print("\n" + "=" * 70)
    print("Running optimizer step...")
    mem_before_step = torch.cuda.memory_allocated(device)
    optimizer.step()
    mem_after_step = torch.cuda.memory_allocated(device)
    opt_step_mem = mem_after_step - mem_before_step
    peak_final = torch.cuda.max_memory_allocated(device)
    print_memory_line("Optimizer state (actual)", mb(opt_step_mem), total_vram_mb)
    print_memory_line("Peak after optim step", mb(peak_final), total_vram_mb)

    return fwd_mem, bwd_mem, opt_step_mem, peak_final


def main():
    parser = argparse.ArgumentParser(description="Profile VRAM usage for one training step")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    parsed = FileParser(args.config).parse_args()
    setup_config = parsed.setup
    dataset_config = parsed.dataset
    model_config = parsed.model

    device = torch.device(setup_config.get("device", "cuda:0"))
    dtype = getattr(torch, setup_config.get("dtype", "torch.float32").replace("torch.", ""))

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available")
        sys.exit(1)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    total_vram_mb = torch.cuda.get_device_properties(device).total_memory / 1024**2
    print(f"\nGPU: {torch.cuda.get_device_name(device)}")
    print(f"Total VRAM: {total_vram_mb:.0f} MB\n")

    model, param_mem, _model_mem = _profile_model(model_config, device, dtype, total_vram_mb)
    optimizer = _profile_optimizer(model, param_mem, total_vram_mb)
    batch, data_mem = _profile_data(dataset_config, device, total_vram_mb)

    # Latent tokens
    latent_shape = model_config["args"]["latent_tokens"]
    n_tokens = reduce(operator.mul, latent_shape, 1)
    mem_before_tokens = torch.cuda.memory_allocated(device)
    latent_tokens = torch.randn(batch.num_graphs, n_tokens, 3, device=device, dtype=dtype)
    mem_after_tokens = torch.cuda.memory_allocated(device)
    print_memory_line("Latent tokens", mb(mem_after_tokens - mem_before_tokens), total_vram_mb)

    fwd_mem, bwd_mem, opt_step_mem, peak_final = _profile_forward_backward(
        model, batch, latent_tokens, optimizer, total_vram_mb
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print_memory_line("Model parameters", mb(param_mem), total_vram_mb)
    print_memory_line("Optimizer state", mb(opt_step_mem), total_vram_mb)
    print_memory_line("Data batch", mb(data_mem), total_vram_mb)
    print_memory_line("Activations + intermediates", mb(fwd_mem), total_vram_mb)
    print_memory_line("Gradients", mb(bwd_mem), total_vram_mb)
    accounted = mb(param_mem + opt_step_mem + data_mem + fwd_mem + bwd_mem)
    print(f"\n  Accounted total:  {accounted:>8.1f} MB")
    print(f"  Current allocated: {mb(torch.cuda.memory_allocated(device)):>8.1f} MB")
    print(f"  Current reserved:  {mb(torch.cuda.memory_reserved(device)):>8.1f} MB")
    print(f"  Peak allocated:    {mb(peak_final):>8.1f} MB")
    print(f"  Total VRAM:        {total_vram_mb:>8.1f} MB")
    print(f"  Headroom:          {total_vram_mb - mb(peak_final):>8.1f} MB")
    print()


if __name__ == "__main__":
    main()
