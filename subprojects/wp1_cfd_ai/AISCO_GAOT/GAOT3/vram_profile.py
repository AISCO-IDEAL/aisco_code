#!/usr/bin/env python3
"""
VRAM profiling script for GAOT3D training.

Instruments every stage of a single training step:
  1) Baseline (model + optimizer state on GPU)
  2) Batch to device
  3) Forward pass sub-stages (encode, scalar cond, process, pool, decode scalar, decode field)
  4) Loss computation
  5) Backward pass (gradient accumulation)
  6) Gradient storage breakdown (per module)
  7) Optimizer step (Adam moment buffers)
  8) After zero_grad

Run:
    python subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/vram_profile.py \
        -c subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/config/newer_main_config.yaml

Requires: same environment as regular training.
"""

import argparse
import gc
import os
import sys
from typing import Any, cast

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch.nn.functional as F  # noqa: N812
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from src.model.gaot_3d import GAOT3D, masked_multi_head_loss
from src.trainer.training_orchestrator import TrainingOrchestrator


# ─── helpers ──────────────────────────────────────────────────────────
def mb(bytes_val: int | float) -> float:
    return bytes_val / 1e6


def gpu_snapshot(label: str, prev_alloc: float = 0.0) -> float:
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    delta = alloc - prev_alloc
    print(f"  [{label:.<55s}]  allocated={mb(alloc):>10.1f} MB  reserved={mb(reserved):>10.1f} MB  Δ={mb(delta):>+10.1f} MB")
    return alloc


def separator(title: str):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def grad_memory_by_module(model: torch.nn.Module) -> dict[str, float]:
    """Return gradient memory in MB grouped by top-level module."""
    grad_mem: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            top_module = name.split(".")[0]
            size_bytes = param.grad.nelement() * param.grad.element_size()
            grad_mem[top_module] = grad_mem.get(top_module, 0) + size_bytes
    return {k: mb(v) for k, v in sorted(grad_mem.items(), key=lambda x: -x[1])}


def param_memory_by_module(model: torch.nn.Module) -> dict[str, float]:
    """Return parameter memory in MB grouped by top-level module."""
    param_mem: dict[str, float] = {}
    for name, param in model.named_parameters():
        top_module = name.split(".")[0]
        size_bytes = param.nelement() * param.element_size()
        param_mem[top_module] = param_mem.get(top_module, 0) + size_bytes
    return {k: mb(v) for k, v in sorted(param_mem.items(), key=lambda x: -x[1])}


def optimizer_state_memory(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Return AdamW state memory breakdown in MB."""
    total_exp_avg = 0
    total_exp_avg_sq = 0
    total_other = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            for key, val in state.items():
                if isinstance(val, torch.Tensor):
                    nbytes = val.nelement() * val.element_size()
                    if key == "exp_avg":
                        total_exp_avg += nbytes
                    elif key == "exp_avg_sq":
                        total_exp_avg_sq += nbytes
                    else:
                        total_other += nbytes
    return {
        "exp_avg (1st moment)": mb(total_exp_avg),
        "exp_avg_sq (2nd moment)": mb(total_exp_avg_sq),
        "other": mb(total_other),
    }


# ─── Hooked forward pass ────────────────────────────────────────────
def profiled_forward(model: GAOT3D, batch: Batch, latent_tokens: torch.Tensor, device: torch.device):
    """
    Re-implements GAOT3D.forward() with VRAM snapshots between stages.
    Returns the same output dict as model.forward().
    """
    separator("FORWARD PASS (stage-by-stage)")
    prev = gpu_snapshot("before forward")

    num_graphs = batch.num_graphs

    # ── latent tokens ────────────────────────────────────────────────
    latent_tokens_batched = latent_tokens.to(device).repeat(num_graphs, 1)
    batch_idx_latent = torch.arange(num_graphs, device=device).repeat_interleave(model.num_latent_tokens)
    phys_pos_query_eff = batch.pos
    batch_idx_phys_query_eff = batch.batch
    prev = gpu_snapshot("latent token setup", prev)

    # ── encode (MAGNO) ───────────────────────────────────────────────
    rndata = model.encoder(
        batch=batch,
        latent_tokens_pos=latent_tokens_batched,
        latent_tokens_batch_idx=batch_idx_latent,
    )
    prev = gpu_snapshot("encoder (MAGNO GNO)", prev)

    # ── scalar conditioning ──────────────────────────────────────────
    if model.scalar_encoder is not None:
        rndata = model.scalar_encoder(rndata, batch)
        prev = gpu_snapshot("scalar_encoder", prev)

    # ── process (patchify + transformer + unpatchify) ────────────────
    rndata_proc = model.process(rndata=rndata, condition=None)
    prev = gpu_snapshot("process (patch+transformer+unpatch)", prev)

    # ── late scalar conditioning ─────────────────────────────────────
    if model.late_scalar_encoder is not None:
        rndata_proc = model.late_scalar_encoder(rndata_proc, batch)
        prev = gpu_snapshot("late_scalar_encoder", prev)

    # ── scalar path ──────────────────────────────────────────────────
    output = {}
    if model.decoder.head_names:
        pooled_features = model.attn_pool(rndata_proc)
        prev = gpu_snapshot("attention pooling", prev)
        output = model.decoder(pooled_features)
        prev = gpu_snapshot("scalar decoder (force/cd heads)", prev)

    # ── field path ───────────────────────────────────────────────────
    if model.field_output_size > 0:
        rndata_proc_flat = rndata_proc.reshape(-1, model.node_latent_size)
        field_pred = model.field_decoder(
            rndata_flat=rndata_proc_flat,
            phys_pos_query=phys_pos_query_eff,
            batch_idx_phys_query=batch_idx_phys_query_eff,
            latent_tokens_pos=latent_tokens_batched,
            latent_tokens_batch_idx=batch_idx_latent,
            batch=batch,
        )
        output["field"] = field_pred
        prev = gpu_snapshot("field decoder (MAGNO GNO out)", prev)

    return output


def main():  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    cmd_args = parser.parse_args()

    # ── load config ──────────────────────────────────────────────────
    with open(cmd_args.config) as f:
        cfg: dict[str, Any] = cast(dict[str, Any], OmegaConf.to_container(OmegaConf.load(f), resolve=True))
    cfg["setup"]["distributed"] = False
    cfg["setup"]["wandb"] = False
    cfg["setup"]["rank"] = 0
    cfg["setup"]["world_size"] = 1
    cfg["setup"]["local_rank"] = 0
    # Only 1 batch for profiling
    cfg["optimizer"]["args"]["epoch"] = 1
    device = torch.device(cfg["setup"]["device"])

    # ── autocast setup (respects config dtype) ───────────────────────
    dtype_str = cfg["setup"].get("dtype", "torch.float32")
    use_bf16 = dtype_str in ("bfloat16", "torch.bfloat16", "bf16")
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else torch.amp.autocast(device_type="cuda", enabled=False)
    )
    print(f"\n  Autocast: {'bfloat16' if use_bf16 else 'disabled (float32)'}")

    # ── clean GPU ────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    separator("PHASE 0: EMPTY GPU")
    gpu_snapshot("empty GPU baseline")

    # ── build trainer (loads data + model + optimizer) ───────────────
    separator("PHASE 1: MODEL + DATA LOADING")

    # We build the trainer normally but intercept the training loop
    cfg["datarow"] = {}
    trainer = TrainingOrchestrator(cfg)._build_trainer()
    model = trainer.model
    model.to(device)
    model.train()

    gpu_snapshot("model on GPU (parameters only)")
    print(f"\n  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("  Model param memory:")
    for mod, mem in param_memory_by_module(model).items():
        print(f"    {mod:.<45s} {mem:>8.1f} MB")

    # ── optimizer ────────────────────────────────────────────────────
    separator("PHASE 2: OPTIMIZER INITIALIZATION")
    opt_params = list(model.parameters())
    # Include uncertainty sigmas if present
    if hasattr(trainer, "log_sigma_field"):
        opt_params.append(trainer.log_sigma_field)
    if hasattr(trainer, "log_sigma_scalar"):
        opt_params.append(trainer.log_sigma_scalar)
    optimizer = torch.optim.AdamW(
        opt_params, lr=cfg["optimizer"]["args"]["lr"], weight_decay=cfg["optimizer"]["args"]["weight_decay"]
    )
    gpu_snapshot("optimizer created (no state yet)")

    # Force optimizer state creation by doing a dummy step
    print("\n  (AdamW state is lazily initialized — it will be created on first step)")

    # ── get one batch ────────────────────────────────────────────────
    separator("PHASE 3: LOAD BATCH TO GPU")
    batch_iter = iter(trainer.train_loader)
    batch = next(batch_iter)
    gpu_snapshot("batch loaded (CPU)")

    batch = batch.to(device)
    torch.cuda.synchronize()
    gpu_snapshot("batch on GPU")

    # Print batch stats
    print("\n  Batch info:")
    print(f"    num_graphs = {batch.num_graphs}")
    print(f"    total nodes (batch.pos) = {batch.pos.shape}")
    if hasattr(batch, "x") and batch.x is not None:
        print(f"    field features (batch.x) = {batch.x.shape}")
    if hasattr(batch, "edge_index") and batch.edge_index is not None:
        print(f"    edge_index = {batch.edge_index.shape}")
    for attr in ["enc_edge_index", "enc_edge_weight", "dec_edge_index", "dec_edge_weight"]:
        if hasattr(batch, attr) and getattr(batch, attr) is not None:
            t = getattr(batch, attr)
            print(f"    {attr} = {t.shape} ({mb(t.nelement() * t.element_size()):.1f} MB)")

    # ── forward pass (instrumented) ──────────────────────────────────
    separator("PHASE 4: FORWARD PASS")
    latent_tokens = trainer.latent_tokens if hasattr(trainer, "latent_tokens") else model.latent_tokens
    torch.cuda.reset_peak_memory_stats()
    before_fwd = torch.cuda.memory_allocated()

    with autocast_ctx:
        pred = profiled_forward(model, batch, latent_tokens, device)

    after_fwd = torch.cuda.memory_allocated()
    peak_fwd = torch.cuda.max_memory_allocated()
    print("\n  ── Forward summary ──")
    print(f"  allocated before forward: {mb(before_fwd):>10.1f} MB")
    print(f"  allocated after forward:  {mb(after_fwd):>10.1f} MB")
    print(f"  peak during forward:      {mb(peak_fwd):>10.1f} MB")
    print(f"  activations held (after-before): {mb(after_fwd - before_fwd):>10.1f} MB")
    print(f"  peak activations (peak-before):  {mb(peak_fwd - before_fwd):>10.1f} MB")

    # ── loss computation ─────────────────────────────────────────────
    separator("PHASE 5: LOSS COMPUTATION")
    gpu_snapshot("before loss")

    # Replicate what train_step does for loss
    with autocast_ctx:
        if "field" in pred and hasattr(batch, "x") and batch.x is not None:
            ft = batch.x.to(device)
            valid = ~torch.isnan(ft)
            field_loss = F.mse_loss(pred["field"][valid], ft[valid]) if valid.any() else torch.tensor(0.0, device=device)

            targets = {}
            if hasattr(batch, "force") and batch.force is not None:
                targets["force"] = batch.force
            if hasattr(batch, "cd") and batch.cd is not None:
                targets["cd"] = batch.cd
            if targets:
                scalar_loss = masked_multi_head_loss(pred, targets)
                loss = scalar_loss + cfg["dataset"]["field_loss_weight"] * field_loss
            else:
                loss = field_loss
        else:
            targets = {"force": batch.force, "cd": batch.cd}
            loss = masked_multi_head_loss(pred, targets)

    gpu_snapshot("after loss computation")
    print(f"  loss = {loss.item():.6f}")

    # ── backward pass ────────────────────────────────────────────────
    separator("PHASE 6: BACKWARD PASS (gradient computation)")
    torch.cuda.reset_peak_memory_stats()
    before_bwd = torch.cuda.memory_allocated()
    gpu_snapshot("before backward")

    loss.backward()

    after_bwd = torch.cuda.memory_allocated()
    peak_bwd = torch.cuda.max_memory_allocated()
    gpu_snapshot("after backward")

    print("\n  ── Backward summary ──")
    print(f"  allocated before backward: {mb(before_bwd):>10.1f} MB")
    print(f"  allocated after backward:  {mb(after_bwd):>10.1f} MB")
    print(f"  peak during backward:      {mb(peak_bwd):>10.1f} MB")
    print(f"  gradient memory (after-before): {mb(after_bwd - before_bwd):>10.1f} MB")
    print(f"  peak backward activations:      {mb(peak_bwd - before_bwd):>10.1f} MB")

    # ── gradient breakdown ───────────────────────────────────────────
    separator("PHASE 7: GRADIENT MEMORY BREAKDOWN (by module)")
    grad_mem = grad_memory_by_module(model)
    total_grad = sum(grad_mem.values())
    print(f"  {'Module':<45s} {'Grad MB':>10s} {'% of total':>10s}")
    print(f"  {'─' * 65}")
    for mod, mem in grad_mem.items():
        pct = 100.0 * mem / total_grad if total_grad > 0 else 0
        print(f"  {mod:<45s} {mem:>10.1f} {pct:>9.1f}%")
    print(f"  {'─' * 65}")
    print(f"  {'TOTAL GRADIENTS':<45s} {total_grad:>10.1f} MB")

    # Also break down by deeper module path (2 levels)
    print("\n  ── Gradient breakdown (2 levels deep) ──")
    grad_mem_deep: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            parts = name.split(".")
            key = ".".join(parts[: min(3, len(parts))])
            size_bytes = param.grad.nelement() * param.grad.element_size()
            grad_mem_deep[key] = grad_mem_deep.get(key, 0) + size_bytes
    grad_mem_deep = {k: mb(v) for k, v in sorted(grad_mem_deep.items(), key=lambda x: -x[1])}
    for mod, mem in grad_mem_deep.items():
        pct = 100.0 * mem / total_grad if total_grad > 0 else 0
        print(f"  {mod:<45s} {mem:>10.1f} {pct:>9.1f}%")

    # ── optimizer step ───────────────────────────────────────────────
    separator("PHASE 8: OPTIMIZER STEP (creates Adam state)")
    gpu_snapshot("before optimizer.step()")

    optimizer.step()
    gpu_snapshot("after optimizer.step()")

    # Now Adam state exists
    opt_mem = optimizer_state_memory(optimizer)
    print("\n  ── AdamW state memory ──")
    total_opt = sum(opt_mem.values())
    for key, mem in opt_mem.items():
        print(f"  {key:<45s} {mem:>10.1f} MB")
    print(f"  {'TOTAL OPTIMIZER STATE':<45s} {total_opt:>10.1f} MB")

    # ── zero_grad ────────────────────────────────────────────────────
    separator("PHASE 9: ZERO_GRAD")
    gpu_snapshot("before zero_grad")
    optimizer.zero_grad(set_to_none=True)
    gpu_snapshot("after zero_grad (set_to_none=True)")

    # ── final summary ────────────────────────────────────────────────
    separator("FINAL SUMMARY")

    param_total = sum(param_memory_by_module(model).values())
    print(f"\n  {'Component':<45s} {'MB':>10s}")
    print(f"  {'═' * 55}")
    print(f"  {'Model parameters':<45s} {param_total:>10.1f}")
    print(f"  {'Gradients (same size as params)':<45s} {total_grad:>10.1f}")
    print(f"  {'AdamW state (2 x params: moments)':<45s} {total_opt:>10.1f}")
    print(f"  {'──────────────────────────────────────────────────────':s}")
    static_total = param_total + total_grad + total_opt
    print(f"  {'Static total (params+grads+optimizer)':<45s} {static_total:>10.1f}")
    print()

    torch.cuda.synchronize()
    final_alloc = torch.cuda.memory_allocated()
    print(f"  {'Batch data on GPU (estimated)':<45s} {mb(final_alloc) - static_total:>10.1f}")
    print(f"  {'Current GPU allocated':<45s} {mb(final_alloc):>10.1f}")
    print(f"  {'Peak GPU allocated (during backward)':<45s} {mb(peak_bwd):>10.1f}")
    peak_activations = mb(peak_bwd) - static_total
    print(f"  {'Peak activations (peak - static)':<45s} {peak_activations:>10.1f}")
    print()
    print("  Memory budget breakdown (at peak):")
    print(f"    Parameters ........... {100 * param_total / mb(peak_bwd):>5.1f}%")
    print(f"    Gradients ............ {100 * total_grad / mb(peak_bwd):>5.1f}%")
    print(f"    Optimizer state ...... {100 * total_opt / mb(peak_bwd):>5.1f}%")
    print(f"    Activations+batch .... {100 * peak_activations / mb(peak_bwd):>5.1f}%")
    print()


if __name__ == "__main__":
    main()
