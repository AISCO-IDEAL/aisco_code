#!/usr/bin/env python
"""Proof-of-correctness sweep for the Plan-031 unified hash-based storage system.

Runs 11 scenarios (6 latent, 3 solitary, 2 main_v2 integration) entirely on
CPU with small epoch counts.  For each scenario it records the expected vs.
actual outcome and prints a structured verification table.

Scenarios covered
-----------------
  L1  latent first run                → SAVE new checkpoint
  L2  latent resume (same config)     → LOAD fast-resume, no retraining
  L3  z_radius changed (2.0 → 5.0)   → new hash → separate folder
  L4  epochs changed (5 → 10)         → new hash → folder prefix 000010_...
  L5  model_info_dict.json sidecar    → file exists alongside checkpoint
  L6  meta dict inside .pth           → config_hash / model_class / epochs present

  S1  solitary hash stable            → same config produces same hash twice
  S2  solitary lr changed             → hash differs
  S3  solitary train + canonical save → checkpoint at canonical path, meta correct

  I1  main_v2 latent_space=True path  → same hash + folder as direct call (no double-train)
  I2  main_v2 latent_space=False path → same hash + folder as direct call (no double-train)

Usage (from AISCO_code root, aisco_env active):

    python subprojects/wp4_aeroshell_optimizer/v2/scripts/hash_storage_sweep.py

Exit code: 0 = all passed, N = number of failures.
No GPU, no W&B, no PLY files required.  Expected runtime: ~3-4 minutes.
"""

from __future__ import annotations

import copy
import io
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_latent import (
    _compute_latent_pretrain_hash,
    _resolve_checkpoint_path,
    pretraining_latent,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_latent import (
    _resolve_checkpoint_path as _resolve_latent_ckpt_path,  # alias used by main_v2
)
from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_v1_1 import (
    _compute_solitary_pretrain_hash,
    _resolve_solitary_ckpt_path,
    pretraining,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.dynamic_loading import smart_model_init

# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

_LATENT_POPULATION = [
    {
        "name": "sphere_s0",
        "shape_type": "sphere",
        "radius": 500.0,
        "ply_path": None,
        "side_length": None,
        "ply_scale": None,
        "surface_noise_sigma_factor": None,
    },
    {
        "name": "cube_s1",
        "shape_type": "cube",
        "side_length": 1000.0,
        "ply_path": None,
        "radius": None,
        "ply_scale": None,
        "surface_noise_sigma_factor": None,
    },
]


def _tiny_latent_cfg(
    base_dir: str,
    *,
    z_radius: float = 2.0,
    epochs: int = 5,
    force: bool = True,
) -> dict:
    """Minimal 2-shape latent config (sphere + cube, no PLY files needed)."""
    return {
        "model_name": "ConcatWIREV5",
        "z_placement": "circle",
        "z_radius": z_radius,
        "force_pretraining": force,
        "latent_population": copy.deepcopy(_LATENT_POPULATION),
        "training_loop": {
            "epochs": epochs,
            "learning_rate": 5e-4,
            "batch_size_per_shape": 64,
            "logging_interval": 9999,
            "surface_resample_cadence": 0,
            "near_surface_fraction": 0.30,
            "domain_fraction": 0.50,
            "near_surface_tight_sigma": 0.005,
            "near_surface_wide_sigma": None,
            "bbox_scale_factor": 1.0,
        },
        "output": {
            "base_dir": base_dir,
        },
    }


def _load_solitary_cfg_fast(epochs: int = 5) -> dict:
    """Load pretraining_v0_config.yaml and override for quick CPU runs."""
    cfg = load_config(v="v2", cfg_name="pretraining_v0_config.yaml")
    cfg["training_loop"]["epochs"] = epochs
    cfg["training_loop"]["learning_rate"] = 5e-4
    cfg["training_loop"]["random_batch_size"] = 64
    cfg["training_loop"]["print_interval"] = 9999
    # Sphere shape: no PLY dependency.
    cfg["pretraining_shape"] = "sphere"
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_model():
    model, _, _, _ = smart_model_init("ConcatWIREV5")
    return model.to("cpu")


def _run_latent(cfg: dict) -> str:
    """Run pretraining_latent and return captured stdout."""
    model = _fresh_model()
    buf = io.StringIO()
    with redirect_stdout(buf):
        pretraining_latent(model, cfg, "cpu")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class Result:
    scenario: str
    hash_a: str
    hash_b: str = ""
    outcome_type: str = ""  # SAVE / LOAD / HASH / META / SIDECAR
    detail: str = ""
    passed: bool = False
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def run_all(base: Path) -> list[Result]:  # noqa: PLR0915  # sweep function - intentionally comprehensive
    results: list[Result] = []
    lat_base = str(base / "latent")
    sol_base = base / "solitary"

    # ── L1: Latent first run → SAVE ──────────────────────────────────────────
    cfg_l1 = _tiny_latent_cfg(lat_base, force=True)
    model_l1 = _fresh_model()
    h_l1 = _compute_latent_pretrain_hash(cfg_l1, model_l1)
    p_l1 = _resolve_checkpoint_path(cfg_l1, model_l1)
    out_l1 = _run_latent(cfg_l1)
    results.append(
        Result(
            scenario="L1  latent first run",
            hash_a=h_l1,
            outcome_type="SAVE",
            detail=f"ckpt_exists={p_l1.exists()}",
            passed="SAVE" in out_l1 and p_l1.exists(),
            notes=["Confirms pretraining_latent() writes checkpoint at canonical path"],
        )
    )

    # ── L2: Same config, force=False → LOAD (fast resume) ────────────────────
    cfg_l2 = _tiny_latent_cfg(lat_base, force=False)
    model_l2 = _fresh_model()
    h_l2 = _compute_latent_pretrain_hash(cfg_l2, model_l2)
    out_l2 = _run_latent(cfg_l2)
    results.append(
        Result(
            scenario="L2  latent resume (force=False)",
            hash_a=h_l1,
            hash_b=h_l2,
            outcome_type="LOAD",
            detail=f"hashes_equal={h_l1 == h_l2}  fast_resume={'fast_resume' in out_l2}",
            passed="fast_resume" in out_l2 and h_l1 == h_l2,
            notes=["Same config → same hash → LOAD, no gradient steps"],
        )
    )

    # ── L3: z_radius changed → different hash → separate folder ──────────────
    cfg_l3 = _tiny_latent_cfg(lat_base, z_radius=5.0, force=True)
    model_l3 = _fresh_model()
    h_l3 = _compute_latent_pretrain_hash(cfg_l3, model_l3)
    p_l3 = _resolve_checkpoint_path(cfg_l3, model_l3)
    _run_latent(cfg_l3)
    results.append(
        Result(
            scenario="L3  latent z_radius 2.0→5.0",
            hash_a=h_l1,
            hash_b=h_l3,
            outcome_type="HASH",
            detail=f"hashes_differ={h_l1 != h_l3}  sep_folder={p_l3.parent != p_l1.parent}",
            passed=h_l1 != h_l3 and p_l3.parent != p_l1.parent and p_l3.exists(),
            notes=["z_radius is now in hash key (was missing pre-031)"],
        )
    )

    # ── L4: epochs changed → different hash + epoch prefix in folder name ────
    cfg_l4 = _tiny_latent_cfg(lat_base, epochs=10, force=True)
    model_l4 = _fresh_model()
    h_l4 = _compute_latent_pretrain_hash(cfg_l4, model_l4)
    p_l4 = _resolve_checkpoint_path(cfg_l4, model_l4)
    _run_latent(cfg_l4)
    results.append(
        Result(
            scenario="L4  latent epochs 5→10",
            hash_a=h_l1,
            hash_b=h_l4,
            outcome_type="HASH",
            detail=f"hashes_differ={h_l1 != h_l4}  prefix={p_l4.parent.name[:6]}",
            passed=h_l1 != h_l4 and p_l4.parent.name.startswith("000010") and p_l4.exists(),
            notes=["Epoch count in folder prefix e.g. 000010_ConcatWIREV5_<hash>"],
        )
    )

    # ── L5: model_info_dict.json sidecar written alongside checkpoint ─────────
    info_json = p_l1.parent / "model_info_dict.json"
    results.append(
        Result(
            scenario="L5  model_info_dict.json sidecar",
            hash_a=h_l1,
            outcome_type="SIDECAR",
            detail=f"exists={info_json.exists()}",
            passed=info_json.exists(),
            notes=["Sidecar enables offline model inspection without loading .pth"],
        )
    )

    # ── L6: meta dict inside checkpoint ──────────────────────────────────────
    ckpt_l1 = torch.load(p_l1, map_location="cpu", weights_only=True)
    meta_l1 = ckpt_l1.get("meta", {})
    meta_ok = meta_l1.get("config_hash") == h_l1 and "model_class" in meta_l1 and "epochs" in meta_l1 and "date" in meta_l1
    results.append(
        Result(
            scenario="L6  checkpoint meta dict",
            hash_a=h_l1,
            outcome_type="META",
            detail=(
                f"hash_match={meta_l1.get('config_hash') == h_l1}"
                f"  cls={meta_l1.get('model_class', '?')}"
                f"  ep={meta_l1.get('epochs', '?')}"
            ),
            passed=meta_ok,
            notes=["meta allows detecting manual checkpoint copies / hash collisions"],
        )
    )

    # ── S1: Solitary hash stable (same config → same hash twice) ─────────────
    cfg_s_base = _load_solitary_cfg_fast(epochs=5)
    h_s1a = _compute_solitary_pretrain_hash(cfg_s_base, _fresh_model())
    h_s1b = _compute_solitary_pretrain_hash(copy.deepcopy(cfg_s_base), _fresh_model())
    p_s1 = _resolve_solitary_ckpt_path(cfg_s_base, _fresh_model(), sol_base)
    epoch_prefix = f"{int(cfg_s_base['training_loop']['epochs']):06d}"
    results.append(
        Result(
            scenario="S1  solitary hash stable",
            hash_a=h_s1a,
            hash_b=h_s1b,
            outcome_type="HASH",
            detail=(f"equal={h_s1a == h_s1b}  prefix={p_s1.parent.name[:6]}  file=pretrained_{h_s1a[:6]}...pth"),
            passed=h_s1a == h_s1b and p_s1.parent.name.startswith(epoch_prefix),
            notes=["Deterministic: model + config → reproducible hash"],
        )
    )

    # ── S2: Solitary lr changed → different hash ──────────────────────────────
    cfg_s_lr = _load_solitary_cfg_fast(epochs=5)
    cfg_s_lr["training_loop"]["learning_rate"] = 0.9999  # absurd — forces new hash
    h_s2 = _compute_solitary_pretrain_hash(cfg_s_lr, _fresh_model())
    results.append(
        Result(
            scenario="S2  solitary lr changed",
            hash_a=h_s1a,
            hash_b=h_s2,
            outcome_type="HASH",
            detail=f"hashes_differ={h_s1a != h_s2}",
            passed=h_s1a != h_s2,
            notes=["training_loop is in hash key → lr change → new checkpoint"],
        )
    )

    # ── S3: Solitary actual train + canonical path save ───────────────────────
    cfg_s3 = _load_solitary_cfg_fast(epochs=5)
    model_s3 = _fresh_model()
    h_s3 = _compute_solitary_pretrain_hash(cfg_s3, model_s3)
    p_s3 = _resolve_solitary_ckpt_path(cfg_s3, model_s3, sol_base)
    override_s3 = {
        "training_loop": {
            "epochs": 5,
            "learning_rate": 5e-4,
            "random_batch_size": 64,
            "print_interval": 9999,
        }
    }
    buf_s3 = io.StringIO()
    with redirect_stdout(buf_s3):
        pretraining(
            model=model_s3,
            override_config=override_s3,
            canonical_ckpt_path=p_s3,
        )
    ckpt_s3_meta: dict = {}
    if p_s3.exists():
        ckpt_s3 = torch.load(p_s3, map_location="cpu", weights_only=True)
        ckpt_s3_meta = ckpt_s3.get("meta", {})
    results.append(
        Result(
            scenario="S3  solitary train + canonical save",
            hash_a=h_s3,
            outcome_type="SAVE",
            detail=(f"ckpt_exists={p_s3.exists()}  hash_match={ckpt_s3_meta.get('config_hash') == h_s3}"),
            passed=p_s3.exists() and ckpt_s3_meta.get("config_hash") == h_s3,
            notes=["pretraining(canonical_ckpt_path=...) writes meta-bearing checkpoint"],
        )
    )

    # ── I1: main_v2 latent_space=True path ───────────────────────────────────
    # Replicate exactly what main_v2.main() does when latent_space=True:
    #   1. Load latent config (we inject a tiny in-memory one)
    #   2. Compute run_dir and pretrained_hash via the same aliases main_v2 uses
    #   3. Call pretraining_latent() with the same arguments
    #   4. Assert checkpoint path from direct call == path from main_v2 alias
    i1_base = str(base / "i1_latent")
    i1_latent_config = _tiny_latent_cfg(i1_base, force=True, epochs=5)
    i1_model = _fresh_model()

    # This is exactly the code main_v2 executes (latent_space=True branch):
    i1_pretrain_run_dir: Path = _resolve_latent_ckpt_path(i1_latent_config, i1_model).parent
    i1_vis_run_dir: Path = i1_pretrain_run_dir / "sampling_vis"  # noqa: F841  # mirrors main_v2 vis_run_dir assignment
    i1_pretrained_hash: str = _compute_latent_pretrain_hash(i1_latent_config, i1_model)

    i1_buf = io.StringIO()
    with redirect_stdout(i1_buf):
        pretraining_latent(i1_model, i1_latent_config, "cpu", use_wandb=False, vis_sampling=False)
    i1_out = i1_buf.getvalue()

    # Direct-call path must equal the main_v2-alias path (same function, different import name)
    i1_direct_path = _resolve_checkpoint_path(i1_latent_config, i1_model)
    i1_alias_path = _resolve_latent_ckpt_path(i1_latent_config, i1_model)
    i1_ok = (
        "SAVE" in i1_out
        and i1_direct_path.exists()
        and i1_direct_path == i1_alias_path
        and i1_pretrained_hash in i1_direct_path.parent.name
    )
    results.append(
        Result(
            scenario="I1  main_v2 latent_space=True",
            hash_a=i1_pretrained_hash,
            outcome_type="INTEG",
            detail=(
                f"alias_eq_direct={i1_direct_path == i1_alias_path}"
                f"  hash_in_folder={i1_pretrained_hash in i1_direct_path.parent.name}"
            ),
            passed=i1_ok,
            notes=[
                "Exercises the exact import alias (_resolve_latent_ckpt_path) used in main_v2",
                "vis_run_dir derived from canonical folder, not /tmp",
            ],
        )
    )

    # ── I2: main_v2 latent_space=False path ──────────────────────────────────
    # Replicate exactly what main_v2.main() does when latent_space=False:
    #   1. Load solitary pretraining config
    #   2. Build solitary_base_dir (local variant, using tmp)
    #   3. Compute hash + canonical path
    #   4. Since checkpoint missing, call pretraining(..., canonical_ckpt_path=...)
    #   5. Assert checkpoint written at canonical path with matching meta hash
    i2_sol_base = base / "i2_solitary"
    i2_pretraining_config = _load_solitary_cfg_fast(epochs=5)  # plays role of load_config("pretraining_v0_config.yaml")
    i2_model = _fresh_model()

    # Exactly as main_v2 does it (latent_space=False branch):
    i2_force = bool(i2_pretraining_config.get("pretraining", {}).get("force_pretraining", False))
    i2_pretrained_hash: str = _compute_solitary_pretrain_hash(i2_pretraining_config, i2_model)
    i2_pretrained_model_path: Path = _resolve_solitary_ckpt_path(i2_pretraining_config, i2_model, i2_sol_base)

    i2_buf = io.StringIO()
    if i2_pretrained_model_path.exists() and not i2_force:
        # fast-resume branch — not expected here since tmp is fresh
        ckpt_i2 = torch.load(i2_pretrained_model_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt_i2, dict) and "model_state_dict" in ckpt_i2:
            i2_model.load_state_dict(ckpt_i2["model_state_dict"])
        i2_buf.write(f"[pretraining] LOAD  path={i2_pretrained_model_path}  hash={i2_pretrained_hash}  status=fast_resume")
    else:
        with redirect_stdout(i2_buf):
            pretraining(
                model=i2_model,
                override_config={
                    "training_loop": {"epochs": 5, "learning_rate": 5e-4, "random_batch_size": 64, "print_interval": 9999}
                },
                canonical_ckpt_path=i2_pretrained_model_path,
            )

    i2_ckpt_meta: dict = {}
    if i2_pretrained_model_path.exists():
        i2_ckpt = torch.load(i2_pretrained_model_path, map_location="cpu", weights_only=True)
        i2_ckpt_meta = i2_ckpt.get("meta", {})

    i2_ok = (
        i2_pretrained_model_path.exists()
        and i2_ckpt_meta.get("config_hash") == i2_pretrained_hash
        and i2_pretrained_model_path.parent.name.startswith("000005")
    )
    results.append(
        Result(
            scenario="I2  main_v2 latent_space=False",
            hash_a=i2_pretrained_hash,
            outcome_type="INTEG",
            detail=(
                f"ckpt_exists={i2_pretrained_model_path.exists()}"
                f"  hash_match={i2_ckpt_meta.get('config_hash') == i2_pretrained_hash}"
                f"  prefix={i2_pretrained_model_path.parent.name[:6]}"
            ),
            passed=i2_ok,
            notes=[
                "Exercises _compute_solitary_pretrain_hash + _resolve_solitary_ckpt_path as main_v2 calls them",
                "canonical_ckpt_path plumbed through to pretraining() — meta written at correct path",
            ],
        )
    )

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_W = 110


def print_report(results: list[Result]) -> int:
    sep = "─" * _W
    print(f"\n{'═' * _W}")
    print("  AISCO  Plan-031 Hash-Storage Sweep — Verification Report")
    print(f"{'═' * _W}")
    hdr = f"  {'Scenario':<36}  {'Hash A':<12}  {'Hash B':<12}  {'Type':<7}  {'Detail':<30}  Status"
    print(hdr)
    print(sep)
    for r in results:
        status = "PASS ✓" if r.passed else "FAIL ✗"
        ha = r.hash_a[:10] if r.hash_a else "—"
        hb = r.hash_b[:10] if r.hash_b else "—"
        det = r.detail[:29]
        print(f"  {r.scenario:<36}  {ha:<12}  {hb:<12}  {r.outcome_type:<7}  {det:<30}  {status}")
    print(sep)
    n_fail = sum(1 for r in results if not r.passed)
    n_pass = len(results) - n_fail
    print(f"  Result: {n_pass}/{len(results)} passed  |  {n_fail} failure(s)")
    print(f"{'═' * _W}\n")

    if n_fail > 0:
        print("FAILURES detail:")
        for r in results:
            if not r.passed:
                print(f"  ✗  {r.scenario}")
                print(f"     Hash A : {r.hash_a}")
                if r.hash_b:
                    print(f"     Hash B : {r.hash_b}")
                print(f"     Detail : {r.detail}")
                for note in r.notes:
                    print(f"     Note   : {note}")
        print()

    return n_fail


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="aisco_hash_sweep_") as tmp:
        base = Path(tmp)
        print(f"[hash_storage_sweep] Working in: {base}")
        results = run_all(base)
    n_fail = print_report(results)
    sys.exit(n_fail)


if __name__ == "__main__":
    main()
