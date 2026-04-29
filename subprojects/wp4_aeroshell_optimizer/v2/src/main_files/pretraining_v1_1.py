import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from subprojects.wp2_assembler.version4.utils.reg_imp import get_outer_prism_bounds_mm
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.ground_truth_masks.simple_geoms import (
    cube_gt,
    ply_gt,
    sphere_gt,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsModelContract
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.debug_manager import (
    DebugManager,
    collect_cuda_memory,
    collect_optimizer_state,
    collect_param_stats,
)
from subprojects.wp5.unified_mesh_extraction import MeshExtractionConfig, MeshSDFExtractor

# --- Module constants (Plan 013 M2) ---
_DEFAULT_TIGHT_SIGMA_FACTOR = 0.005  # per config default; tight band sigma_factor * bbox_extent_mm


def dynamic_model_loading(config_name: str = "pretraining_v0_config.yaml") -> SupportsModelContract:
    """
    Dynamically loads the model architecture based on the configuration file.
    I.e. imports only from the relevant file, not all the models
    """
    config = load_config(cfg_name=config_name)  # still not great
    model_name = config["model"]  # dynamically import the model module
    model_file_name = config["model_filename"]  # file where the model is located
    module = importlib.import_module("subprojects.wp4_aeroshell_optimizer.v2.src.models." + model_file_name)
    ModelClass = getattr(module, model_name)

    # adjust to include config parameters here, stil WIP
    model = ModelClass()
    return model


def _apply_override_config(config: dict, override_config: dict) -> dict:
    for k, v in override_config.items():
        if k in config:
            if isinstance(config[k], dict) and isinstance(v, dict):
                config[k].update(v)
            else:
                config[k] = v
        elif "." in k:
            parts = k.split(".")
            if parts[0] in config and isinstance(config[parts[0]], dict):
                config[parts[0]][parts[1]] = v
    return config


def _load_ply_mesh(config: dict, device: str, center_mesh: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    ply_path = config.get("pretraining_ply_path")
    if not ply_path:
        raise ValueError("pretraining_shape='ply' requires 'pretraining_ply_path' in pretraining config")
    if not os.path.isfile(ply_path):
        fallback = config.get("beast_pretraining_ply_path", "")
        if fallback and os.path.isfile(fallback):
            print(f"[pretraining] '{ply_path}' not found, falling back to beast_pretraining_ply_path: '{fallback}'")
            ply_path = fallback
        else:
            raise ValueError(
                f"PLY file not found: '{ply_path}'. Set 'beast_pretraining_ply_path' in pretraining config for local runs."
            )
    ply_scale = float(config.get("pretraining_ply_scale", 1000.0))  # default scaling from m to mm
    raw_mesh = trimesh.load(ply_path)
    assert isinstance(raw_mesh, trimesh.Trimesh), f"Expected Trimesh, got {type(raw_mesh)}"
    verts_np = np.array(raw_mesh.vertices, dtype="float32") * ply_scale

    # Axis permutation: PLY_Z→AISCO_X, PLY_X→AISCO_Y, PLY_Y→AISCO_Z
    verts_np = verts_np[:, [2, 0, 1]]

    # X centering
    x_center = (float(verts_np[:, 0].max()) + float(verts_np[:, 0].min())) / 2.0
    verts_np[:, 0] -= x_center

    # Z grounding: Z_min = ground_clearance
    prism = get_outer_prism_bounds_mm()
    ground_clearance = abs(prism["z"][0])
    verts_np[:, 2] += ground_clearance - float(verts_np[:, 2].min())

    ply_verts_t = torch.tensor(verts_np, device=device)
    ply_faces_t = torch.tensor(np.array(raw_mesh.faces, dtype="int64"), device=device)
    extents = verts_np.max(axis=0) - verts_np.min(axis=0)
    print(
        f"[pretraining] Loaded PLY '{ply_path}' | scale={ply_scale} | "
        f"ext=({extents[0]:.0f},{extents[1]:.0f},{extents[2]:.0f})mm | "
        f"vertices={ply_verts_t.shape[0]} faces={ply_faces_t.shape[0]}"
    )
    if center_mesh:
        centroid = ply_verts_t.mean(dim=0)
        ply_verts_t -= centroid
        print(f"[pretraining] Centered PLY mesh by subtracting centroid: {centroid.cpu().numpy()}")
    return ply_verts_t, ply_faces_t


def _compute_gt_target(
    pretraining_shape: str,
    input_data: torch.Tensor,
    ply_verts_t: torch.Tensor | None,
    ply_faces_t: torch.Tensor | None,
) -> torch.Tensor:
    if pretraining_shape == "cube":
        return cube_gt(input_data, side_length=2000.0)
    if pretraining_shape == "ply":
        assert ply_verts_t is not None and ply_faces_t is not None
        return ply_gt(input_data, ply_verts_t, ply_faces_t)
    return sphere_gt(input_data)


def _compute_eikonal(model, input_data: torch.Tensor, device: str) -> torch.Tensor:
    input_m = (input_data / 1000.0).detach().requires_grad_(True)
    input_mm_for_model = input_m * 1000.0
    sdf_m = (model(input_mm_for_model).squeeze()) / 1000.0
    grad = torch.autograd.grad(
        outputs=sdf_m,
        inputs=input_m,
        grad_outputs=torch.ones_like(sdf_m),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad is not None:
        return ((grad.norm(dim=-1) - 1.0) ** 2).mean()
    print("WARNING: Gradient is still None. Check model connectivity.")
    return torch.tensor(0.0, device=device)


def _compute_eikonal_latent(
    model: SupportsModelContract,
    input_data: torch.Tensor,
    z_expanded: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Compute eikonal loss for a latent-concat model with explicit z.

    Like ``_compute_eikonal`` but passes ``z=z_expanded`` explicitly to
    ``model.forward()``.  Required for parallel-batch training with ConcatWIREV2
    where each row of the batch carries a different latent vector.

    Args:
        model: ConcatWIREV2 (or any model whose forward accepts ``z`` keyword).
        input_data: (N, 3) query points in mm.
        z_expanded: (N, z_dim) latent vectors aligned with the point batch.
        device: PyTorch device string.

    Returns:
        Scalar eikonal loss tensor (differentiable w.r.t. model parameters).
    """
    input_m = (input_data / 1000.0).detach().requires_grad_(True)
    input_mm_for_model = input_m * 1000.0
    sdf_m = (model(input_mm_for_model, z=z_expanded).squeeze()) / 1000.0
    grad = torch.autograd.grad(
        outputs=sdf_m,
        inputs=input_m,
        grad_outputs=torch.ones_like(sdf_m),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad is not None:
        return ((grad.norm(dim=-1) - 1.0) ** 2).mean()
    print("WARNING: Gradient is still None in _compute_eikonal_latent. Check model connectivity.")
    return torch.tensor(0.0, device=device)


def _sample_points_for_shape(  # noqa: PLR0915  # per-shape sampling is intrinsically sequential; decomposition would obscure control flow
    shape_type: str,
    shape_params: dict,
    batch_size: int,
    device: str | torch.device,
    *,  # keyword-only args follow
    near_surface_fraction: float = 0.50,
    domain_fraction: float = 0.50,
    near_surface_tight_sigma: float | None = None,
    near_surface_wide_sigma: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a surface-concentrated batch of 3D points and ground-truth SDF values.

    Sampling mixture (approximate):
      * sphere: 50 % domain (outer_prism cuboid), 30 % exact surface, 20 % near-surface
      * cube:   50 % domain (outer_prism cuboid), 30 % face-uniform, 20 % near-surface
      * ply:    50 % domain (outer_prism cuboid), 50 % trimesh surface + Gaussian jitter

    Args:
        shape_type: ``"sphere"``, ``"cube"``, or ``"ply"``.
        shape_params: Dict of shape parameters:
            - sphere: ``{"radius": float}``
            - cube:   ``{"side_length": float}``
            - ply:    ``{"ply_verts": Tensor, "ply_faces": Tensor, "bbox_extent": float}``
        batch_size: Total number of points to return.
        device: PyTorch device string or device object.

    Returns:
        Tuple ``(pts, gt_sdf)`` where both tensors have shape ``(batch_size,)`` /
        ``(batch_size, 3)`` respectively.  ``gt_sdf`` is signed (positive outside,
        negative inside) in mm for sphere/cube; for ply the sign follows mesh_sdf.
    """
    device = torch.device(device) if isinstance(device, str) else device

    if shape_type == "sphere":
        radius: float = float(shape_params["radius"])
        n_domain = int(batch_size * 0.50)
        n_surface = int(batch_size * 0.30)
        n_near = batch_size - n_domain - n_surface

        # Sphere center: X=0 (centered), Y=0 (centered), Z=100+radius (grounded)
        sphere_center = torch.tensor([0.0, 0.0, 100.0 + radius], dtype=torch.float32, device=device)

        # 50 % uniform domain in outer_prism cuboid (shared across all pretraining shapes)
        prism = get_outer_prism_bounds_mm()
        lo = torch.tensor([prism["x"][0], prism["y"][0], prism["z"][0]], dtype=torch.float32, device=device)
        hi = torch.tensor([prism["x"][1], prism["y"][1], prism["z"][1]], dtype=torch.float32, device=device)
        domain_pts = lo + torch.rand(n_domain, 3, device=device) * (hi - lo)
        domain_gt = sphere_gt(domain_pts, coordinates=sphere_center, radius=radius)

        # 30 % exact surface (center + direction * radius)
        rand_dirs = torch.randn(n_surface, 3, device=device)
        rand_dirs = rand_dirs / rand_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        surface_pts = sphere_center + rand_dirs * radius
        surface_gt = sphere_gt(surface_pts, coordinates=sphere_center, radius=radius)

        # 20 % near-surface band (|SDF| < 0.3 * radius, sampled from outer_prism)
        threshold = 0.3 * radius
        cand_pts = lo + torch.rand(n_near * 20, 3, device=device) * (hi - lo)
        cand_gt = sphere_gt(cand_pts, coordinates=sphere_center, radius=radius)
        mask = cand_gt.abs() < threshold
        near_pts_all = cand_pts[mask][:n_near]
        near_gt_all = cand_gt[mask][:n_near]

        # Pad deficit with exact-surface points
        deficit = n_near - near_pts_all.shape[0]
        if deficit > 0:
            pad_dirs = torch.randn(deficit, 3, device=device)
            pad_dirs = pad_dirs / pad_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            pad_pts = sphere_center + pad_dirs * radius
            near_pts_all = torch.cat([near_pts_all, pad_pts], dim=0)
            near_gt_all = torch.cat([near_gt_all, sphere_gt(pad_pts, coordinates=sphere_center, radius=radius)], dim=0)

        pts = torch.cat([domain_pts, surface_pts, near_pts_all], dim=0)
        gt = torch.cat([domain_gt, surface_gt, near_gt_all], dim=0)

    elif shape_type == "cube":
        side: float = float(shape_params["side_length"])
        half = side / 2.0
        n_domain = int(batch_size * 0.50)
        n_face = int(batch_size * 0.30)
        n_near = batch_size - n_domain - n_face

        # Cube center: X=0 (centered), Y=0 (centered), Z=100+half (grounded)
        cube_center = torch.tensor([0.0, 0.0, 100.0 + half], dtype=torch.float32, device=device)

        # 50 % uniform domain in outer_prism cuboid (shared across all pretraining shapes)
        prism = get_outer_prism_bounds_mm()
        lo = torch.tensor([prism["x"][0], prism["y"][0], prism["z"][0]], dtype=torch.float32, device=device)
        hi = torch.tensor([prism["x"][1], prism["y"][1], prism["z"][1]], dtype=torch.float32, device=device)
        domain_pts = lo + torch.rand(n_domain, 3, device=device) * (hi - lo)
        domain_gt = cube_gt(domain_pts, coordinates=cube_center, side_length=side)

        # 30 % uniform on 6 faces (centered around cube_center)
        face_idx = torch.randint(0, 6, (n_face,), device=device)
        face_pts = torch.rand(n_face, 3, device=device) * side - half
        # Snap each point to its designated face
        face_pts[face_idx == 0, 0] = half
        face_pts[face_idx == 1, 0] = -half
        face_pts[face_idx == 2, 1] = half
        face_pts[face_idx == 3, 1] = -half
        face_pts[face_idx == 4, 2] = half
        face_pts[face_idx == 5, 2] = -half
        # Translate face points to cube center
        face_pts = face_pts + cube_center
        face_gt = cube_gt(face_pts, coordinates=cube_center, side_length=side)

        # 20 % near-surface band (|SDF| < 0.15 * side, sampled from outer_prism)
        threshold = 0.15 * side
        cand_pts = lo + torch.rand(n_near * 20, 3, device=device) * (hi - lo)
        cand_gt = cube_gt(cand_pts, coordinates=cube_center, side_length=side)
        mask = cand_gt.abs() < threshold
        near_pts_all = cand_pts[mask][:n_near]
        near_gt_all = cand_gt[mask][:n_near]

        # Pad deficit with face points
        deficit = n_near - near_pts_all.shape[0]
        if deficit > 0:
            pad_face_idx = torch.randint(0, 6, (deficit,), device=device)
            pad_pts = torch.rand(deficit, 3, device=device) * side - half
            pad_pts[pad_face_idx == 0, 0] = half
            pad_pts[pad_face_idx == 1, 0] = -half
            pad_pts[pad_face_idx == 2, 1] = half
            pad_pts[pad_face_idx == 3, 1] = -half
            pad_pts[pad_face_idx == 4, 2] = half
            pad_pts[pad_face_idx == 5, 2] = -half
            # Translate pad points to cube center
            pad_pts = pad_pts + cube_center
            near_pts_all = torch.cat([near_pts_all, pad_pts], dim=0)
            near_gt_all = torch.cat([near_gt_all, cube_gt(pad_pts, coordinates=cube_center, side_length=side)], dim=0)

        pts = torch.cat([domain_pts, face_pts, near_pts_all], dim=0)
        gt = torch.cat([domain_gt, face_gt, near_gt_all], dim=0)

    elif shape_type == "ply":
        # PLY tensors live on CPU in the cache; move to device only for the GPU BVH query.
        ply_verts: torch.Tensor = shape_params["ply_verts"].to(device)
        ply_faces: torch.Tensor = shape_params["ply_faces"].to(device)
        bbox_extent: float = float(shape_params["bbox_extent"])

        n_domain = int(batch_size * domain_fraction)
        n_near = int(batch_size * near_surface_fraction)
        n_exact = batch_size - n_domain - n_near  # exact surface pts (zero jitter)

        # Validate that fractions don't exceed batch size
        if near_surface_fraction + domain_fraction > 1.0:
            raise ValueError(
                f"near_surface_fraction + domain_fraction must sum to ≤ 1.0, "
                f"got {near_surface_fraction} + {domain_fraction} = {near_surface_fraction + domain_fraction}"
            )

        # Domain — sample from the outer-prism cuboid
        prism = get_outer_prism_bounds_mm()
        lo = torch.tensor([prism["x"][0], prism["y"][0], prism["z"][0]], dtype=torch.float32, device=device)
        hi = torch.tensor([prism["x"][1], prism["y"][1], prism["z"][1]], dtype=torch.float32, device=device)
        domain_pts = lo + torch.rand(n_domain, 3, device=device) * (hi - lo)
        domain_gt = ply_gt(domain_pts, ply_verts, ply_faces)

        # Use CPU arrays for trimesh surface sampling (avoids moving trimesh to device)
        verts_np = shape_params["ply_verts"].numpy()  # already CPU
        faces_np = shape_params["ply_faces"].numpy()  # already CPU
        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)

        if near_surface_wide_sigma is None:
            # --- Single-band mode (backward-compatible default) ---
            sigma = (
                near_surface_tight_sigma * bbox_extent
                if near_surface_tight_sigma is not None
                else float(shape_params.get("surface_noise_sigma_factor", 0.05)) * bbox_extent
            )
            near_np, _ = trimesh.sample.sample_surface(mesh, n_near)
            near_pts = torch.tensor(near_np, dtype=torch.float32, device=device)
            near_pts = near_pts + torch.randn_like(near_pts) * sigma
            near_gt = ply_gt(near_pts, ply_verts, ply_faces)
            surface_pts_list = [near_pts]
            surface_gt_list = [near_gt]
        else:
            # --- Dual-band mode ---
            tight_sigma = (
                near_surface_tight_sigma * bbox_extent
                if near_surface_tight_sigma is not None
                else _DEFAULT_TIGHT_SIGMA_FACTOR * bbox_extent
            )
            wide_sigma = near_surface_wide_sigma * bbox_extent

            n_tight = n_near // 2
            n_wide = n_near - n_tight  # tight gets floor(n_near/2), wide absorbs remainder (asymmetric split favors broader band)

            tight_np, _ = trimesh.sample.sample_surface(mesh, n_tight)
            tight_pts = torch.tensor(tight_np, dtype=torch.float32, device=device)
            tight_pts = tight_pts + torch.randn_like(tight_pts) * tight_sigma
            tight_gt = ply_gt(tight_pts, ply_verts, ply_faces)

            wide_np, _ = trimesh.sample.sample_surface(mesh, n_wide)
            wide_pts = torch.tensor(wide_np, dtype=torch.float32, device=device)
            wide_pts = wide_pts + torch.randn_like(wide_pts) * wide_sigma
            wide_gt = ply_gt(wide_pts, ply_verts, ply_faces)

            surface_pts_list = [tight_pts, wide_pts]
            surface_gt_list = [tight_gt, wide_gt]

        # Exact surface pts (zero jitter) — always last in surface half
        if n_exact > 0:
            exact_np, _ = trimesh.sample.sample_surface(mesh, n_exact)
            exact_pts = torch.tensor(exact_np, dtype=torch.float32, device=device)
            exact_gt = ply_gt(exact_pts, ply_verts, ply_faces)
            surface_pts_list.append(exact_pts)
            surface_gt_list.append(exact_gt)

        pts = torch.cat([domain_pts, *surface_pts_list], dim=0)
        gt = torch.cat([domain_gt, *surface_gt_list], dim=0)

    else:
        raise ValueError(
            f"Unknown shape_type '{shape_type}'. Expected 'sphere', 'cube', or 'ply'. "
            "Check the shape_type field in pretraining_latent_config.yaml."
        )

    return pts, gt


def _make_mesh_extractor_config(device: str, mesh_res: int, outer_prism: dict[str, Any]) -> MeshExtractionConfig:
    """Build a MeshExtractionConfig using outer_prism bounds.

    Single source of truth for mesh extraction parameters in the pretraining
    pipeline.  Always uses the AISCO outer_prism (asymmetric bounding box),
    never symmetric ±mesh_box.

    Args:
        device: Compute device string (e.g. "cuda" or "cpu").
        mesh_res: Marching-cubes grid resolution.
        outer_prism: Dict from get_outer_prism_bounds_mm() with keys 'x', 'y', 'z'.
    """
    return MeshExtractionConfig(
        device=device,
        isovalue=0.0,
        bbox=[
            [float(outer_prism["x"][0]), float(outer_prism["x"][1])],
            [float(outer_prism["y"][0]), float(outer_prism["y"][1])],
            [float(outer_prism["z"][0]), float(outer_prism["z"][1])],
        ],
        base_resolution=mesh_res,
        subdivision_factor=1,
        max_refine_levels=0,
        batch_size=65536,
    )


def _extract_mesh(
    model: SupportsModelContract,
    pth_path: Path | str | None,
    device: str,
    mesh_res: int,
    outer_prism: dict[str, Any],
    *,
    return_mesh: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract a mesh from a trained model.

    When ``return_mesh=False`` (default), saves the mesh as a PLY file at
    ``pth_path.with_suffix('.ply')`` and returns ``None``.  ``pth_path`` must
    not be ``None`` in this mode.

    When ``return_mesh=True``, returns ``(verts_np, faces_np)`` as float32 /
    int64 numpy arrays and does **not** write to disk.  Exceptions are caught
    and ``None`` is returned on failure.  ``pth_path`` is ignored in this mode.

    Args:
        model: Trained SDF model implementing SupportsModelContract.
        pth_path: Path used to derive the output ``.ply`` file.  Ignored when
            ``return_mesh=True``; must not be ``None`` when ``return_mesh=False``.
        device: Compute device string.
        mesh_res: Marching-cubes grid resolution.
        outer_prism: Dict from get_outer_prism_bounds_mm() with keys 'x', 'y', 'z'.
        return_mesh: When ``True``, return numpy arrays instead of saving to disk.

    Returns:
        ``None`` when ``return_mesh=False`` (file saved to disk).
        ``(verts_np, faces_np)`` or ``None`` when ``return_mesh=True``.
    """
    try:
        cfg = _make_mesh_extractor_config(device, mesh_res, outer_prism)
        if not return_mesh:
            x_range = outer_prism["x"][1] - outer_prism["x"][0]
            y_range = outer_prism["y"][1] - outer_prism["y"][0]
            z_range = outer_prism["z"][1] - outer_prism["z"][0]
            print(
                f"[extract_mesh] Extracting mesh: res={mesh_res}, bounds=X({x_range:.0f})Y({y_range:.0f})Z({z_range:.0f})mm, device={device} ..."
            )
        verts, faces = MeshSDFExtractor(config=cfg).extract_mesh(model)
        verts_np = verts.detach().cpu().numpy().astype("float32")
        faces_np = faces.detach().cpu().numpy().astype("int64")
        if return_mesh:
            return verts_np, faces_np
        assert pth_path is not None, "_extract_mesh: pth_path must not be None when return_mesh=False"
        ply_path = Path(pth_path).with_suffix(".ply")
        trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False).export(ply_path)
        print(f"[extract_mesh] Saved mesh to {ply_path}")
    except Exception as exc:
        if return_mesh:
            print(f"[extract_mesh] WARNING: mesh extraction failed: {exc}")
            return None
        raise
    return None


# 2026 03 05, eschaerer:
# look i'm not sure if the changes i made also work on the HPC. Please revert them if they don't
def pretraining(
    model: SupportsModelContract | None = None,
    pretraining_config_name: str = "pretraining_v0_config.yaml",
    override_config: dict | None = None,
    debug_manager: DebugManager | None = None,
) -> SupportsModelContract:
    # Does pretraining, for now only on a sphere

    config = load_config(v="v2", cfg_name=pretraining_config_name)
    if override_config:
        config = _apply_override_config(config, override_config)

    # Try to use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        print("Careful: No model provided, dynamically loading model based on config...")
        model = dynamic_model_loading(config_name=pretraining_config_name)
    model = model.to(device)

    pretraining_shape = config.get("pretraining_shape", "sphere")

    # Pre-load PLY mesh once so mesh_sdf can be called efficiently each batch
    ply_verts_t = ply_faces_t = None
    if pretraining_shape == "ply":
        ply_verts_t, ply_faces_t = _load_ply_mesh(config, device)

    # Initialise a no-op DebugManager if none was provided
    dbg = debug_manager if debug_manager is not None else DebugManager(config={"GLOBAL_DEBUG": False})

    # Training loop
    learning_rate = config["training_loop"]["learning_rate"]
    num_epochs = config["training_loop"]["epochs"]
    batch_size = config["training_loop"]["random_batch_size"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ── Debug: one-time run config dump ──────────────────────
    dbg.log_run_config(
        model=model,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        optimizer_type="Adam",
        extra={
            "pretraining_shape": pretraining_shape,
            "batch_size": batch_size,
            "print_interval": config["training_loop"]["print_interval"],
            "eikonal_weight": 0.1,
        },
    )

    for i in range(num_epochs):
        # ── Debug: start-of-iteration stats tracking ─────────────
        dbg.begin_iteration(i)

        optimizer.zero_grad()

        # Constrained uniform sampling in a tighter box around the sphere (±2.5 m)
        input_data = (torch.rand(config["training_loop"]["random_batch_size"], 3, device=device) - 0.5) * 5000.0

        output = model(input_data)
        target = _compute_gt_target(pretraining_shape, input_data, ply_verts_t, ply_faces_t)
        eikonal = _compute_eikonal(model, input_data, device)

        loss = torch.nn.MSELoss()((target / 1000).squeeze(), (output / 1000).squeeze()) + 0.1 * eikonal

        loss.backward()

        # ── Debug: compute aggregate gradient norm ───────────────
        param_grad_norm = sum(p.grad.detach().pow(2).sum().item() for p in model.parameters() if p.grad is not None) ** 0.5

        loss_val = loss.item()
        mse_val = (loss - 0.1 * eikonal).item()
        eikonal_val = eikonal.item()
        lr = optimizer.param_groups[0]["lr"]

        # ── Debug: print raw values ──────────────────────────────
        dbg.log_values(
            i,
            inputs=input_data,
            outputs=output,
            loss=loss_val,
            loss_components={"mse": mse_val, "eikonal": eikonal_val, "eikonal_weighted": 0.1 * eikonal_val},
            grad_norm=param_grad_norm,
            lr=lr,
            param_stats=collect_param_stats(model) if dbg.print_raw_values and dbg.log_param_stats else None,
            optimizer_state=collect_optimizer_state(optimizer) if dbg.print_raw_values and dbg.log_optimizer_state else None,
            cuda_memory=collect_cuda_memory() if dbg.print_raw_values and dbg.log_cuda_memory else None,
            loss_weights={"mse": 1.0, "eikonal": 0.1},
            extra={
                "target_stats": f"min={target.min().item():.2f} max={target.max().item():.2f} mean={target.mean().item():.2f}",
                "output_stats": f"min={output.min().item():.2f} max={output.max().item():.2f} mean={output.mean().item():.2f}",
                "batch_size": input_data.shape[0],
                "device": device,
            },
        )

        # ── Debug: check warnings ────────────────────────────────
        dbg.check_warnings(
            i,
            loss=loss_val,
            grad_norm=param_grad_norm,
            loss_components={"mse": mse_val, "eikonal": eikonal_val},
        )

        # ── Debug: generate insights ─────────────────────────────
        dbg.check_insights(
            i,
            loss=loss_val,
            lr=lr,
            grad_norm=param_grad_norm,
        )

        optimizer.step()

        # ── Debug: end-of-iteration stats ────────────────────────
        dbg.end_iteration(
            i,
            loss=loss_val,
            grad_norm=param_grad_norm,
            lr=lr,
        )

        if i % config["training_loop"]["print_interval"] == 0:
            print(f"Epoch {i}, Loss: {loss_val:.4f}, Eikonal: {eikonal_val:.4f}")

    # ── Debug: cleanup ───────────────────────────────────────────
    dbg.stop()
    return model
