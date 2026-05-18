import random
from pathlib import Path

import torch
import trimesh
from torch import nn

from subprojects.wp3_solar_estimator.rollout_losses.SR_SolarLoss import solar_loss_sr
from subprojects.wp3_solar_estimator.utils.wp3_vision_cone import (
    get_vision_cone_face_mask,
    get_vision_cone_mesh,
    mesh_intersection,
    render_intersection,
    render_vis_cone,
)
from subprojects.wp5.unified_mesh_extraction import MeshExtractionConfig, MeshSDFExtractor


def _box_mesh(center: tuple[float, float, float], size: float) -> tuple[torch.Tensor, torch.Tensor]:
    x, y, z = center
    h = size / 2
    verts = torch.tensor(
        [
            [x - h, y - h, z - h],
            [x + h, y - h, z - h],
            [x + h, y + h, z - h],
            [x - h, y + h, z - h],
            [x - h, y - h, z + h],
            [x + h, y - h, z + h],
            [x + h, y + h, z + h],
            [x - h, y + h, z + h],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=torch.long,
    )
    return verts, faces


class _AeroLikeSDF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("center", torch.tensor([350.0, 0.0, 900.0], dtype=torch.float32))
        self.register_buffer("radii", torch.tensor([1850.0, 780.0, 520.0], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = (x - self.center.to(x.device)) / self.radii.to(x.device)
        return (torch.linalg.norm(q, dim=-1, keepdim=True) - 1.0) * 520.0


def _extract_wp4_generated_aero_mesh() -> tuple[torch.Tensor, torch.Tensor]:
    extractor = MeshSDFExtractor(
        MeshExtractionConfig(
            device="cpu",
            isovalue=0.0,
            base_resolution=56,
            subdivision_factor=2,
            max_refine_levels=0,
            batch_size=65536,
            bbox=[[-3000.0, 3000.0], [-3000.0, 3000.0], [-500.0, 2200.0]],
            target_num_vertices=None,
        )
    )
    return extractor.extract_mesh(_AeroLikeSDF())


def test_mesh_intersection_inner_box_selected() -> None:
    # Large box contains small box — all faces of the small box should be returned.
    outer_verts, outer_faces = _box_mesh(center=(0.0, 0.0, 0.0), size=4.0)
    inner_verts, inner_faces = _box_mesh(center=(0.0, 0.0, 0.0), size=1.0)

    verts_i, faces_i = mesh_intersection(outer_verts, outer_faces, inner_verts, inner_faces)

    assert verts_i.shape[1] == 3
    assert faces_i.shape[1] == 3
    assert len(verts_i) == len(inner_verts)
    assert len(faces_i) == len(inner_faces)


def test_mesh_intersection_non_overlapping_boxes_empty() -> None:
    verts1, faces1 = _box_mesh(center=(0.0, 0.0, 0.0), size=1.0)
    verts2, faces2 = _box_mesh(center=(10.0, 10.0, 10.0), size=1.0)

    verts_i, faces_i = mesh_intersection(verts1, faces1, verts2, faces2)

    assert len(verts_i) == 0
    assert len(faces_i) == 0


def test_vision_cone_mesh_cpu_and_glb_export(tmp_path: Path) -> None:
    verts, faces = get_vision_cone_mesh(0, device="cpu")

    assert verts.device.type == "cpu"
    assert verts.shape[1] == 3
    assert faces.shape[1] == 3
    assert len(verts) > 0
    assert len(faces) > 0
    assert torch.isfinite(verts).all()
    assert int(faces.min()) >= 0
    assert int(faces.max()) < len(verts)

    out_path = render_vis_cone(0, tmp_path / "vision_cone.glb", device="cpu")

    assert out_path.exists()
    scene = trimesh.load(out_path, force="scene")
    assert "vision_cone" in scene.geometry


def test_render_intersection_box_box(tmp_path: Path) -> None:
    outer_verts, outer_faces = _box_mesh(center=(0.0, 0.0, 0.0), size=4.0)
    inner_verts, inner_faces = _box_mesh(center=(0.0, 0.0, 0.0), size=1.0)

    sel_path, not_sel_path = render_intersection(outer_verts, outer_faces, inner_verts, inner_faces, tmp_path, "test")

    assert sel_path.exists()
    assert not_sel_path.exists()
    sel_mesh = trimesh.load(sel_path, force="mesh")
    not_sel_mesh = trimesh.load(not_sel_path, force="mesh")
    assert len(sel_mesh.faces) > 0  # inner box faces selected
    assert len(not_sel_mesh.faces) == 0  # nothing outside when inner is fully inside outer


def test_vision_cone_no_overlap_with_far_box() -> None:
    cone_verts, cone_faces = get_vision_cone_mesh(0, device="cpu")
    # Place a small box 100 km away — guaranteed no overlap.
    far_box_verts, far_box_faces = _box_mesh((1_000_000.0, 1_000_000.0, 1_000_000.0), size=1.0)

    verts_i, faces_i = mesh_intersection(cone_verts, cone_faces, far_box_verts, far_box_faces)

    assert len(verts_i) == 0
    assert len(faces_i) == 0


_WP4_PLY_DIR = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM_federico/wp4_runs/9f0d75f6d9/43_solar-smoke/aero/cp_field")
_WP4_RUNS_OUT = Path("local_storage_dir/wp3_vision_cone/wp4_runs")


def test_vision_cone_intersects_15_wp4_real_meshes() -> None:
    ply_files = sorted(_WP4_PLY_DIR.glob("epoch_*.ply"))
    assert len(ply_files) >= 15, f"Expected at least 15 PLY files, found {len(ply_files)}"

    random.seed(42)
    selected_plys = random.sample(ply_files, 15)

    cone_verts, cone_faces = get_vision_cone_mesh(0, device="cpu")

    for ply_path in selected_plys:
        epoch = ply_path.stem  # e.g. "epoch_00030"
        out_dir = _WP4_RUNS_OUT / epoch
        out_dir.mkdir(parents=True, exist_ok=True)

        mesh = trimesh.load(str(ply_path), force="mesh")
        aero_verts = torch.as_tensor(mesh.vertices, dtype=torch.float32)
        aero_faces = torch.as_tensor(mesh.faces, dtype=torch.long)

        sel_path, not_sel_path = render_intersection(cone_verts, cone_faces, aero_verts, aero_faces, out_dir, epoch)

        assert sel_path.exists(), f"{epoch}: selected GLB not saved"
        assert not_sel_path.exists(), f"{epoch}: not_selected GLB not saved"

        sel_mesh = trimesh.load(str(sel_path), force="mesh")
        not_sel_mesh = trimesh.load(str(not_sel_path), force="mesh")
        total = len(aero_faces)
        assert len(sel_mesh.faces) + len(not_sel_mesh.faces) == total, (
            f"{epoch}: face count mismatch — {len(sel_mesh.faces)} + {len(not_sel_mesh.faces)} != {total}"
        )


def test_solar_loss_sr_cone_mask_reduces_power() -> None:
    """Faces masked by cone_face_mask should contribute ~0 power — masked power < unmasked power."""
    # Flat grid of upward-facing triangles, all perfectly aligned with sun_dir=[0,0,1]
    N = 10
    verts_list = []
    faces_list = []
    face_idx = 0
    for i in range(N):
        for j in range(N):
            x, y = float(i) * 100.0, float(j) * 100.0
            verts_list += [[x, y, 0.0], [x + 100.0, y, 0.0], [x, y + 100.0, 0.0]]
            faces_list.append([face_idx, face_idx + 1, face_idx + 2])
            face_idx += 3
    vertices = torch.tensor(verts_list, dtype=torch.float32)
    faces = torch.tensor(faces_list, dtype=torch.long)
    sun_dir = torch.tensor([0.0, 0.0, 1.0])

    power_full, _ = solar_loss_sr(vertices, faces, sun_dir, area_budget_mm2=1e12)

    all_masked = torch.ones(len(faces), dtype=torch.bool)
    power_masked, _ = solar_loss_sr(vertices, faces, sun_dir, area_budget_mm2=1e12, cone_face_mask=all_masked)

    half_masked = torch.zeros(len(faces), dtype=torch.bool)
    half_masked[: len(faces) // 2] = True
    power_half, _ = solar_loss_sr(vertices, faces, sun_dir, area_budget_mm2=1e12, cone_face_mask=half_masked)

    assert power_masked.item() < 1e-3, f"All-masked power should be ~0, got {power_masked.item()}"
    assert power_half.item() < power_full.item(), "Half-masked power should be less than unmasked"
    assert power_half.item() > power_masked.item(), "Half-masked power should be greater than all-masked"


_SCRATCH_DIR = Path("/cluster/scratch/fsanche/wp3_vision_cone_solar")


def test_solar_loss_sr_cone_mask_with_real_vision_cone() -> None:
    """End-to-end: cone face mask from the real vision cone geometry reduces solar power on the aero mesh.

    Saves selected/excluded face GLBs to /cluster/scratch/fsanche/wp3_vision_cone_solar/ for visual inspection.
    """
    import numpy as np  # noqa: PLC0415

    wp4_verts, wp4_faces = _extract_wp4_generated_aero_mesh()
    cone_verts, cone_faces = get_vision_cone_mesh(0, device="cpu")
    sun_dir = torch.tensor([-1.0, 0.0, 1.0])

    cone_mask = get_vision_cone_face_mask(cone_verts, cone_faces, wp4_verts, wp4_faces)

    power_full, diag_full = solar_loss_sr(wp4_verts, wp4_faces, sun_dir, area_budget_mm2=6_000_000.0)
    power_masked, diag_masked = solar_loss_sr(wp4_verts, wp4_faces, sun_dir, area_budget_mm2=6_000_000.0, cone_face_mask=cone_mask)

    n_masked = int(cone_mask.sum().item())
    print(f"  Cone mask: {n_masked}/{len(wp4_faces)} faces excluded")
    print(f"  Power full:   {diag_full['total_power_watts']:.1f} W")
    print(f"  Power masked: {diag_masked['total_power_watts']:.1f} W")

    # Save GLBs for visual inspection
    _SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    verts_np = wp4_verts.numpy()
    faces_np = wp4_faces.numpy()
    mask_np = cone_mask.numpy()

    def _save_glb(path: Path, face_indices: "np.ndarray", rgb: tuple[int, int, int]) -> None:
        import trimesh  # noqa: PLC0415

        used_idx = np.unique(face_indices)
        remap = np.zeros(len(verts_np), dtype=np.int64)
        remap[used_idx] = np.arange(len(used_idx))
        mesh = trimesh.Trimesh(vertices=verts_np[used_idx], faces=remap[face_indices], process=False)
        mesh.visual.face_colors = np.tile(np.array([*rgb, 255], dtype=np.uint8), (len(face_indices), 1))
        mesh.export(str(path))
        print(f"  Saved {len(face_indices)} faces → {path}")

    _save_glb(_SCRATCH_DIR / "cone_excluded_faces.glb", faces_np[mask_np], (230, 50, 50))
    _save_glb(_SCRATCH_DIR / "solar_selectable_faces.glb", faces_np[~mask_np], (50, 200, 80))

    assert n_masked > 0, "Expected at least some faces inside the vision cone"
    assert power_masked.item() <= power_full.item() + 1e-3, "Masked power should not exceed unmasked power"


def test_vision_cone_intersects_wp4_generated_aero_mesh() -> None:
    wp4_verts, wp4_faces = _extract_wp4_generated_aero_mesh()
    cone_verts, cone_faces = get_vision_cone_mesh(0, device="cpu")

    verts_i, faces_i = mesh_intersection(cone_verts, cone_faces, wp4_verts, wp4_faces)

    assert wp4_verts.shape[0] > 0
    assert wp4_faces.shape[0] > 0
    assert verts_i.device.type == "cpu"
    assert verts_i.shape[1] == 3
    assert faces_i.shape[1] == 3
    assert len(verts_i) > 0
    assert len(faces_i) > 0
    assert torch.isfinite(verts_i).all()
    assert int(faces_i.min()) >= 0
    assert int(faces_i.max()) < len(verts_i)
