import random
from pathlib import Path

import torch
import trimesh
from torch import nn

from subprojects.wp3_solar_estimator.utils.wp3_vision_cone import (
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
