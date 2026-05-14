"""Tests for save_population_meshes (Plan 037 / M2).

Covers:
  - save_population_meshes calls set_latent_vector for each slot
  - save_population_meshes saves exactly N PLY files in the expected directory
  - save_population_meshes skips gracefully when population_z is empty

Run:
    pytest tests/wp4_aeroshell_optimizer/v2/test_save_population_meshes.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

_STORAGE_UTILS = "subprojects.wp4_aeroshell_optimizer.v2.src.utils.storage_utils"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_model(z_dim: int = 2) -> MagicMock:
    """Return a mock that satisfies SupportsModelContract + SupportsLatentConcat."""
    model = MagicMock()
    model.get_model_info_dict.return_value = {"arch": "test", "z_dim": z_dim}
    model.set_latent_vector = MagicMock()
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_save_population_meshes_import() -> None:
    """save_population_meshes must be importable from storage_utils."""
    from subprojects.wp4_aeroshell_optimizer.v2.src.utils.storage_utils import save_population_meshes  # noqa: PLC0415, F401


def test_save_population_meshes_calls_set_latent_for_each_slot(tmp_path: Path) -> None:
    """set_latent_vector must be called exactly once per population slot."""
    from subprojects.wp4_aeroshell_optimizer.v2.src.utils.storage_utils import save_population_meshes  # noqa: PLC0415

    n_slots = 3
    z_dim = 2
    population_z = torch.randn(n_slots, z_dim)
    model = _make_fake_model(z_dim)

    fake_verts = torch.zeros(8, 3)
    fake_faces = torch.zeros(6, 3, dtype=torch.long)

    fake_mesh_cfg = MagicMock(base_resolution=128, device="cpu")

    with (
        patch(f"{_STORAGE_UTILS}.load_mesh_extraction_config", return_value=fake_mesh_cfg),
        patch(f"{_STORAGE_UTILS}.replace", side_effect=lambda obj, **_: obj),
        patch(f"{_STORAGE_UTILS}.MeshSDFExtractor") as mock_extractor_cls,
        patch(f"{_STORAGE_UTILS}.get_outer_prism_bounds_mm") as mock_prism,
        patch(f"{_STORAGE_UTILS}.kaolin_to_PLY"),
        patch(f"{_STORAGE_UTILS}.dict_hashing", return_value="abc123"),
        patch(f"{_STORAGE_UTILS}.copy") as mock_copy,
    ):
        mock_prism.return_value = {"x": [0, 1], "y": [0, 1], "z": [0, 1]}
        mock_extractor = MagicMock()
        mock_extractor.extract_mesh.return_value = (fake_verts, fake_faces)
        mock_extractor_cls.return_value = mock_extractor
        mock_copy.deepcopy.return_value = model

        save_population_meshes(
            model=model,
            population_z=population_z,
            storage_path=str(tmp_path),
            wandb_run="test-run-1",
            epoch=100,
            resolution=128,
            verbose=False,
            device="cpu",
        )

    assert model.set_latent_vector.call_count == n_slots


def test_save_population_meshes_saves_n_ply_files(tmp_path: Path) -> None:
    """kaolin_to_PLY must be called exactly once per slot with a unique filename."""
    from subprojects.wp4_aeroshell_optimizer.v2.src.utils.storage_utils import save_population_meshes  # noqa: PLC0415

    n_slots = 3
    z_dim = 2
    population_z = torch.zeros(n_slots, z_dim)
    model = _make_fake_model(z_dim)

    fake_verts = torch.zeros(8, 3)
    fake_faces = torch.zeros(6, 3, dtype=torch.long)

    ply_calls: list[str] = []

    def capture_ply(*_: object, **kwargs: object) -> None:
        ply_calls.append(str(kwargs.get("filename", "")))

    fake_mesh_cfg = MagicMock(base_resolution=128, device="cpu")

    with (
        patch(f"{_STORAGE_UTILS}.load_mesh_extraction_config", return_value=fake_mesh_cfg),
        patch(f"{_STORAGE_UTILS}.replace", side_effect=lambda obj, **_: obj),
        patch(f"{_STORAGE_UTILS}.MeshSDFExtractor") as mock_extractor_cls,
        patch(f"{_STORAGE_UTILS}.get_outer_prism_bounds_mm") as mock_prism,
        patch(f"{_STORAGE_UTILS}.kaolin_to_PLY", side_effect=capture_ply),
        patch(f"{_STORAGE_UTILS}.dict_hashing", return_value="abc123"),
        patch(f"{_STORAGE_UTILS}.copy") as mock_copy,
    ):
        mock_prism.return_value = {"x": [0, 1], "y": [0, 1], "z": [0, 1]}
        mock_extractor = MagicMock()
        mock_extractor.extract_mesh.return_value = (fake_verts, fake_faces)
        mock_extractor_cls.return_value = mock_extractor
        mock_copy.deepcopy.return_value = model

        save_population_meshes(
            model=model,
            population_z=population_z,
            storage_path=str(tmp_path),
            wandb_run="test-run-1",
            epoch=100,
            resolution=128,
            verbose=False,
            device="cpu",
        )

    assert len(ply_calls) == n_slots
    # Each filename must be unique
    assert len(set(ply_calls)) == n_slots
    # Each filename must contain the slot index
    for i, fname in enumerate(ply_calls):
        assert f"slot_{i}" in fname


def test_save_population_meshes_empty_population(tmp_path: Path) -> None:
    """Empty population_z must not call set_latent_vector or kaolin_to_PLY."""
    from subprojects.wp4_aeroshell_optimizer.v2.src.utils.storage_utils import save_population_meshes  # noqa: PLC0415

    model = _make_fake_model()
    population_z = torch.zeros(0, 2)

    fake_mesh_cfg = MagicMock(base_resolution=128, device="cpu")

    with (
        patch(f"{_STORAGE_UTILS}.load_mesh_extraction_config", return_value=fake_mesh_cfg),
        patch(f"{_STORAGE_UTILS}.replace", side_effect=lambda obj, **_: obj),
        patch(f"{_STORAGE_UTILS}.MeshSDFExtractor"),
        patch(f"{_STORAGE_UTILS}.get_outer_prism_bounds_mm") as mock_prism,
        patch(f"{_STORAGE_UTILS}.kaolin_to_PLY") as mock_ply,
        patch(f"{_STORAGE_UTILS}.dict_hashing", return_value="abc123"),
        patch(f"{_STORAGE_UTILS}.copy"),
    ):
        mock_prism.return_value = {"x": [0, 1], "y": [0, 1], "z": [0, 1]}

        save_population_meshes(
            model=model,
            population_z=population_z,
            storage_path=str(tmp_path),
            wandb_run="test-run-1",
            epoch=0,
            resolution=128,
            verbose=False,
            device="cpu",
        )

    model.set_latent_vector.assert_not_called()
    mock_ply.assert_not_called()
