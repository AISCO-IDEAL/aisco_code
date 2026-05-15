"""TDD tests for Plan 041 Milestone 5: Training vis overhaul in DiversityTrainer.

Tests are written BEFORE implementation (Red phase).

Scope:
- T1: DiversityTrainer accepts vis_cadence_optimisation and Z_pretrained params
- T2: DiversityTrainer accepts vis_strategy param
- T3: DiversityTrainer accepts vis_all_losses param
- T4: DiversityTrainer accepts z_resample_mode param
- T5: _render() calls vis_mesh_assembly_v4 per slot when vis_manager is active
- T6: vis_strategy=static uses Z_pretrained for mesh extraction (not current Z)
- T7: vis_strategy=dynamic uses current Z for mesh extraction
- T8: vis_strategy=static with Z_pretrained=None raises ValueError
- T9: z_resample_mode=circle resamples onto pretrained circle (not random)
- T10: vis_cadence_optimisation controls render frequency (not rendering_freq)
- T11: main_v2.py passes vis_cadence_optimisation from vis_config to DiversityTrainer
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from subprojects.wp4_aeroshell_optimizer.v2.src.GINN.diversity_manager import DiversityTrainer
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.device_detection import get_config_base_path

_REPO = get_config_base_path() / "AISCO_code"
DIVERSITY_MANAGER_PATH = _REPO / "subprojects/wp4_aeroshell_optimizer/v2/src/GINN/diversity_manager.py"
MAIN_V2_PATH = _REPO / "subprojects/wp4_aeroshell_optimizer/v2/src/main_files/main_v2.py"


def _load_source(path: Path) -> str:
    return path.read_text()


class TestDiversityTrainerNewParams:
    """T1-T4: DiversityTrainer constructor accepts new M5 parameters."""

    def test_diversity_trainer_accepts_vis_cadence_optimisation(self) -> None:
        """DiversityTrainer.__init__ must accept vis_cadence_optimisation: int."""

        sig = inspect.signature(DiversityTrainer.__init__)
        assert "vis_cadence_optimisation" in sig.parameters, (
            "DiversityTrainer.__init__ missing vis_cadence_optimisation parameter. Add: vis_cadence_optimisation: int = 0"
        )

    def test_diversity_trainer_accepts_z_pretrained(self) -> None:
        """DiversityTrainer.__init__ must accept Z_pretrained: Tensor | None."""

        sig = inspect.signature(DiversityTrainer.__init__)
        assert "Z_pretrained" in sig.parameters, (
            "DiversityTrainer.__init__ missing Z_pretrained parameter. Add: Z_pretrained: torch.Tensor | None = None"
        )

    def test_diversity_trainer_accepts_vis_strategy(self) -> None:
        """DiversityTrainer.__init__ must accept vis_strategy: str."""

        sig = inspect.signature(DiversityTrainer.__init__)
        assert "vis_strategy" in sig.parameters, (
            "DiversityTrainer.__init__ missing vis_strategy parameter. Add: vis_strategy: str = 'dynamic'"
        )

    def test_diversity_trainer_accepts_vis_all_losses(self) -> None:
        """DiversityTrainer.__init__ must accept vis_all_losses: bool."""

        sig = inspect.signature(DiversityTrainer.__init__)
        assert "vis_all_losses" in sig.parameters, (
            "DiversityTrainer.__init__ missing vis_all_losses parameter. Add: vis_all_losses: bool = False"
        )

    def test_diversity_trainer_accepts_z_resample_mode(self) -> None:
        """DiversityTrainer.__init__ must accept z_resample_mode: str."""

        sig = inspect.signature(DiversityTrainer.__init__)
        assert "z_resample_mode" in sig.parameters, (
            "DiversityTrainer.__init__ missing z_resample_mode parameter. Add: z_resample_mode: str = 'circle'"
        )


class TestRenderCallsVisAssembly:
    """T5: _render() calls vis_mesh_assembly_v4 per slot."""

    def test_render_source_calls_vis_mesh_assembly_v4(self) -> None:
        """diversity_manager.py source must contain vis_mesh_assembly_v4 call in _render."""
        source = _load_source(DIVERSITY_MANAGER_PATH)
        assert "vis_mesh_assembly_v4" in source, (
            "diversity_manager.py does not call vis_manager.vis_mesh_assembly_v4(). "
            "Add call in _render() after vis_mesh_shape() per slot."
        )

    def test_render_source_has_vis_mesh_assembly_v4_in_render_method(self) -> None:
        """vis_mesh_assembly_v4 call must be inside the _render method body."""
        source = _load_source(DIVERSITY_MANAGER_PATH)
        # Find _render method and check vis_mesh_assembly_v4 is within it
        render_idx = source.find("def _render(")
        assembly_idx = source.find("vis_mesh_assembly_v4")
        assert render_idx != -1, "_render method not found"
        assert assembly_idx != -1, "vis_mesh_assembly_v4 not found"
        # vis_mesh_assembly_v4 call must appear after _render definition
        assert assembly_idx > render_idx, "vis_mesh_assembly_v4 appears before _render method — must be inside _render."


class TestVisStrategy:
    """T6-T8: vis_strategy static/dynamic behavior."""

    def test_source_has_vis_strategy_static_branch(self) -> None:
        """diversity_manager.py must contain vis_strategy=static branch logic."""
        source = _load_source(DIVERSITY_MANAGER_PATH)
        assert "vis_strategy" in source, (
            "diversity_manager.py has no vis_strategy reference. Add vis_strategy branching in _render()."
        )
        assert "static" in source, (
            "diversity_manager.py has no 'static' vis_strategy branch. "
            "Add: if vis_strategy=static use self._Z_pretrained for mesh extraction."
        )

    def test_source_has_z_pretrained_in_render(self) -> None:
        """_render must reference _Z_pretrained for static strategy."""
        source = _load_source(DIVERSITY_MANAGER_PATH)
        assert "_Z_pretrained" in source, (
            "diversity_manager.py missing _Z_pretrained attribute. "
            "Store Z_pretrained at init and use in _render for static vis_strategy."
        )

    def test_source_raises_when_static_strategy_but_no_z_pretrained(self) -> None:
        """Source must guard: if vis_strategy=static and _Z_pretrained is None, raise."""
        source = _load_source(DIVERSITY_MANAGER_PATH)
        # Check that there's a ValueError guard for static + no Z_pretrained
        assert "vis_strategy" in source and "ValueError" in source, (
            "diversity_manager.py must raise ValueError when vis_strategy='static' "
            "but Z_pretrained is None. Add guard in __init__ or _render."
        )


class TestZResampleMode:
    """T9: z_resample_mode=circle behavior."""

    def test_source_has_z_resample_mode_circle_logic(self) -> None:
        """diversity_manager.py must contain circle resample logic."""
        source = _load_source(DIVERSITY_MANAGER_PATH)
        assert "z_resample_mode" in source, (
            "diversity_manager.py missing z_resample_mode. Add z_resample_mode param and branching in _resample_population."
        )
        assert "circle" in source, (
            "diversity_manager.py missing 'circle' resample mode logic. "
            "Add: if z_resample_mode == 'circle' resample onto pretrained circle."
        )


class TestZResampleModeRename:
    """circle→on_circle rename + in_circle new mode."""

    def test_on_circle_is_valid_mode(self) -> None:
        """on_circle must be in _valid_resample_modes (renamed from circle)."""
        source = _load_source(DIVERSITY_MANAGER_PATH)
        assert '"on_circle"' in source or "'on_circle'" in source, (
            "diversity_manager.py missing 'on_circle' as a valid resample mode. "
            "Rename 'circle' to 'on_circle' in _valid_resample_modes."
        )

    def test_on_circle_is_default(self) -> None:
        """Default z_resample_mode parameter must be 'on_circle'."""
        sig = inspect.signature(DiversityTrainer.__init__)
        default = sig.parameters["z_resample_mode"].default
        assert default == "on_circle", (
            f"z_resample_mode default should be 'on_circle', got {default!r}. Rename the default from 'circle' to 'on_circle'."
        )

    def test_circle_bare_is_not_valid(self) -> None:
        """Bare 'circle' must no longer be a valid mode (replaced by on_circle)."""
        source = _load_source(DIVERSITY_MANAGER_PATH)
        # 'circle' must only appear as part of 'on_circle' or 'in_circle'
        bare_circle = re.search(r"['\"]{circle}['\"]", source)
        assert bare_circle is None, (
            "diversity_manager.py still has bare 'circle' as a valid mode string. Remove it — use 'on_circle' instead."
        )

    def test_in_circle_is_valid_mode(self) -> None:
        """in_circle must be in _valid_resample_modes."""
        source = _load_source(DIVERSITY_MANAGER_PATH)
        assert '"in_circle"' in source or "'in_circle'" in source, (
            "diversity_manager.py missing 'in_circle' as a valid resample mode. "
            "Add 'in_circle' to _valid_resample_modes and implement uniform disk sampling."
        )

    def test_in_circle_branch_uses_disk_sampling(self) -> None:
        """in_circle branch must use sqrt for uniform-area disk sampling."""
        source = _load_source(DIVERSITY_MANAGER_PATH)
        resample_idx = source.find("def _resample_population")
        assert resample_idx != -1
        resample_body = source[resample_idx:]
        assert "in_circle" in resample_body, "in_circle branch not found inside _resample_population."
        assert (
            ".sqrt()" in resample_body or "** 0.5" in resample_body or "**0.5" in resample_body or "math.sqrt" in resample_body
        ), (
            "_resample_population in_circle branch must use sqrt() for uniform disk area sampling "
            "(radius = r_max * sqrt(U) avoids density bias at center)."
        )

    def test_training_config_uses_on_circle(self) -> None:
        """training_config.yaml must use on_circle (not bare circle) for z_resample_mode."""
        cfg_path = _REPO / "subprojects/wp4_aeroshell_optimizer/v2/configs/training_config.yaml"
        content = cfg_path.read_text()
        assert "z_resample_mode: on_circle" in content, (
            "training_config.yaml should have z_resample_mode: on_circle (renamed from circle)."
        )


class TestVisCadenceOptimisation:
    """T10: vis_cadence_optimisation controls render frequency."""

    def test_source_uses_vis_cadence_optimisation_in_render_gate(self) -> None:
        """_render() must gate on vis_cadence_optimisation (not just rendering_freq)."""
        source = _load_source(DIVERSITY_MANAGER_PATH)
        assert "_vis_cadence_optimisation" in source or "vis_cadence_optimisation" in source, (
            "diversity_manager.py does not reference vis_cadence_optimisation. "
            "Add vis_cadence_optimisation param and use it in _render gate."
        )


class TestMainV2PassesVisCadence:
    """T11: main_v2.py passes vis_cadence_optimisation to DiversityTrainer."""

    def test_main_v2_passes_vis_cadence_optimisation(self) -> None:
        """main_v2.py DiversityTrainer construction must include vis_cadence_optimisation=."""
        source = _load_source(MAIN_V2_PATH)
        # Find DiversityTrainer( block and check for vis_cadence_optimisation
        trainer_idx = source.find("DiversityTrainer(")
        assert trainer_idx != -1, "DiversityTrainer( not found in main_v2.py"
        # Find closing paren of trainer construction block
        block = source[trainer_idx : trainer_idx + 2000]
        assert "vis_cadence_optimisation" in block, (
            "main_v2.py DiversityTrainer constructor does not pass vis_cadence_optimisation. "
            "Add: vis_cadence_optimisation=<value from vis_config latent_optimisation section>"
        )

    def test_main_v2_passes_z_pretrained_to_diversity_trainer(self) -> None:
        """main_v2.py must pass Z_pretrained to DiversityTrainer (not just initial_Z)."""
        source = _load_source(MAIN_V2_PATH)
        trainer_idx = source.find("DiversityTrainer(")
        block = source[trainer_idx : trainer_idx + 2000]
        assert "Z_pretrained" in block, (
            "main_v2.py DiversityTrainer does not pass Z_pretrained. "
            "Add: Z_pretrained=Z_pretrained to enable vis_strategy=static."
        )

    def test_main_v2_passes_vis_strategy_to_diversity_trainer(self) -> None:
        """main_v2.py must pass vis_strategy from vis_config to DiversityTrainer."""
        source = _load_source(MAIN_V2_PATH)
        trainer_idx = source.find("DiversityTrainer(")
        block = source[trainer_idx : trainer_idx + 2000]
        assert "vis_strategy" in block, (
            "main_v2.py DiversityTrainer does not pass vis_strategy. "
            "Add: vis_strategy=<from vis_config latent_optimisation section>"
        )


class TestPerSlotGradientVis:
    """Per-slot gradient vis when vis_all_losses=True."""

    def test_per_vector_backward_collects_per_slot_payloads_when_vis_all_losses(self) -> None:
        """_per_vector_backward must collect gradient payloads per slot when vis_all_losses is True.

        When vis_all_losses=True the per-slot gradient payloads must be harvested inside
        the population loop (after each compute() but before the next call resets the
        extractor buffers).  The collected list must be returned from _per_vector_backward
        so _render() can use it.
        """
        source = _load_source(DIVERSITY_MANAGER_PATH)
        pvb_idx = source.find("def _per_vector_backward(")
        assert pvb_idx != -1, "_per_vector_backward not found"
        # Find the end of _per_vector_backward by the next def at same indentation
        next_def_idx = source.find("\n    def ", pvb_idx + 1)
        pvb_body = source[pvb_idx:next_def_idx] if next_def_idx != -1 else source[pvb_idx:]

        assert "per_slot_grad_payloads" in pvb_body, (
            "_per_vector_backward must accumulate per-slot gradient payloads in a "
            "'per_slot_grad_payloads' list. This list is collected after each compute() "
            "call when vis_all_losses is True so that _render() can do per-slot vis."
        )
        assert "collect_vertex_grads_for_visualisation" in pvb_body, (
            "_per_vector_backward must call collect_vertex_grads_for_visualisation() "
            "inside the population loop to harvest per-slot gradient payloads."
        )

    def test_per_vector_backward_returns_per_slot_grad_payloads(self) -> None:
        """_per_vector_backward return type must include per-slot gradient payloads."""
        source = _load_source(DIVERSITY_MANAGER_PATH)
        pvb_idx = source.find("def _per_vector_backward(")
        next_def_idx = source.find("\n    def ", pvb_idx + 1)
        pvb_body = source[pvb_idx:next_def_idx] if next_def_idx != -1 else source[pvb_idx:]

        assert "per_slot_grad_payloads" in pvb_body, (
            "_per_vector_backward must return per_slot_grad_payloads as part of its tuple. Update the return statement."
        )
        # The return tuple in the last return statement should include per_slot_grad_payloads
        last_return_idx = pvb_body.rfind("return (")
        assert last_return_idx != -1, "return tuple not found in _per_vector_backward"
        return_block = pvb_body[last_return_idx:]
        assert "per_slot_grad_payloads" in return_block, (
            "per_slot_grad_payloads must be in the final return tuple of _per_vector_backward."
        )

    def test_render_uses_per_slot_payloads_for_gradient_vis(self) -> None:
        """_render() must call vis_mesh_gradient per slot using per-slot payloads when vis_all_losses."""
        source = _load_source(DIVERSITY_MANAGER_PATH)
        render_idx = source.find("def _render(")
        next_def_idx = source.find("\n    def ", render_idx + 1)
        render_body = source[render_idx:next_def_idx] if next_def_idx != -1 else source[render_idx:]

        assert "per_slot_grad_payloads" in render_body, (
            "_render() must accept and use per_slot_grad_payloads. "
            "When vis_all_losses is True, iterate over per_slot_grad_payloads for per-slot gradient vis."
        )
        assert "_vis_all_losses" in render_body, (
            "_render() must check self._vis_all_losses to decide whether to use "
            "per-slot gradient payloads or the existing single-slot fallback."
        )

    def test_render_sets_active_slot_for_each_gradient_vis_slot(self) -> None:
        """_render() must call set_active_slot(i) before each per-slot vis_mesh_gradient call."""
        source = _load_source(DIVERSITY_MANAGER_PATH)
        render_idx = source.find("def _render(")
        next_def_idx = source.find("\n    def ", render_idx + 1)
        render_body = source[render_idx:next_def_idx] if next_def_idx != -1 else source[render_idx:]

        # The render body must set_active_slot inside the per-slot gradient loop
        # We check that set_active_slot appears after the per_slot_grad_payloads reference
        per_slot_idx = render_body.find("per_slot_grad_payloads")
        set_slot_after = render_body.find("set_active_slot", per_slot_idx) if per_slot_idx != -1 else -1
        assert set_slot_after != -1, (
            "_render() must call set_active_slot(slot_idx) inside the per-slot gradient vis loop "
            "so each slot's PLY is written to the correct per-slot subdirectory."
        )
