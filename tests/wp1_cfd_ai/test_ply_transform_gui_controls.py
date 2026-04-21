"""Regression tests for PLY transform GUI dataset viewer controls."""

import re

from subprojects.wp1_cfd_ai.utilities import ply_transform_gui


def test_dataset_autorange_uses_configurable_percentile_control() -> None:
    """Dataset autorange must expose a configurable percentile instead of fixed p1/p99."""
    html = ply_transform_gui._load_html()

    assert 'id="ds-autorange-pct"' in html
    assert "function dsResolveAutoRangePercentiles()" in html
    assert "const [lowerPct, upperPct] = dsResolveAutoRangePercentiles();" in html


def test_scalar_bar_is_vertical_in_top_right() -> None:
    """The scalar bar must be vertical and anchored in the top-right corner."""
    html = ply_transform_gui._load_html()

    scalar_bar_match = re.search(
        r"#scalar-bar\s*\{[^}]*top:\s*52px;[^}]*right:\s*16px;[^}]*bottom:\s*auto;[^}]*left:\s*auto;",
        html,
        re.DOTALL,
    )
    gradient_match = re.search(
        r"\.sbar-gradient\s*\{[^}]*width:\s*12px;[^}]*height:\s*170px;[^}]*background:\s*linear-gradient\(to top,",
        html,
        re.DOTALL,
    )
    labels_match = re.search(
        r"\.sbar-labels\s*\{[^}]*flex-direction:\s*column;",
        html,
        re.DOTALL,
    )

    assert scalar_bar_match is not None
    assert gradient_match is not None
    assert labels_match is not None
