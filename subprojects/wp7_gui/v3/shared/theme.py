"""Theme and vis-config helpers for WP7 GUI v3.

This module provides a single read path for vis_config.yaml so all routes and
APIs consume the same parsed data with lightweight defaults.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import yaml

from subprojects.wp7_gui.v3.core.settings import SETTINGS


_DEFAULT_GUI_THEME: dict[str, Any] = {
    "fonts": {
        "ui": "Questrial, 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        "mono": "'Courier New', monospace",
    },
    "palette": {
        "background": "#23283E",
        "background_soft": "#1D2233",
        "background_deep": "#151A28",
        "panel": "#27314A",
        "panel_2": "#303D5D",
        "panel_top_start": "#35466D",
        "panel_top_end": "#2B3754",
        "border": "#2D3A55",
        "text": "#FFFFFF",
        "muted": "#D2E0F2",
        "title_1": "#FFFFFF",
        "title_2": "#FF504D",
        "surface_large": "#C33531",
        "text_box": "#D2E0F2",
        "assembly": "#F39C12",
        "aeroshell": "#5DADE2",
        "outer_bbox": "#23283E",
        "white": "#FFFFFF",
        "black": "#000000",
    },
    "status": {
        "success_text": "#2D5016",
        "success_bg": "#E8F5E9",
        "error_text": "#7A0E0E",
        "error_bg": "#FFEBEE",
        "warning_text": "#F8D9D9",
        "warning_bg": "rgba(200, 100, 0, 0.25)",
        "ok_text": "#B8F0B8",
        "ok_bg": "rgba(0, 120, 0, 0.25)",
    },
    "metrics": {
        "drag_border": "#66A3FF",
        "drag_text": "#A8D8F8",
        "solar_border": "#FFD966",
        "solar_text": "#FFD966",
        "speed_border": "#66CC99",
        "speed_text": "#66CC99",
        "unit_text": "#888888",
    },
    "rgba_effects": {
        "shadow": "0 14px 30px rgba(0, 0, 0, 0.26)",
        "hover_glow": "0 0 10px 2px rgba(200, 33, 33, 0.4)",
        "active_glow": "0 0 14px 2px rgba(200, 33, 33, 0.5)",
        "panel_overlay": "rgba(255, 255, 255, 0.06)",
        "workspace_top": "rgba(0, 0, 0, 0.22)",
    },
    "gradients": {
        "page_bg": "radial-gradient(circle at top left, #344267 0%, #1D2233 45%, #151A28 100%)",
        "top_bar": "linear-gradient(180deg, #35466D 0%, #2B3754 100%)",
        "intro_card": "linear-gradient(135deg, #1A2744 0%, #0D1120 100%)",
        "arcade_bg": "radial-gradient(ellipse at 38% 28%, #27366B 0%, #18233D 45%, #0D1120 100%)",
    },
    "threejs": {
        "mesh_background": "#F1F7F3",
        "mesh_fallback": "#99A3A4",
        "slider_accent": "#C33531",
        "control_text": "#000000",
    },
    "assembly_selection": {
        "border": "#374670",
        "panel": "#1D2233",
        "text": "#FFFFFF",
        "muted": "#8FA3B1",
        "blue": "#2563EB",
        "green": "#22C55E",
        "red": "#FF504D",
        "canvas_bg": "#111827",
        "canvas_grid": "rgba(255,255,255,0.10)",
        "canvas_axis": "rgba(255,255,255,0.4)",
        "canvas_axis_x": "#C33531",
        "canvas_axis_y": "#5DADE2",
        "canvas_label": "#FFFFFF",
        "canvas_poly_stroke": "#FFFFFF",
        "canvas_poly_fill": "rgba(255, 255, 255, 0.10)",
        "canvas_wheel": "#FFFFFF",
        "canvas_wheel_hover": "#FFFFFF",
        "canvas_seat": "#FFFFFF",
        "canvas_seat_hover": "#FFFFFF",
        "canvas_seat_stroke": "#FFFFFF",
    },
}


def _merge_dict(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        base_value = out.get(key)
        if isinstance(base_value, dict) and isinstance(value, Mapping):
            out[key] = _merge_dict(base_value, value)
        else:
            out[key] = value
    return out


def load_vis_config() -> dict[str, Any]:
    """Load the WP4 vis_config.yaml used by WP7 GUI integration."""
    try:
        cfg = yaml.safe_load((SETTINGS.wp4_configs_dir / "vis_config.yaml").read_text(encoding="utf-8"))
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def get_mesh_assembly_config(vis_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = vis_cfg if vis_cfg is not None else load_vis_config()
    visual_types = cfg.get("visual_types", {}) if isinstance(cfg, Mapping) else {}
    assembly_cfg = visual_types.get("mesh_assembly_v4", {}) if isinstance(visual_types, Mapping) else {}
    return dict(assembly_cfg) if isinstance(assembly_cfg, Mapping) else {}


def get_gui_theme(vis_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = vis_cfg if vis_cfg is not None else load_vis_config()
    gui_theme = cfg.get("gui_theme", {}) if isinstance(cfg, Mapping) else {}
    return _merge_dict(_DEFAULT_GUI_THEME, gui_theme if isinstance(gui_theme, Mapping) else {})


def theme_to_css_vars(gui_theme: Mapping[str, Any]) -> dict[str, str]:
    fonts = gui_theme.get("fonts", {})
    palette = gui_theme.get("palette", {})
    status = gui_theme.get("status", {})
    metrics = gui_theme.get("metrics", {})
    rgba_effects = gui_theme.get("rgba_effects", {})
    gradients = gui_theme.get("gradients", {})
    threejs = gui_theme.get("threejs", {})
    sel = gui_theme.get("assembly_selection", {})

    return {
        "--theme-font-ui": str(fonts.get("ui", _DEFAULT_GUI_THEME["fonts"]["ui"])),
        "--theme-font-mono": str(fonts.get("mono", _DEFAULT_GUI_THEME["fonts"]["mono"])),
        "--theme-bg": str(palette.get("background", "#23283E")),
        "--theme-bg-soft": str(palette.get("background_soft", "#1D2233")),
        "--theme-bg-deep": str(palette.get("background_deep", "#151A28")),
        "--theme-panel": str(palette.get("panel", "#27314A")),
        "--theme-panel-2": str(palette.get("panel_2", "#303D5D")),
        "--theme-panel-top-start": str(palette.get("panel_top_start", "#35466D")),
        "--theme-panel-top-end": str(palette.get("panel_top_end", "#2B3754")),
        "--theme-border": str(palette.get("border", "#2D3A55")),
        "--theme-text": str(palette.get("text", "#FFFFFF")),
        "--theme-muted": str(palette.get("muted", "#D2E0F2")),
        "--theme-title-1": str(palette.get("title_1", "#FFFFFF")),
        "--theme-title-2": str(palette.get("title_2", "#FF504D")),
        "--theme-surface-large": str(palette.get("surface_large", "#C33531")),
        "--theme-text-box": str(palette.get("text_box", "#D2E0F2")),
        "--theme-assembly": str(palette.get("assembly", "#F39C12")),
        "--theme-aeroshell": str(palette.get("aeroshell", "#5DADE2")),
        "--theme-outer-bbox": str(palette.get("outer_bbox", "#23283E")),
        "--theme-white": str(palette.get("white", "#FFFFFF")),
        "--theme-black": str(palette.get("black", "#000000")),
        "--theme-success-text": str(status.get("success_text", "#2D5016")),
        "--theme-success-bg": str(status.get("success_bg", "#E8F5E9")),
        "--theme-error-text": str(status.get("error_text", "#7A0E0E")),
        "--theme-error-bg": str(status.get("error_bg", "#FFEBEE")),
        "--theme-warning-text": str(status.get("warning_text", "#F8D9D9")),
        "--theme-warning-bg": str(status.get("warning_bg", "rgba(200, 100, 0, 0.25)")),
        "--theme-ok-text": str(status.get("ok_text", "#B8F0B8")),
        "--theme-ok-bg": str(status.get("ok_bg", "rgba(0, 120, 0, 0.25)")),
        "--theme-metric-drag-border": str(metrics.get("drag_border", "#66A3FF")),
        "--theme-metric-drag-text": str(metrics.get("drag_text", "#A8D8F8")),
        "--theme-metric-solar-border": str(metrics.get("solar_border", "#FFD966")),
        "--theme-metric-solar-text": str(metrics.get("solar_text", "#FFD966")),
        "--theme-metric-speed-border": str(metrics.get("speed_border", "#66CC99")),
        "--theme-metric-speed-text": str(metrics.get("speed_text", "#66CC99")),
        "--theme-metric-unit-text": str(metrics.get("unit_text", "#888888")),
        "--theme-shadow": str(rgba_effects.get("shadow", "0 14px 30px rgba(0, 0, 0, 0.26)")),
        "--theme-hover-glow": str(rgba_effects.get("hover_glow", "0 0 10px 2px rgba(200, 33, 33, 0.4)")),
        "--theme-active-glow": str(rgba_effects.get("active_glow", "0 0 14px 2px rgba(200, 33, 33, 0.5)")),
        "--theme-panel-overlay": str(rgba_effects.get("panel_overlay", "rgba(255, 255, 255, 0.06)")),
        "--theme-workspace-top": str(rgba_effects.get("workspace_top", "rgba(0, 0, 0, 0.22)")),
        "--theme-gradient-page-bg": str(
            gradients.get("page_bg", "radial-gradient(circle at top left, #344267 0%, #1D2233 45%, #151A28 100%)")
        ),
        "--theme-gradient-top-bar": str(gradients.get("top_bar", "linear-gradient(180deg, #35466D 0%, #2B3754 100%)")),
        "--theme-gradient-intro-card": str(gradients.get("intro_card", "linear-gradient(135deg, #1A2744 0%, #0D1120 100%)")),
        "--theme-gradient-arcade-bg": str(
            gradients.get("arcade_bg", "radial-gradient(ellipse at 38% 28%, #27366B 0%, #18233D 45%, #0D1120 100%)")
        ),
        "--theme-mesh-bg": str(threejs.get("mesh_background", "#F1F7F3")),
        "--theme-mesh-fallback": str(threejs.get("mesh_fallback", "#99A3A4")),
        "--theme-slider-accent": str(threejs.get("slider_accent", "#C33531")),
        "--theme-control-text": str(threejs.get("control_text", "#000000")),
        "--theme-sel-border": str(sel.get("border", "#D7DDE5")),
        "--theme-sel-panel": str(sel.get("panel", "#F7F9FC")),
        "--theme-sel-text": str(sel.get("text", "#111827")),
        "--theme-sel-muted": str(sel.get("muted", "#667085")),
        "--theme-sel-blue": str(sel.get("blue", "#2563EB")),
        "--theme-sel-green": str(sel.get("green", "#138A43")),
        "--theme-sel-red": str(sel.get("red", "#B42318")),
        "--theme-sel-canvas-grid": str(sel.get("canvas_grid", "#E5E7EB")),
        "--theme-sel-canvas-bg": str(sel.get("canvas_bg", "#111827")),
        "--theme-sel-canvas-axis": str(sel.get("canvas_axis", "rgba(255,255,255,0.4)")),
        "--theme-sel-canvas-axis-x": str(sel.get("canvas_axis_x", "#C33531")),
        "--theme-sel-canvas-axis-y": str(sel.get("canvas_axis_y", "#5DADE2")),
        "--theme-sel-canvas-label": str(sel.get("canvas_label", "#FFFFFF")),
        "--theme-sel-canvas-poly-stroke": str(sel.get("canvas_poly_stroke", "#FFFFFF")),
        "--theme-sel-canvas-poly-fill": str(sel.get("canvas_poly_fill", "rgba(255, 255, 255, 0.10)")),
        "--theme-sel-canvas-wheel": str(sel.get("canvas_wheel", "#FFFFFF")),
        "--theme-sel-canvas-wheel-hover": str(sel.get("canvas_wheel_hover", "#FFFFFF")),
        "--theme-sel-canvas-seat": str(sel.get("canvas_seat", "#FFFFFF")),
        "--theme-sel-canvas-seat-hover": str(sel.get("canvas_seat_hover", "#FFFFFF")),
        "--theme-sel-canvas-seat-stroke": str(sel.get("canvas_seat_stroke", "#FFFFFF")),
    }


def build_template_theme_context() -> dict[str, Any]:
    vis_cfg = load_vis_config()
    gui_theme = get_gui_theme(vis_cfg)
    css_vars = theme_to_css_vars(gui_theme)
    css_text = "\n".join(f"{key}: {value};" for key, value in css_vars.items())
    return {
        "gui_theme": gui_theme,
        "gui_theme_json": json.dumps(gui_theme),
        "theme_css_vars": css_text,
    }
