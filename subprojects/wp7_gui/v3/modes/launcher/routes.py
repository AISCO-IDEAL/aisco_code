"""Launcher mode router. Sub-pages map onto v2's launcher-side functionality:
config_overrides, assembly_selection (pre-run WP2 config), optimization (live
3-panel view), solar (focused live solar), testing (CLI runner). Real handlers
land in the next checkpoint."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from subprojects.wp7_gui.v3.core.settings import SETTINGS

router = APIRouter(prefix="/launcher", tags=["launcher"])
_templates = Jinja2Templates(directory=str(SETTINGS.templates_dir))

_SUBPAGES = {
    "config-overrides": ("Launcher — Config Overrides", "launcher/config_overrides.html"),
    "assembly-selection": ("Launcher — Assembly Selection", "launcher/assembly_selection.html"),
    "optimization": ("Launcher — Live Run", "launcher/optimization.html"),
    "solar": ("Launcher — Solar (Live)", "launcher/solar.html"),
    "testing": ("Launcher — CLI Testing", "launcher/testing.html"),
}


@router.get("")
def launcher_index(request: Request):
    return _templates.TemplateResponse(
        request,
        "launcher/index.html",
        {
            "active_mode": "launcher",
            "active_page": "config-overrides",
            "page_title": "Launcher",
        },
    )


@router.get("/{page_key}")
def launcher_subpage(request: Request, page_key: str):
    if page_key not in _SUBPAGES:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail={"message": f"Unknown launcher page '{page_key}'"})
    title, template = _SUBPAGES[page_key]
    return _templates.TemplateResponse(
        request,
        template,
        {
            "active_mode": "launcher",
            "active_page": page_key,
            "page_title": title,
        },
    )
