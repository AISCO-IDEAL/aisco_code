"""Viewer mode router. Stubs only at this checkpoint — real run-list and
run-detail handlers land in the next checkpoint."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from subprojects.wp7_gui.v3.core.settings import SETTINGS

router = APIRouter(prefix="/viewer", tags=["viewer"])
_templates = Jinja2Templates(directory=str(SETTINGS.templates_dir))


@router.get("")
def viewer_index(request: Request):
    return _templates.TemplateResponse(
        request,
        "viewer/index.html",
        {
            "active_mode": "viewer",
            "active_page": "runs",
            "page_title": "Viewer — Runs",
        },
    )


@router.get("/pretraining")
def viewer_pretraining(request: Request):
    return _templates.TemplateResponse(
        request,
        "viewer/pretraining.html",
        {
            "active_mode": "viewer",
            "active_page": "pretraining",
            "page_title": "Viewer — Pretraining GIFs",
        },
    )
