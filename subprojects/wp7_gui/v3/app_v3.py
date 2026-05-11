"""WP7 GUI v3 entrypoint.

Thin by design: this file only wires the FastAPI app to the two mode routers,
the introduction page, and the static-asset mount. All business logic lives
under core/, shared/, modes/<mode>/, and shared/.

Run directly:   python app_v3.py           (from anywhere)
Run as module:  python -m subprojects.wp7_gui.v3.app_v3   (from repo root)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the repo root (AISCO_code/) is on sys.path so that the
# `subprojects.*` package imports resolve regardless of CWD or how this
# file is invoked.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402

from subprojects.wp7_gui.v3.core.settings import SETTINGS  # noqa: E402
from subprojects.wp7_gui.v3.modes.launcher.routes import router as launcher_router  # noqa: E402
from subprojects.wp7_gui.v3.modes.viewer.routes import router as viewer_router  # noqa: E402
from subprojects.wp7_gui.v3.shared.api_router import router as api_router  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


app = FastAPI(title="AISCO GUI v3")
templates = Jinja2Templates(directory=str(SETTINGS.templates_dir))


@app.get("/")
def root(request: Request):
    return templates.TemplateResponse(
        request,
        "introduction.html",
        {
            "active_mode": None,
            "active_page": "introduction",
            "page_title": "Welcome",
        },
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok", "wp4_active_version": SETTINGS.wp4_active_version}


app.include_router(viewer_router)
app.include_router(launcher_router)
app.include_router(api_router)

if SETTINGS.static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(SETTINGS.static_dir)), name="static")
else:
    LOGGER.warning("Static directory missing: %s", SETTINGS.static_dir)


def main() -> None:
    print("--- AISCO GUI v3 ---")
    print(f"Templates: {SETTINGS.templates_dir}")
    print(f"Statics:   {SETTINGS.static_dir}")
    print(f"WP4 root:  {SETTINGS.wp4_package_root} (active: {SETTINGS.wp4_active_version})")
    print(f"Local URL: http://localhost:{SETTINGS.port}")
    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port, log_level="info")


if __name__ == "__main__":
    main()
