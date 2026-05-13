# WP7 GUI

## v3 (Current — v0.7.1)

Two arcade-style wizard pipelines served by FastAPI + Jinja2 (uvicorn).

### Pipelines

| Pipeline | Steps |
|---|---|
| **Live optimisation** | Home → Assembly Selection → Config Override → Run → Output → Home |
| **Completed runs (viewer)** | Home → Assembly Selection → Config Override (view-only) → Output → Home |

Pipeline context is carried via `?pipeline=live` or `?pipeline=viewer` URL query params throughout the wizard.

### Start the server

```bash
source aisco_env/bin/activate
python -m uvicorn subprojects.wp7_gui.v3.app_v3:app --host 0.0.0.0 --port 8000 --reload
```

### Key files

```text
subprojects/wp7_gui/v3/
├── app_v3.py                        # FastAPI entrypoint
├── modes/
│   ├── launcher/routes.py           # Wizard page routes (?pipeline= aware)
│   └── viewer/routes.py             # Viewer routes + /api/runs?cfg_session= filter
├── shared/
│   └── run_paths.py                 # run discovery, read_cfg_session()
└── frontend/templates/
    ├── introduction.html            # Pipeline selector (homescreen)
    ├── launcher/
    │   ├── assembly_selection.html  # Step 1 — geometry + sliders
    │   ├── config_overrides.html    # Step 2 — YAML config (live mode editable)
    │   ├── optimization.html        # Step 3 (live only) — run controls + log
    │   └── output.html              # Step 4 — GIFs, mesh viewer, artifacts
    └── viewer/
        └── index.html               # Standalone run browser (cfg_session auto-select)
```

### cfg_session correlation

When a run is launched via the Live pipeline, `wp4/main_v2.py` writes `gui_cfg_session.txt`
into the run's rendering directory. This file contains the path of the config override folder
used for that run. The Viewer pipeline uses the basename of this file to automatically
locate and display the correct completed run on the Output page.

---

## Minimal browser GUI (WSL -> Windows Firefox)

### Features
- Browser UI is served from `subprojects/wp7_gui/v1/templates/index.html`.
- Run logs are written to `subprojects/wp7_gui/v1/logs/`.
- Slow-GIF cache is written to `subprojects/wp7_gui/v1/cache/`.
- Start button launches the active WP4 optimizer `run_local.sh` via `subprojects.wp4_aeroshell_optimizer`.

### Run in WSL
From repo root:

```bash
source aisco_env/bin/activate
python subprojects/wp7_gui/v1/app_v1.py
```

### Open in Windows Firefox
In another WSL terminal:

```bash
/mnt/c/Program\ Files/Mozilla\ Firefox/firefox.exe http://localhost:8000
```

You can also copy `http://localhost:8000` into Firefox manually on Windows.

## Folder structure

```text
subprojects/wp7_gui/
├── __init__.py
├── README.md
├── archive/
└── v1/
    ├── app_v1.py
    ├── templates/
    │   ├── index.html
    │   ├── logo.png
    │   ├── ideal.png
    │   └── aCe.png
    ├── logs/
    ├── cache/
    └── gaot/
```
