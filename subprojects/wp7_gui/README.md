# WP7 GUI

## Status
We don't know yet wether we have to use Euler or get a local Computer, so the current GUIs focus on the client side representation. Leaving the backend communication open to decide on later.

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
