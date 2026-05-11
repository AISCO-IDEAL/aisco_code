# WP7 GUI

## Status
We don't know yet wether we have to use Euler or get a local Computer, so the current GUIs focus on the client side representation. Leaving the backend communication open to decide on later.

## Minimal browser GUI (WSL -> Windows Firefox)

### Features
- Button 1 runs a dummy file (`dummy_logger.js`) and logs numbers `1` to `5` on screen.
- Button 2 displays a local image (`demo_image.png`) on screen.

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
subprojects/wp7_gui/archive/v0
└── README.md
└── app.py
└── templates/
    ├── index.html
    ├── dummy_logger.js
    ├── demo_image.png
    └── demo_image.svg
```
