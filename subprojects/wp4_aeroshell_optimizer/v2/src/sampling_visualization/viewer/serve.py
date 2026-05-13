#!/usr/bin/env python3
"""Launch the mesh viewer for a .glb file.

Usage:
    python serve.py path/to/file.glb
    python serve.py path/to/file.glb --port 8081
    python serve.py path/to/file.glb --no-browser   # headless / X11-forward
"""

import argparse
import http.server
import json
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

import yaml

_VIEWER_HTML = Path(__file__).parent / "index.html"
# vis_config.yaml lives at v2/configs/ — four parents up from this file.
_VIS_CONFIG_PATH = Path(__file__).parents[3] / "configs" / "vis_config.yaml"


class _Handler(http.server.BaseHTTPRequestHandler):
    glb_path: Path  # set on the class before the server starts

    def do_GET(self) -> None:
        # Strip query string before routing — self.path includes it (e.g. "/?model=...")
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            # Always inject ?model=/model.glb so the viewer loads the mesh
            # regardless of how the browser navigated here (VSCode port panel, etc.)
            html = _VIEWER_HTML.read_text()
            html = html.replace(
                "new URLSearchParams(window.location.search).get('model')",
                "new URLSearchParams(window.location.search).get('model') || '/model.glb'",
            )
            body = html.encode()
            self._respond(200, "text/html; charset=utf-8", body)
        elif path == "/model.glb":
            body = self.glb_path.read_bytes()
            self._respond(200, "model/gltf-binary", body)
        elif path == "/config.json":
            # Expose mesh_assembly_v4 color/alpha settings so the viewer can
            # override baked-in vertex colors without needing a re-export.
            try:
                vis_cfg = yaml.safe_load(_VIS_CONFIG_PATH.read_text())
                assembly_cfg = vis_cfg.get("visual_types", {}).get("mesh_assembly_v4", {})
            except Exception:
                assembly_cfg = {}
            body = json.dumps(assembly_cfg).encode()
            self._respond(200, "application/json", body)
        else:
            self.send_error(404)

    def _respond(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the mesh viewer for a .glb file.")
    parser.add_argument("glb", type=Path, help="Path to the .glb file to visualise.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true", help="Don't try to open a browser (use on headless nodes).")
    args = parser.parse_args()

    glb_path = args.glb.resolve()
    if not glb_path.exists():
        raise FileNotFoundError(f"GLB file not found: {glb_path}")

    _Handler.glb_path = glb_path
    server = http.server.HTTPServer(("localhost", args.port), _Handler)
    url = f"http://localhost:{args.port}/"

    print(f"Viewing: {glb_path}")
    print(f"URL:     {url}")
    print("Stop:    Ctrl-C\n")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    if not args.no_browser:
        webbrowser.open(url)

    try:
        thread.join()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
