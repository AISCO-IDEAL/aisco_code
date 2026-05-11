import logging
import threading
import time
import webbrowser
from collections.abc import Callable

from flask import Flask, jsonify, render_template_string, request

_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Assembly Parameters</title>
  <style>
    body { font-family: monospace; padding: 20px; max-width: 600px; }
    h3 { margin: 18px 0 4px; border-bottom: 1px solid #ccc; }
    .row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
    .row label { width: 120px; }
    .row input[type=range] { flex: 1; }
    .row span { width: 60px; text-align: right; }
    #status { margin-top: 16px; color: gray; font-size: 0.9em; }
    .wheel-buttons { margin-top: 12px; display: flex; gap: 8px; }
    button { font-family: monospace; padding: 4px 12px; cursor: pointer; }
  </style>
</head>
<body>
  <h2>Assembly Parameters</h2>
  <div id="controls"></div>
  <div class="wheel-buttons">
    <button onclick="addWheel()">+ Add wheel</button>
    <button onclick="removeWheel()">- Remove wheel</button>
    <button onclick="addToMap()">Add to assembly map</button>
  </div>
  <div id="status">Loading...</div>

  <script>
    let params = {};
    let pendingUpdate = null;

    async function loadParams() {
      const r = await fetch('/params');
      params = await r.json();
      render();
      document.getElementById('status').textContent = 'Ready.';
    }

    function render() {
      const div = document.getElementById('controls');
      div.innerHTML = '';
      for (const [section, vals] of Object.entries(params)) {
        const h = document.createElement('h3');
        h.textContent = section;
        div.appendChild(h);
        for (const [axis, val] of Object.entries(vals)) {
          const lo = (axis === 'z') ? 0 : (axis === 'x' ? -3000 : -2000);
          const hi = (axis === 'z') ? 600 : (axis === 'x' ? 3000 : 2000);
          const row = document.createElement('div');
          row.className = 'row';
          row.innerHTML = `
            <label>${section}.${axis}</label>
            <input type="range" min="${lo}" max="${hi}" step="1" value="${val}"
              data-key="${section}" data-axis="${axis}"
              oninput="onMove(this)" onchange="onRelease(this)">
            <span id="lbl_${section}_${axis}">${Math.round(val)}</span>
          `;
          div.appendChild(row);
        }
      }
    }

    function onMove(el) {
      const v = parseFloat(el.value);
      params[el.dataset.key][el.dataset.axis] = v;
      document.getElementById('lbl_' + el.dataset.key + '_' + el.dataset.axis).textContent = Math.round(v);
    }

    function onRelease(el) {
      clearTimeout(pendingUpdate);
      document.getElementById('status').textContent = 'Rebuilding...';
      pendingUpdate = setTimeout(async () => {
        const r = await fetch('/update', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(params)
        });
        const res = await r.json();
        document.getElementById('status').textContent = res.status;
      }, 50);
    }

    async function addWheel() {
      const r = await fetch('/add_wheel', { method: 'POST' });
      const res = await r.json();
      params = res.params;
      render();
      document.getElementById('status').textContent = res.status;
    }

    async function removeWheel() {
      const r = await fetch('/remove_wheel', { method: 'POST' });
      const res = await r.json();
      if (res.params) {
        params = res.params;
        render();
      }
      document.getElementById('status').textContent = res.status;
    }

    async function addToMap() {
      document.getElementById('status').textContent = 'Checking assembly...';
      const r = await fetch('/add_to_map', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params)
      });
      const res = await r.json();
      document.getElementById('status').textContent = res.status;
    }

    loadParams();
  </script>
</body>
</html>
"""


def run(
    read_params: Callable,
    write_params: Callable,
    call_assembly_from_params: Callable,
    add_current_gui_assembly_to_map: Callable | None = None,
    port: int = 5123,
) -> None:
    from ocp_vscode import show  # noqa: PLC0415

    params_store: dict = read_params()
    _rebuild_lock = threading.Lock()

    app = Flask(__name__)
    app.logger.setLevel(logging.ERROR)

    @app.route("/")
    def index():
        return render_template_string(_HTML)

    @app.route("/params")
    def get_params():
        return jsonify(params_store)

    @app.route("/update", methods=["POST"])
    def update():
        new_params = request.get_json()
        params_store.clear()
        params_store.update(new_params)

        if not _rebuild_lock.acquire(blocking=False):
            return jsonify({"status": "Busy, skipped."})
        try:
            write_params(params_store)
            inner, outer, vision = call_assembly_from_params()
            show(inner, outer, vision)
            return jsonify({"status": f"[{time.strftime('%H:%M:%S')}] Updated."})
        except Exception as e:
            return jsonify({"status": f"Error: {e}"})
        finally:
            _rebuild_lock.release()

    @app.route("/add_wheel", methods=["POST"])
    def add_wheel():
        existing = [k for k in params_store if k.startswith("wheel_")]
        next_idx = max((int(k.split("_")[1]) for k in existing), default=-1) + 1
        params_store[f"wheel_{next_idx}"] = {"x": 0.0, "y": 0.0, "z": 0.0}
        write_params(params_store)
        return jsonify({"params": params_store, "status": f"wheel_{next_idx} added."})

    @app.route("/remove_wheel", methods=["POST"])
    def remove_wheel():
        existing = sorted(k for k in params_store if k.startswith("wheel_"))
        if not existing:
            return jsonify({"status": "No wheels to remove."})
        params_store.pop(existing[-1])
        write_params(params_store)
        return jsonify({"params": params_store, "status": f"{existing[-1]} removed."})

    @app.route("/add_to_map", methods=["POST"])
    def add_to_map():
        if add_current_gui_assembly_to_map is None:
            return jsonify({"status": "Add-to-map command is not configured."})

        new_params = request.get_json()
        params_store.clear()
        params_store.update(new_params)
        write_params(params_store)

        assembly_id = add_current_gui_assembly_to_map()
        if assembly_id is None:
            return jsonify({"status": "Invalid assembly. Not added. See checks.txt."})
        return jsonify({"status": f"Added to assembly_map.yaml as id {assembly_id}."})

    def _initial_render() -> None:
        time.sleep(0.5)
        try:
            inner, outer, vision = call_assembly_from_params()
            show(inner, outer, vision)
        except Exception as e:
            print(f"Initial render error: {e}")

    threading.Thread(target=_initial_render, daemon=True).start()

    url = f"http://localhost:{port}"
    print(f"Slider UI: {url}")
    webbrowser.open(url)
    app.run(port=port, threaded=True, use_reloader=False)
