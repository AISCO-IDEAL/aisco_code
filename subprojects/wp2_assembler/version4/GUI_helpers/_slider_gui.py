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
    :root {
      --border: #d7dde5;
      --panel: #f7f9fc;
      --text: #111827;
      --muted: #667085;
      --blue: #2563eb;
      --green: #138a43;
      --red: #b42318;
    }
    body {
      margin: 0;
      padding: 24px;
      max-width: 1120px;
      color: var(--text);
      font-family: Arial, sans-serif;
      background: white;
    }
    h2 { margin: 0 0 16px; font-size: 22px; }
    .layout {
      display: grid;
      grid-template-columns: minmax(420px, 1fr) 360px;
      gap: 18px;
      align-items: start;
    }
    canvas {
      width: 100%;
      height: 620px;
      display: block;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fbfcff;
      cursor: grab;
      touch-action: none;
    }
    canvas.dragging { cursor: grabbing; }
    .side {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
    }
    th, td {
      border-bottom: 1px solid var(--border);
      padding: 7px;
      text-align: left;
      font-size: 13px;
    }
    th {
      background: var(--panel);
      color: var(--muted);
      font-weight: 600;
    }
    tr:last-child td { border-bottom: 0; }
    input[type=number] {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 6px;
      font: inherit;
    }
    .part-name { font-family: monospace; font-weight: 600; }
    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    button {
      border: 1px solid var(--border);
      border-radius: 6px;
      background: white;
      padding: 8px 10px;
      cursor: pointer;
      font: inherit;
      font-size: 13px;
    }
    button.primary {
      grid-column: span 2;
      border-color: var(--blue);
      background: var(--blue);
      color: white;
    }
    #status { color: var(--muted); min-height: 20px; font-size: 13px; }
    #checks {
      min-height: 96px;
      margin: 0;
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--panel);
      white-space: pre-wrap;
      font: 12px/1.35 monospace;
    }
    .ok { color: var(--green); }
    .bad { color: var(--red); }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      canvas { height: 520px; }
    }
  </style>
</head>
<body>
  <h2>Assembly Parameters</h2>

  <div class="layout">
    <canvas id="preview" width="760" height="620"></canvas>

    <div class="side">
      <table>
        <thead>
          <tr>
            <th>Part</th>
            <th>X</th>
            <th>Y</th>
            <th>Z</th>
          </tr>
        </thead>
        <tbody id="controls"></tbody>
      </table>

      <div class="actions">
        <button class="primary" onclick="generateAssembly()">Generate PLY/STEP</button>
        <button onclick="addToMap()">Add to assembly map</button>
        <button onclick="addWheel()">+ Add wheel</button>
        <button onclick="removeWheel()">- Remove wheel</button>
      </div>

      <div id="status">Loading...</div>
      <pre id="checks">checks.txt will appear here after generation.</pre>
    </div>
  </div>

  <script>
    let params = {};
    let dragPart = null;
    let hoverPart = null;
    const canvas = document.getElementById('preview');
    const ctx = canvas.getContext('2d');
    const bounds = {xmin: -3000, xmax: 3000, ymin: -2200, ymax: 2200};
    const wheelRadius = 145;
    const seatHalfX = 330;
    const seatHalfY = 180;

    async function loadParams() {
      const r = await fetch('/params');
      params = await r.json();
      render();
      drawPreview();
      document.getElementById('status').textContent = 'Ready.';
      await loadChecks();
    }

    function orderedParts() {
      return Object.entries(params).sort(([a], [b]) => {
        if (a === 'seat') return -1;
        if (b === 'seat') return 1;
        const ai = parseInt(a.split('_')[1] || '0', 10);
        const bi = parseInt(b.split('_')[1] || '0', 10);
        return ai - bi;
      });
    }

    function render() {
      const body = document.getElementById('controls');
      body.innerHTML = '';
      for (const [part, vals] of orderedParts()) {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td class="part-name">${part}</td>
          ${inputCell(part, 'x', vals.x)}
          ${inputCell(part, 'y', vals.y)}
          ${inputCell(part, 'z', vals.z)}
        `;
        body.appendChild(row);
      }
    }

    function inputCell(part, axis, value) {
      return `
        <td>
          <input type="number" step="1" value="${Math.round(value)}"
            data-key="${part}" data-axis="${axis}" oninput="onEdit(this)">
        </td>
      `;
    }

    function onEdit(el) {
      params[el.dataset.key][el.dataset.axis] = parseFloat(el.value || '0');
      drawPreview();
    }

    function toCanvas(x, y) {
      const pad = 42;
      const sx = (canvas.width - 2 * pad) / (bounds.xmax - bounds.xmin);
      const sy = (canvas.height - 2 * pad) / (bounds.ymax - bounds.ymin);
      const s = Math.min(sx, sy);
      const cx = pad + (x - bounds.xmin) * s;
      const cy = canvas.height - pad - (y - bounds.ymin) * s;
      return {x: cx, y: cy, s};
    }

    function fromCanvas(cx, cy) {
      const origin = toCanvas(bounds.xmin, bounds.ymin);
      const x = bounds.xmin + (cx - origin.x) / origin.s;
      const y = bounds.ymin + (origin.y - cy) / origin.s;
      return {x: Math.round(x), y: Math.round(y)};
    }

    function drawPreview() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawGrid();
      drawWheelPolygon();
      for (const [part, vals] of orderedParts()) {
        if (part === 'seat') drawSeat(part, vals);
        else drawWheel(part, vals);
      }
    }

    function drawGrid() {
      const zero = toCanvas(0, 0);
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      for (let x = -3000; x <= 3000; x += 1000) {
        const p = toCanvas(x, 0);
        ctx.beginPath();
        ctx.moveTo(p.x, 20);
        ctx.lineTo(p.x, canvas.height - 20);
        ctx.stroke();
      }
      for (let y = -2000; y <= 2000; y += 1000) {
        const p = toCanvas(0, y);
        ctx.beginPath();
        ctx.moveTo(20, p.y);
        ctx.lineTo(canvas.width - 20, p.y);
        ctx.stroke();
      }
      ctx.strokeStyle = '#9ca3af';
      ctx.beginPath();
      ctx.moveTo(zero.x, 20);
      ctx.lineTo(zero.x, canvas.height - 20);
      ctx.moveTo(20, zero.y);
      ctx.lineTo(canvas.width - 20, zero.y);
      ctx.stroke();
    }

    function drawWheelPolygon() {
      const wheels = orderedParts().filter(([part]) => part.startsWith('wheel_'));
      if (wheels.length < 3) return;
      ctx.strokeStyle = '#93c5fd';
      ctx.fillStyle = 'rgba(147, 197, 253, 0.14)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      wheels.forEach(([_, vals], i) => {
        const p = toCanvas(vals.x, vals.y);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }

    function drawWheel(part, vals) {
      const p = toCanvas(vals.x, vals.y);
      const r = Math.max(8, wheelRadius * p.s);
      ctx.fillStyle = part === hoverPart ? '#1d4ed8' : '#111827';
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();
      drawLabel(part, p.x, p.y - r - 8);
    }

    function drawSeat(part, vals) {
      const p = toCanvas(vals.x, vals.y);
      const w = seatHalfX * 2 * p.s;
      const h = seatHalfY * 2 * p.s;
      ctx.fillStyle = part === hoverPart ? '#f97316' : '#f59e0b';
      ctx.fillRect(p.x - w / 2, p.y - h / 2, w, h);
      ctx.strokeStyle = '#92400e';
      ctx.strokeRect(p.x - w / 2, p.y - h / 2, w, h);
      drawLabel(part, p.x, p.y - h / 2 - 8);
    }

    function drawLabel(text, x, y) {
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      ctx.fillStyle = '#374151';
      ctx.fillText(text, x, y);
    }

    function hitTest(cx, cy) {
      const parts = orderedParts().reverse();
      for (const [part, vals] of parts) {
        const p = toCanvas(vals.x, vals.y);
        if (part === 'seat') {
          const hw = seatHalfX * p.s;
          const hh = seatHalfY * p.s;
          if (Math.abs(cx - p.x) <= hw && Math.abs(cy - p.y) <= hh) return part;
        } else {
          const r = Math.max(10, wheelRadius * p.s);
          if (Math.hypot(cx - p.x, cy - p.y) <= r) return part;
        }
      }
      return null;
    }

    function pointerPos(event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: (event.clientX - rect.left) * canvas.width / rect.width,
        y: (event.clientY - rect.top) * canvas.height / rect.height
      };
    }

    canvas.addEventListener('pointerdown', (event) => {
      const p = pointerPos(event);
      dragPart = hitTest(p.x, p.y);
      if (dragPart) {
        canvas.setPointerCapture(event.pointerId);
        canvas.classList.add('dragging');
      }
    });

    canvas.addEventListener('pointermove', (event) => {
      const p = pointerPos(event);
      if (dragPart) {
        const world = fromCanvas(p.x, p.y);
        params[dragPart].x = world.x;
        params[dragPart].y = world.y;
        render();
        drawPreview();
        return;
      }
      hoverPart = hitTest(p.x, p.y);
      drawPreview();
    });

    canvas.addEventListener('pointerup', (event) => {
      dragPart = null;
      canvas.classList.remove('dragging');
      try { canvas.releasePointerCapture(event.pointerId); } catch (_) {}
    });

    async function generateAssembly() {
      document.getElementById('status').textContent = 'Rebuilding...';
      const r = await fetch('/update', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params)
      });
      const res = await r.json();
      document.getElementById('status').textContent = res.status;
      await loadChecks();
    }

    async function addWheel() {
      const r = await fetch('/add_wheel', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params)
      });
      const res = await r.json();
      params = res.params;
      render();
      drawPreview();
      document.getElementById('status').textContent = res.status;
      await loadChecks();
    }

    async function removeWheel() {
      const r = await fetch('/remove_wheel', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params)
      });
      const res = await r.json();
      if (res.params) {
        params = res.params;
        render();
        drawPreview();
      }
      document.getElementById('status').textContent = res.status;
      await loadChecks();
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
      await loadChecks();
    }

    async function loadChecks() {
      const box = document.getElementById('checks');
      const r = await fetch('/checks');
      if (!r.ok) {
        box.textContent = 'checks.txt is not available.';
        box.className = '';
        return;
      }
      const res = await r.json();
      box.textContent = res.checks || '';
      box.className = (res.checks || '').startsWith('OK:') ? 'ok' : 'bad';
    }

    loadParams();
  </script>
</body>
</html>
"""


def run(  # noqa: PLR0915
    read_params: Callable,
    write_params: Callable,
    call_assembly_from_params: Callable,
    add_current_gui_assembly_to_map: Callable | None = None,
    read_checks: Callable | None = None,
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

    @app.route("/checks")
    def get_checks():
        if read_checks is None:
            return jsonify({"checks": "checks.txt is not configured."})
        return jsonify({"checks": read_checks()})

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
        new_params = request.get_json(silent=True)
        if new_params:
            params_store.clear()
            params_store.update(new_params)

        existing = [k for k in params_store if k.startswith("wheel_")]
        next_idx = max((int(k.split("_")[1]) for k in existing), default=-1) + 1
        params_store[f"wheel_{next_idx}"] = {"x": 0.0, "y": 0.0, "z": 0.0}
        write_params(params_store)
        return jsonify({"params": params_store, "status": f"wheel_{next_idx} added."})

    @app.route("/remove_wheel", methods=["POST"])
    def remove_wheel():
        new_params = request.get_json(silent=True)
        if new_params:
            params_store.clear()
            params_store.update(new_params)

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
