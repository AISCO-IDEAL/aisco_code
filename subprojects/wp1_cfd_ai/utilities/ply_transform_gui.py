#!/usr/bin/env python3
"""
PLY Transform GUI — web-based
Usage:
    python ply_transform_gui.py [--port 7860]
Then open  http://localhost:7860  in your browser.
On HPC with SSH: forward the port first:
    ssh -L 7860:localhost:7860 <your-hpc-host>
"""

import argparse
import math
import os
import struct
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

# ─────────────────────────────────────────────────────────────────────────────
# PLY I/O  (binary + ascii)
# ─────────────────────────────────────────────────────────────────────────────

PLY_TYPES = {
    "char": ("b", 1),
    "uchar": ("B", 1),
    "short": ("h", 2),
    "ushort": ("H", 2),
    "int": ("i", 4),
    "uint": ("I", 4),
    "float": ("f", 4),
    "double": ("d", 8),
    "int8": ("b", 1),
    "uint8": ("B", 1),
    "int16": ("h", 2),
    "uint16": ("H", 2),
    "int32": ("i", 4),
    "uint32": ("I", 4),
    "float32": ("f", 4),
    "float64": ("d", 8),
}


def _parse_header(f):
    lines = []
    while True:
        line = f.readline()
        lines.append(line)
        if line.strip() == b"end_header":
            break
    header_bytes = b"".join(lines)
    fmt = "ascii"
    elements = []
    current = None
    for line in lines:
        s = line.decode("ascii", errors="replace").strip()
        if s.startswith("format"):
            fmt = s.split()[1]
        elif s.startswith("element"):
            parts = s.split()
            current = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elements.append(current)
        elif s.startswith("property") and current is not None:
            parts = s.split()
            if parts[1] == "list":
                current["properties"].append(
                    {
                        "kind": "list",
                        "count_type": parts[2],
                        "value_type": parts[3],
                        "name": parts[4],
                    }
                )
            else:
                current["properties"].append(
                    {
                        "kind": "scalar",
                        "type": parts[1],
                        "name": parts[2],
                    }
                )
    return header_bytes, fmt, elements


def read_ply(filepath):
    with open(filepath, "rb") as f:
        header_bytes, fmt, elements = _parse_header(f)
        vertex_el = next((e for e in elements if e["name"] == "vertex"), None)
        face_el = next((e for e in elements if e["name"] == "face"), None)
        if vertex_el is None:
            raise ValueError("No vertex element in PLY")
        props = vertex_el["properties"]
        xyz_indices = [i for i, p in enumerate(props) if p["name"] in ("x", "y", "z")]
        raw_vdata: Any  # bytes (binary) or list[str] (ascii)
        raw_fdata: Any  # bytes (binary) or list[str] (ascii)
        if fmt == "ascii":
            raw_vdata, verts = [], []
            for _ in range(vertex_el["count"]):
                line = f.readline().decode("ascii").strip()
                raw_vdata.append(line)
                parts = line.split()
                verts.append([float(parts[i]) for i in xyz_indices])
            raw_fdata = [f.readline().decode("ascii").strip() for _ in range(face_el["count"])] if face_el else []
        else:
            endian = "<" if "little" in fmt else ">"
            vfmt = endian + "".join(PLY_TYPES[p["type"]][0] for p in props if p["kind"] == "scalar")
            vsz = struct.calcsize(vfmt)
            raw_vdata = f.read(vsz * vertex_el["count"])
            verts = []
            for i in range(vertex_el["count"]):
                row = struct.unpack_from(vfmt, raw_vdata, i * vsz)
                verts.append([row[j] for j in xyz_indices])
            raw_fdata = f.read() if face_el else b""
    return verts, raw_vdata, raw_fdata, header_bytes, fmt, elements


def write_ply(filepath, verts, raw_vdata, raw_fdata, header_bytes, fmt, elements):
    vertex_el = next(e for e in elements if e["name"] == "vertex")
    props = vertex_el["properties"]
    xyz_indices = [i for i, p in enumerate(props) if p["name"] in ("x", "y", "z")]
    with open(filepath, "wb") as f:
        f.write(header_bytes)
        if fmt == "ascii":
            for idx, line in enumerate(raw_vdata):
                parts = line.split()
                xyz, ptr = verts[idx], 0
                new = []
                for ci, _ in enumerate(parts):
                    if ci in xyz_indices:
                        new.append(f"{xyz[ptr]:.8g}")
                        ptr += 1
                    else:
                        new.append(parts[ci])
                f.write((" ".join(new) + "\n").encode("ascii"))
            for line in raw_fdata:
                f.write((line + "\n").encode("ascii"))
        else:
            endian = "<" if "little" in fmt else ">"
            vfmt = endian + "".join(PLY_TYPES[p["type"]][0] for p in props if p["kind"] == "scalar")
            vsz = struct.calcsize(vfmt)
            buf = bytearray(raw_vdata)
            for i in range(vertex_el["count"]):
                row = list(struct.unpack_from(vfmt, buf, i * vsz))
                xyz = verts[i]
                for slot, ci in enumerate(xyz_indices):
                    row[ci] = type(row[ci])(xyz[slot])
                struct.pack_into(vfmt, buf, i * vsz, *row)
            f.write(bytes(buf))
            f.write(raw_fdata)


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────


def _scale(verts, sx, sy, sz):
    return [[v[0] * sx, v[1] * sy, v[2] * sz] for v in verts]


def _rotate(verts, axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    R = {
        "x": ((1, 0, 0), (0, c, -s), (0, s, c)),
        "y": ((c, 0, s), (0, 1, 0), (-s, 0, c)),
        "z": ((c, -s, 0), (s, c, 0), (0, 0, 1)),
    }[axis]
    return [
        [
            R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
            R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
            R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
        ]
        for v in verts
    ]


def _mirror(verts, axes):
    sx = -1 if "x" in axes else 1
    sy = -1 if "y" in axes else 1
    sz = -1 if "z" in axes else 1
    return [[v[0] * sx, v[1] * sy, v[2] * sz] for v in verts]


def _parse_faces_for_viewer(raw_fdata, face_el, fmt):
    """Parse face data into a flat list of triangle vertex indices."""
    if not face_el:
        return []
    list_prop = next((p for p in face_el["properties"] if p["kind"] == "list"), None)
    if not list_prop:
        return []

    if fmt == "ascii":
        indices = []
        for line in raw_fdata if isinstance(raw_fdata, list) else []:
            parts = line.split()
            if not parts:
                continue
            n = int(parts[0])
            face_verts = [int(parts[i + 1]) for i in range(n)]
            for i in range(1, n - 1):
                indices.extend([face_verts[0], face_verts[i], face_verts[i + 1]])
        return indices
    else:
        endian = "<" if "little" in fmt else ">"
        cf, cs = PLY_TYPES[list_prop["count_type"]]
        vf, vs = PLY_TYPES[list_prop["value_type"]]
        cfmt = endian + cf
        vfmt = endian + vf
        indices = []
        offset = 0
        data = bytes(raw_fdata) if not isinstance(raw_fdata, (bytes, bytearray)) else raw_fdata
        for _ in range(face_el["count"]):
            for prop in face_el["properties"]:
                if prop["kind"] == "scalar":
                    offset += PLY_TYPES[prop["type"]][1]
                elif prop["kind"] == "list":
                    p_cf, p_cs = PLY_TYPES[prop["count_type"]]
                    p_vs = PLY_TYPES[prop["value_type"]][1]
                    n = struct.unpack_from(endian + p_cf, data, offset)[0]
                    offset += p_cs
                    if prop is list_prop:
                        face_verts = [struct.unpack_from(vfmt, data, offset + i * vs)[0] for i in range(n)]
                        for i in range(1, n - 1):
                            indices.extend([face_verts[0], face_verts[i], face_verts[i + 1]])
                    offset += n * p_vs
        return indices


def _apply_pipeline_step(verts, step):
    op = step.get("op", "")
    if op == "scale-uniform":
        sf = float(step.get("factor", 1.0))
        return _scale(verts, sf, sf, sf)
    elif op == "scale-axis":
        return _scale(verts, float(step.get("sx", 1)), float(step.get("sy", 1)), float(step.get("sz", 1)))
    elif op == "rotate":
        return _rotate(verts, step.get("axis", "z"), float(step.get("deg", 90)))
    elif op == "mirror":
        return _mirror(verts, step.get("axes", ["y"]))
    return verts


def _step_label(step):
    op = step.get("op", "")
    if op == "scale-uniform":
        return f"scale \u00d7{float(step.get('factor', 1)):g}"
    elif op == "scale-axis":
        sx, sy, sz = float(step.get("sx", 1)), float(step.get("sy", 1)), float(step.get("sz", 1))
        return f"scale X\u00d7{sx:g} Y\u00d7{sy:g} Z\u00d7{sz:g}"
    elif op == "rotate":
        return f"rotate {step.get('axis', 'z').upper()} {float(step.get('deg', 90)):g}\u00b0"
    elif op == "mirror":
        return f"mirror {''.join(sorted(step.get('axes', ['y'])))}"
    return "?"


def _step_tag(step):
    op = step.get("op", "")
    if op == "scale-uniform":
        return f"s{float(step.get('factor', 1)):g}"
    elif op == "scale-axis":
        return f"s{float(step.get('sx', 1)):g}_{float(step.get('sy', 1)):g}_{float(step.get('sz', 1)):g}"
    elif op == "rotate":
        return f"r{step.get('axis', 'z')}{float(step.get('deg', 90)):g}"
    elif op == "mirror":
        return f"m{''.join(sorted(step.get('axes', ['y'])))}"
    return "x"


def _build_mesh_binary(verts, indices):
    vc = len(verts)
    tc = len(indices) // 3
    if verts:
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        zs = [v[2] for v in verts]
        extents = [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]
        center = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2]
    else:
        extents = center = [0, 0, 0]
    parts = [struct.pack("<II6f", vc, tc, *extents, *center)]
    if verts:
        flat = []
        for v in verts:
            flat.extend(v)
        parts.append(struct.pack(f"<{len(flat)}f", *flat))
    if indices:
        parts.append(struct.pack(f"<{len(indices)}I", *indices))
    return b"".join(parts)


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Flask app
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

app = Flask(__name__)

_HTML_FILE = Path(__file__).parent / "ply_transform_gui.html"


def _load_html():
    with open(_HTML_FILE, encoding="utf-8") as _f:
        return _f.read()


# (HTML now lives in ply_transform_gui.html)


# ─────────────────────────────────────────────────────────────────────────────
# API routes
# ─────────────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return _load_html()


@app.route("/api/mesh", methods=["POST"])
def api_mesh():
    data = request.get_json(force=True)
    path = data.get("path", "")
    if not os.path.isfile(path):
        return jsonify(ok=False, error=f"File not found: {path}")
    try:
        verts, _raw_vdata, raw_fdata, _header_bytes, fmt, elements = read_ply(path)
        face_el = next((e for e in elements if e["name"] == "face"), None)
        indices = _parse_faces_for_viewer(raw_fdata, face_el, fmt)
        positions = [coord for v in verts for coord in v]
        if verts:
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            zs = [v[2] for v in verts]
            extents = [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]
            center = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2]
        else:
            extents, center = [0, 0, 0], [0, 0, 0]
        return jsonify(
            ok=True,
            positions=positions,
            indices=indices,
            center=center,
            extents=extents,
            vertex_count=len(verts),
            face_count=len(indices) // 3,
        )
    except Exception as e:
        return jsonify(ok=False, error=str(e))


@app.route("/api/mesh_bin", methods=["POST"])
def api_mesh_bin():
    data = request.get_json(force=True)
    path = data.get("path", "")
    if not os.path.isfile(path):
        return jsonify(ok=False, error=f"File not found: {path}")
    try:
        verts, _raw_vdata, raw_fdata, _header_bytes, fmt, elements = read_ply(path)
        face_el = next((e for e in elements if e["name"] == "face"), None)
        indices = _parse_faces_for_viewer(raw_fdata, face_el, fmt)
        return Response(_build_mesh_binary(verts, indices), mimetype="application/octet-stream")
    except Exception as e:
        return jsonify(ok=False, error=str(e))


@app.route("/api/preview", methods=["POST"])
def api_preview():
    data = request.get_json(force=True)
    path = data.get("path", "")
    pipeline = data.get("pipeline", [])
    if not os.path.isfile(path):
        return jsonify(ok=False, error=f"File not found: {path}")
    try:
        verts, _raw_vdata, raw_fdata, _header_bytes, fmt, elements = read_ply(path)
        face_el = next((e for e in elements if e["name"] == "face"), None)
        indices = _parse_faces_for_viewer(raw_fdata, face_el, fmt)
        current = [v[:] for v in verts]
        for step in pipeline:
            current = _apply_pipeline_step(current, step)
        return Response(_build_mesh_binary(current, indices), mimetype="application/octet-stream")
    except Exception as e:
        return jsonify(ok=False, error=str(e))


@app.route("/api/preview_steps", methods=["POST"])
def api_preview_steps():
    """Return binary mesh for each repetition step.

    Format: uint32 num_steps, then per step: uint32 size, <mesh_binary>
    """
    data = request.get_json(force=True)
    path = data.get("path", "")
    pipeline_def = data.get("pipeline", [])
    repeat = int(data.get("repeat", 1))
    if not os.path.isfile(path):
        return jsonify(ok=False, error=f"File not found: {path}")
    try:
        verts, _raw_vdata, raw_fdata, _header_bytes, fmt, elements = read_ply(path)
        face_el = next((e for e in elements if e["name"] == "face"), None)
        indices = _parse_faces_for_viewer(raw_fdata, face_el, fmt)
        current = [v[:] for v in verts]
        parts = [struct.pack("<I", repeat)]
        for _ in range(repeat):
            for step in pipeline_def:
                current = _apply_pipeline_step(current, step)
            mesh_bin = _build_mesh_binary(current, indices)
            parts.append(struct.pack("<I", len(mesh_bin)))
            parts.append(mesh_bin)
        return Response(b"".join(parts), mimetype="application/octet-stream")
    except Exception as e:
        return jsonify(ok=False, error=str(e))


@app.route("/api/inspect", methods=["POST"])
def api_inspect():
    data = request.get_json(force=True)
    path = data.get("path", "")
    if not os.path.isfile(path):
        return jsonify(ok=False, error=f"File not found: {path}")
    try:
        _verts, _, _, _, fmt, elements = read_ply(path)
        vertex_count = next(e["count"] for e in elements if e["name"] == "vertex")
        face_el = next((e for e in elements if e["name"] == "face"), None)
        face_count = face_el["count"] if face_el else 0
        return jsonify(ok=True, vertices=vertex_count, faces=face_count, format=fmt)
    except Exception as e:
        return jsonify(ok=False, error=str(e))


@app.route("/api/transform", methods=["POST"])
def api_transform():
    data = request.get_json(force=True)
    path = data.get("path", "")
    pipeline = data.get("pipeline", [])
    repeat = int(data.get("repeat", 1))
    out_dir = data.get("output_dir", "").strip() or str(Path(path).parent)

    if not os.path.isfile(path):
        return jsonify(ok=False, error=f"File not found: {path}")
    if not pipeline:
        return jsonify(ok=False, error="No transforms in pipeline")

    try:
        verts, raw_vdata, raw_fdata, header_bytes, fmt, elements = read_ply(path)
    except Exception as e:
        return jsonify(ok=False, error=str(e))

    os.makedirs(out_dir, exist_ok=True)
    base = Path(path).stem
    tag = "_".join(_step_tag(s) for s in pipeline)
    desc = " \u2192 ".join(_step_label(s) for s in pipeline)

    current = [v[:] for v in verts]
    created = []
    try:
        for i in range(repeat):
            for step in pipeline:
                current = _apply_pipeline_step(current, step)
            out_path = os.path.join(out_dir, f"{base}_{tag}_step{i + 1}.ply")
            write_ply(out_path, current, raw_vdata, raw_fdata, header_bytes, fmt, elements)
            created.append({"path": out_path, "info": f"step {i + 1}: {desc}"})
    except Exception as e:
        return jsonify(ok=False, error=str(e))

    return jsonify(ok=True, files=created)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on (default: 7860)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    args = parser.parse_args()

    print(f"\n{'=' * 54}")
    print("  PLY Transform GUI")
    print(f"  http://{args.host}:{args.port}")
    print(f"{'=' * 54}")
    print("\n  On HPC — forward port with:")
    print(f"    ssh -L {args.port}:localhost:{args.port} <hpc-host>")
    print(f"  Then open  http://localhost:{args.port}  in your local browser.\n")

    app.run(host=args.host, port=args.port, debug=False)
