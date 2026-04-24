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
import json as json_mod
import math
import os
import re
import shutil
import struct
import threading
import traceback
import uuid
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, jsonify, request

# Background job tracker for cache generation (job_id -> progress dict)
_cache_jobs: dict = {}
_GUI_CACHE_VERSION = "v2"

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


def _parse_faces_for_viewer(raw_fdata, face_el, fmt):  # noqa: PLR0912
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
        _cf, _cs = PLY_TYPES[list_prop["count_type"]]
        vf, vs = PLY_TYPES[list_prop["value_type"]]
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
    downsample = int(data.get("downsample", 0))
    if not os.path.isfile(path):
        return jsonify(ok=False, error=f"File not found: {path}")
    try:
        verts, _raw_vdata, raw_fdata, _header_bytes, fmt, elements = read_ply(path)
        face_el = next((e for e in elements if e["name"] == "face"), None)
        indices = _parse_faces_for_viewer(raw_fdata, face_el, fmt)
        if downsample > 0 and len(verts) > downsample:
            pos_arr = np.array(verts, dtype=np.float32)
            sel = _spatial_subsample(pos_arr, downsample)
            verts = [verts[int(i)] for i in sel]
            indices = []  # point-cloud render when downsampled
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


def _write_ply_points(filepath: str, pos: "np.ndarray") -> None:
    """Write a binary little-endian PLY containing only xyz float32 positions."""
    n = pos.shape[0]
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    ).encode("ascii")
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(pos.astype(np.float32).tobytes())


@app.route("/api/subsample_save", methods=["POST"])
def api_subsample_save():
    """Spatially subsample a PLY to n_target points and write it.

    Writes a point-cloud PLY (no faces) with only xyz float32 properties.
    If overwrite=True, replaces the source file; otherwise writes
    <stem>_sub<N>.ply in the same directory.
    """
    data = request.get_json(force=True)
    path = data.get("path", "").strip()
    n_target = int(data.get("n_target", 0))
    overwrite = bool(data.get("overwrite", False))
    if not os.path.isfile(path):
        return jsonify(ok=False, error=f"File not found: {path}")
    if n_target <= 0:
        return jsonify(ok=False, error="n_target must be > 0")
    try:
        verts, _rv, _rf, _hb, _fmt, _el = read_ply(path)
        pos = np.array(verts, dtype=np.float32)
        n_verts = pos.shape[0]
        if n_target >= n_verts:
            return jsonify(
                ok=False,
                error=f"n_target ({n_target:,}) ≥ vertex count ({n_verts:,}) — nothing to subsample",
            )
        sel = _spatial_subsample(pos, n_target)
        sel_pos = pos[sel]
        out_path = path if overwrite else str(Path(path).parent / f"{Path(path).stem}_sub{len(sel)}.ply")
        _write_ply_points(out_path, sel_pos)
        return jsonify(ok=True, path=out_path, n_original=n_verts, n_saved=len(sel))
    except Exception as e:
        return jsonify(ok=False, error=str(e))


@app.route("/api/list_ply", methods=["POST"])
def api_list_ply():
    """List all .ply files in a directory (non-recursive)."""
    data = request.get_json(force=True)
    folder = data.get("folder", "").strip()
    if not folder or not os.path.isdir(folder):
        return jsonify(ok=False, error=f"Not a directory: {folder}")
    files = sorted(
        [f for f in os.listdir(folder) if f.lower().endswith(".ply")],
        key=_natural_sort_key,
    )
    return jsonify(ok=True, files=files, folder=folder)


@app.route("/api/subdivide", methods=["POST"])
def api_subdivide():
    """Subdivide a PLY mesh, returning the new mesh binary for the viewer.

    Uses Open3D Loop subdivision (smoothing) when available;
    falls back to midpoint subdivision (no smoothing).
    ``iterations`` controls how many subdivision passes to apply (default 1).
    Optionally write the result to disk when ``save_path`` is supplied.
    """
    data = request.get_json(force=True)
    path = data.get("path", "")
    save_path = data.get("save_path", "").strip()
    iterations = int(data.get("iterations", 1))
    if not os.path.isfile(path):
        return jsonify(ok=False, error=f"File not found: {path}")
    try:
        verts, _raw_vdata, raw_fdata, _header_bytes, fmt, elements = read_ply(path)
        face_el = next((e for e in elements if e["name"] == "face"), None)
        tris = _parse_faces_for_viewer(raw_fdata, face_el, fmt)
        if not tris:
            return jsonify(ok=False, error="No faces to subdivide")

        V = np.array(verts, dtype=np.float64)
        F = np.array(tris, dtype=np.int32).reshape(-1, 3)
        method = "midpoint"

        try:
            import open3d as o3d  # noqa: PLC0415

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(V)
            mesh.triangles = o3d.utility.Vector3iVector(F)
            mesh = mesh.subdivide_loop(number_of_iterations=iterations)
            new_V = np.asarray(mesh.vertices, dtype=np.float32)
            new_F = np.asarray(mesh.triangles, dtype=np.int32)
            method = "loop"
        except ImportError:
            # Fallback: midpoint subdivision (no smoothing)
            cur_V, cur_F = V, F
            for _ in range(iterations):
                edge_mid = {}
                new_verts = list(cur_V)
                new_faces = []
                for tri in cur_F:
                    mids = []
                    for a_i in range(3):
                        b_i = (a_i + 1) % 3
                        edge = (min(int(tri[a_i]), int(tri[b_i])), max(int(tri[a_i]), int(tri[b_i])))
                        if edge not in edge_mid:
                            mid = (cur_V[edge[0]] + cur_V[edge[1]]) * 0.5
                            edge_mid[edge] = len(new_verts)
                            new_verts.append(mid)
                        mids.append(edge_mid[edge])
                    v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
                    m01, m12, m20 = mids[0], mids[1], mids[2]
                    new_faces.extend(
                        [
                            [v0, m01, m20],
                            [v1, m12, m01],
                            [v2, m20, m12],
                            [m01, m12, m20],
                        ]
                    )
                cur_V = np.array(new_verts, dtype=np.float64)
                cur_F = np.array(new_faces, dtype=np.int32)
            new_V = cur_V.astype(np.float32)
            new_F = cur_F

        flat_idx = new_F.flatten().tolist()
        blob = _build_mesh_binary(new_V.tolist(), flat_idx)

        if save_path:
            _write_triangle_ply(save_path, new_V, new_F, fmt)

        return Response(
            blob,
            mimetype="application/octet-stream",
            headers={"X-Vertices": str(new_V.shape[0]), "X-Faces": str(new_F.shape[0]), "X-Method": method},
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=str(e))


@app.route("/api/decimate", methods=["POST"])
def api_decimate():
    """Decimate a PLY mesh to a target face ratio, returning new mesh binary.

    Uses quadric edge-collapse (via Open3D if available, else a greedy fallback).
    ``ratio`` is the fraction of faces to keep (0.0-1.0).
    """
    data = request.get_json(force=True)
    path = data.get("path", "")
    ratio = float(data.get("ratio", 0.5))
    save_path = data.get("save_path", "").strip()
    if not os.path.isfile(path):
        return jsonify(ok=False, error=f"File not found: {path}")
    ratio = max(0.01, min(1.0, ratio))
    try:
        verts, _raw_vdata, raw_fdata, _header_bytes, fmt, elements = read_ply(path)
        face_el = next((e for e in elements if e["name"] == "face"), None)
        tris = _parse_faces_for_viewer(raw_fdata, face_el, fmt)
        if not tris:
            return jsonify(ok=False, error="No faces to decimate")

        V = np.array(verts, dtype=np.float64)
        F = np.array(tris, dtype=np.int32).reshape(-1, 3)
        target = max(1, int(F.shape[0] * ratio))

        try:
            import open3d as o3d  # noqa: PLC0415

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(V)
            mesh.triangles = o3d.utility.Vector3iVector(F)
            mesh = mesh.simplify_quadric_decimation(target)
            new_V = np.asarray(mesh.vertices, dtype=np.float32)
            new_F = np.asarray(mesh.triangles, dtype=np.int32)
        except ImportError:
            # Fallback: uniform face sampling (no topology-aware decimation)
            idx = np.linspace(0, F.shape[0] - 1, target, dtype=int)
            new_F = F[idx]
            used = np.unique(new_F.flatten())
            remap = np.zeros(V.shape[0], dtype=np.int32)
            remap[used] = np.arange(len(used), dtype=np.int32)
            new_V = V[used].astype(np.float32)
            new_F = remap[new_F]

        flat_idx = new_F.flatten().tolist()
        blob = _build_mesh_binary(new_V.tolist(), flat_idx)

        if save_path:
            _write_triangle_ply(save_path, new_V, new_F, fmt)

        return Response(
            blob, mimetype="application/octet-stream", headers={"X-Vertices": str(new_V.shape[0]), "X-Faces": str(new_F.shape[0])}
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=str(e))


def _write_triangle_ply(filepath, vertices, faces, fmt="binary_little_endian"):
    """Write a simple triangle PLY (xyz positions + triangle faces)."""
    nv, nf = vertices.shape[0], faces.shape[0]
    header_lines = [
        "ply",
        f"format {'ascii 1.0' if fmt == 'ascii' else 'binary_little_endian 1.0'}",
        f"element vertex {nv}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {nf}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    header = ("\n".join(header_lines) + "\n").encode("ascii")
    with open(filepath, "wb") as f:
        f.write(header)
        if fmt == "ascii":
            f.writelines(f"{v[0]:.8g} {v[1]:.8g} {v[2]:.8g}\n".encode("ascii") for v in vertices)
            f.writelines(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n".encode("ascii") for tri in faces)
        else:
            f.write(np.ascontiguousarray(vertices, dtype="<f4").tobytes())
            for tri in faces:
                f.write(struct.pack("<B", 3))
                f.write(struct.pack("<3i", int(tri[0]), int(tri[1]), int(tri[2])))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset viewer routes
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_torch():
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Dataset Viewer. Run the server with the aisco_env Python.") from exc
    return torch


def _natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def _is_valid_data_pt(filepath: str) -> bool:
    """Check if a .pt file contains valid field data (pos, cp, pressure, x, wall_shear, or face).

    Returns True only if the file is loadable and contains at least 'pos'.
    Returns False for model files or incompatible data.
    """
    if not os.path.isfile(filepath):
        return False
    try:
        import torch  # noqa: PLC0415

        d = torch.load(filepath, weights_only=False, map_location="cpu")
        # Require geometry plus at least one displayable attribute.
        if not hasattr(d, "pos"):
            return False
        # Additional check: pos should be a tensor with shape [N, 3]
        if not hasattr(d.pos, "shape") or len(d.pos.shape) != 2 or d.pos.shape[1] != 3:
            return False
        has_scalar_or_face = any(
            hasattr(d, k) and getattr(d, k) is not None for k in ("x", "cp", "pressure", "wall_shear", "face")
        )
        if not has_scalar_or_face:
            return False
        return True
    except Exception:
        return False


def _to_node_scalar(x):
    """Convert tensor-like per-node data to a [N] float32 array.

    If x is [N, C], use the first channel to keep one scalar per node.
    """
    arr = x.numpy()
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.ndim >= 2:
        return arr[:, 0].astype(np.float32)
    return arr.reshape(-1).astype(np.float32)


def _extract_scalars(d, field):
    """Extract requested scalar field with graceful fallback.

    Returns tuple: (scalars_or_none, effective_field_name_or_none).
    """

    def _get(name):
        if name == "cp" and hasattr(d, "cp") and d.cp is not None:
            return _to_node_scalar(d.cp)
        if name == "pressure" and hasattr(d, "pressure") and d.pressure is not None:
            return _to_node_scalar(d.pressure)
        if name == "wall_shear_mag" and hasattr(d, "wall_shear") and d.wall_shear is not None:
            return np.linalg.norm(d.wall_shear.numpy(), axis=1).astype(np.float32)
        if name == "x" and hasattr(d, "x") and d.x is not None:
            return _to_node_scalar(d.x)
        return None

    # 1) Try requested field exactly.
    requested = _get(field)
    if requested is not None:
        return requested, field

    # 2) Fallback order for coloring if requested field is missing.
    for alt in ("x", "cp", "pressure", "wall_shear_mag"):
        vals = _get(alt)
        if vals is not None:
            return vals, alt

    return None, None


def _spatial_subsample(pos, n_target):
    """Voxel-grid downsampling for uniform spatial coverage.

    Divides space into voxels via binary search on voxel size to get
    approximately n_target points, then picks one point per voxel.
    O(N log N) per iteration, much faster than FPS for large point counts.
    """
    n = pos.shape[0]
    if n_target >= n:
        return np.arange(n)

    mins = pos.min(axis=0)
    extent = pos.max(axis=0) - mins
    extent = np.maximum(extent, 1e-8)

    vol = float(np.prod(extent))
    vs_est = (vol / n_target) ** (1.0 / 3.0)
    lo, hi = vs_est * 0.01, vs_est * 100.0
    best_idx = None

    for _ in range(25):
        vs = (lo + hi) * 0.5
        grid = ((pos - mins) / vs).astype(np.int64)
        gmax = grid.max(axis=0) + 1
        keys = grid[:, 0] * (gmax[1] * gmax[2]) + grid[:, 1] * gmax[2] + grid[:, 2]
        _, uidx = np.unique(keys, return_index=True)
        best_idx = uidx
        if len(uidx) > n_target * 1.05:
            lo = vs
        elif len(uidx) < n_target * 0.95:
            hi = vs
        else:
            break

    assert best_idx is not None
    if len(best_idx) > n_target:
        rng = np.random.default_rng(42)
        best_idx = rng.choice(best_idx, size=n_target, replace=False)

    return np.sort(best_idx)


def _get_dataset_runs(dataset_path: str) -> list:
    """Get all valid run paths in a dataset, supporting both flat and nested structures.

    Returns list of tuples: (run_name, run_pt_path)
    - Flat: (sample_100001, /path/to/dataset/sample_100001.pt)
    - Nested: (run_1, /path/to/dataset/run_1/data.pt)
    """
    entries = sorted(os.listdir(dataset_path), key=_natural_sort_key)
    runs = []

    # Detect structure
    has_flat = False
    has_nested = False

    for e in entries:
        if e.endswith(".pt") and _is_valid_data_pt(os.path.join(dataset_path, e)):
            has_flat = True
        elif "." not in e and os.path.isfile(os.path.join(dataset_path, e, "data.pt")):
            has_nested = True

    # Collect runs based on detected structure
    for e in entries:
        if has_flat and e.endswith(".pt"):
            if _is_valid_data_pt(os.path.join(dataset_path, e)):
                run_name = e[:-3]  # remove .pt
                runs.append((run_name, os.path.join(dataset_path, e)))
        if has_nested and "." not in e:
            nested_pt = os.path.join(dataset_path, e, "data.pt")
            if os.path.isfile(nested_pt):
                runs.append((e, nested_pt))

    return runs


@app.route("/api/dataset/scan", methods=["POST"])
def api_dataset_scan():
    data = request.get_json(force=True)
    base_dir = data.get("base_dir", "").strip()
    if not base_dir or not os.path.isdir(base_dir):
        return jsonify(ok=False, error=f"Directory not found: {base_dir}")
    try:
        entries = sorted(os.listdir(base_dir))
        datasets = []
        for e in entries:
            full = os.path.join(base_dir, e)
            if not os.path.isdir(full):
                continue
            try:
                runs = _get_dataset_runs(full)
            except OSError:
                runs = []
            if runs:
                datasets.append({"name": e, "path": full, "num_runs": len(runs)})
        return jsonify(ok=True, datasets=datasets)
    except Exception as e:
        return jsonify(ok=False, error=str(e))


@app.route("/api/dataset/runs", methods=["POST"])
def api_dataset_runs():
    data = request.get_json(force=True)
    dataset_path = data.get("dataset_path", "").strip()
    if not dataset_path or not os.path.isdir(dataset_path):
        return jsonify(ok=False, error=f"Directory not found: {dataset_path}")
    try:
        runs = [{"name": run_name, "path": run_pt_path} for run_name, run_pt_path in _get_dataset_runs(dataset_path)]
        return jsonify(ok=True, runs=runs)
    except Exception as e:
        return jsonify(ok=False, error=str(e))


@app.route("/api/dataset/info", methods=["POST"])
def api_dataset_info():
    data = request.get_json(force=True)
    path = data.get("path", "").strip()
    if not path or not os.path.isfile(path):
        return jsonify(ok=False, error=f"File not found: {path}")
    try:
        torch = _ensure_torch()
        d = torch.load(path, weights_only=False, map_location="cpu")
        n_verts = d.pos.shape[0]
        n_faces = d.face.shape[1] if hasattr(d, "face") and d.face is not None else 0
        fields = []
        if hasattr(d, "cp") and d.cp is not None:
            fields.append("cp")
        if hasattr(d, "pressure") and d.pressure is not None:
            fields.append("pressure")
        if hasattr(d, "wall_shear") and d.wall_shear is not None:
            fields.append("wall_shear_mag")
        if hasattr(d, "x") and d.x is not None:
            fields.append("x")
        info = {
            "ok": True,
            "vertices": n_verts,
            "faces": n_faces,
            "fields": fields,
            "filename": getattr(d, "filename", ""),
        }
        if hasattr(d, "conditioning") and d.conditioning is not None:
            info["conditioning"] = d.conditioning.tolist()
        if hasattr(d, "cd") and d.cd is not None:
            info["cd"] = float(d.cd.item()) if d.cd.numel() == 1 else d.cd.tolist()
        return jsonify(info)
    except Exception as e:
        return jsonify(ok=False, error=str(e))


def _load_run_cached(run_pt_path, field):
    """Load pos + field + face from a .pt file, using a binary cache for speed.

    Returns (pos [N,3] float32, scalars [N] float32 or None, face [3,F] int32 or None).
    On first call, creates a compact cache file; subsequent calls load from cache (~100x faster).
    Supports both nested (run_dir/data.pt) and flat (sample_123.pt) structures.
    """
    # Determine structure: if run_pt_path is data.pt in a subdirectory (nested),
    # or a direct .pt file in the dataset (flat)
    run_dir = os.path.dirname(run_pt_path)
    run_name = os.path.basename(run_pt_path)[:-3]  # remove .pt extension

    # For nested structure (run_1/data.pt), ds_dir is parent of run_dir
    # For flat structure (sample_123.pt in dataset), ds_dir is run_dir
    if os.path.basename(run_pt_path) == "data.pt":
        # Nested structure: data.pt in subdirectory
        ds_dir = os.path.dirname(run_dir)
        run_name = os.path.basename(run_dir)
    else:
        # Flat structure: direct .pt file in dataset directory
        ds_dir = run_dir

    cache_dir = os.path.join(ds_dir, ".gui_cache", "runs")
    cache_file = os.path.join(cache_dir, f"{run_name}_{field or 'none'}.bin")

    if os.path.isfile(cache_file):
        with open(cache_file, "rb") as fh:
            header = fh.read(10)  # II + H = 4+4+2
            n, nf = struct.unpack_from("<II", header, 0)
            has_s = struct.unpack_from("<H", header, 8)[0]
            # .copy() because frombuffer returns read-only views
            pos = np.frombuffer(fh.read(n * 12), dtype=np.float32).reshape(n, 3).copy()
            face = np.frombuffer(fh.read(nf * 12), dtype=np.int32).reshape(nf, 3).T.copy() if nf > 0 else None
            scalars = np.frombuffer(fh.read(n * 4), dtype=np.float32).copy() if has_s else None
        # Recover from stale no-scalar caches by reloading source and applying fallback.
        if scalars is not None:
            return pos, scalars, face

    torch = _ensure_torch()
    d = torch.load(run_pt_path, weights_only=False, map_location="cpu")
    pos = d.pos.numpy().astype(np.float32)
    n = pos.shape[0]

    scalars, _effective_field = _extract_scalars(d, field)

    face = None
    if hasattr(d, "face") and d.face is not None:
        face = d.face.numpy().astype(np.int32)  # [3, F] int32 (saves ~50% vs int64)

    del d

    # Write cache
    try:
        os.makedirs(cache_dir, exist_ok=True)
        nf = face.shape[1] if face is not None else 0
        has_s = 1 if scalars is not None else 0
        with open(cache_file, "wb") as fh:
            fh.write(struct.pack("<IIH", n, nf, has_s))
            fh.write(pos.tobytes())
            if face is not None:
                fh.write(np.ascontiguousarray(face.T).tobytes())  # write as [F, 3]
            if scalars is not None:
                fh.write(scalars.tobytes())
    except OSError:
        pass

    return pos, scalars, face


@app.route("/api/dataset/load", methods=["POST"])
def api_dataset_load():
    data = request.get_json(force=True)
    path = data.get("path", "").strip()
    downsample = int(data.get("downsample", 0))
    field = data.get("field", "")
    scalar_range = data.get("scalar_range")  # [min, max] or null for auto
    if not path or not os.path.isfile(path):
        return jsonify(ok=False, error=f"File not found: {path}")
    try:
        pos, scalars, face_raw = _load_run_cached(path, field)
        n_verts = pos.shape[0]

        # Downsampling: voxel-grid for good spatial coverage (render as points)
        faces = None
        if downsample > 0 and downsample < n_verts:
            indices = _spatial_subsample(pos, downsample)
            pos = pos[indices]
            if scalars is not None:
                scalars = scalars[indices]
            n_verts = pos.shape[0]  # actual count (voxel-grid may differ slightly)
        else:
            # Full resolution: include faces
            faces = face_raw

        mins = pos.min(axis=0)
        maxs = pos.max(axis=0)
        extents = (maxs - mins).astype(np.float32)
        center = ((mins + maxs) / 2).astype(np.float32)

        n_faces = faces.shape[1] if faces is not None else 0
        has_scalar = 1 if scalars is not None else 0
        if scalars is not None:
            if scalar_range:
                s_min, s_max = float(scalar_range[0]), float(scalar_range[1])
            else:
                s_min, s_max = float(scalars.min()), float(scalars.max())
        else:
            s_min = s_max = 0.0

        parts = [
            struct.pack(
                "<II6fIff",
                n_verts,
                n_faces,
                *extents.tolist(),
                *center.tolist(),
                has_scalar,
                s_min,
                s_max,
            )
        ]
        parts.append(pos.astype(np.float32).tobytes())
        if faces is not None:
            parts.append(faces.T.astype(np.uint32).tobytes())
        if scalars is not None:
            parts.append(scalars.astype(np.float32).tobytes())
        return Response(b"".join(parts), mimetype="application/octet-stream")
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=str(e))


def _cache_worker(args):
    """Worker for parallel cache generation. Must be top-level for pickling."""
    run_pt_path, field = args
    # Extract run name: for nested (run_1/data.pt) use dir name, for flat (sample_123.pt) use file name
    if os.path.basename(run_pt_path) == "data.pt":
        run_name = os.path.basename(os.path.dirname(run_pt_path))
    else:
        run_name = os.path.basename(run_pt_path)[:-3]  # remove .pt
    try:
        _load_run_cached(run_pt_path, field)
        return (run_name, True, None)
    except Exception as e:
        return (run_name, False, str(e))


def _thumb_worker(args):
    """Worker for parallel thumbnail generation. Must be top-level for pickling."""
    run_pt_path, field, thumb_points = args
    # Extract run name: for nested (run_1/data.pt) use dir name, for flat (sample_123.pt) use file name
    if os.path.basename(run_pt_path) == "data.pt":
        run_name = os.path.basename(os.path.dirname(run_pt_path))
    else:
        run_name = os.path.basename(run_pt_path)[:-3]  # remove .pt
    try:
        pos, scalars, _ = _load_run_cached(run_pt_path, field)
        n = pos.shape[0]
        if thumb_points < n:
            idx = _spatial_subsample(pos, thumb_points)
            pos = pos[idx]
            if scalars is not None:
                scalars = scalars[idx]
            n = pos.shape[0]
        return (run_name, n, pos.astype(np.float32), scalars.astype(np.float32) if scalars is not None else None, None)
    except Exception as e:
        return (run_name, 0, None, None, str(e))


@app.route("/api/dataset/cache_status", methods=["POST"])
def api_dataset_cache_status():
    """Return how many runs have binary caches for the given field."""
    data = request.get_json(force=True)
    dataset_path = data.get("dataset_path", "").strip()
    field = data.get("field", "cp")
    if not dataset_path or not os.path.isdir(dataset_path):
        return jsonify(ok=False, error=f"Not found: {dataset_path}")
    runs = _get_dataset_runs(dataset_path)
    cache_dir = os.path.join(dataset_path, ".gui_cache", "runs")
    cached = sum(1 for run_name, _ in runs if os.path.isfile(os.path.join(cache_dir, f"{run_name}_{field or 'none'}.bin")))
    return jsonify(ok=True, total=len(runs), cached=cached, missing=len(runs) - cached)


@app.route("/api/dataset/generate_cache", methods=["POST"])
def api_dataset_generate_cache():
    """Start async cache generation; returns job_id for polling."""
    data = request.get_json(force=True)
    dataset_path = data.get("dataset_path", "").strip()
    field = data.get("field", "cp")
    if not dataset_path or not os.path.isdir(dataset_path):
        return jsonify(ok=False, error=f"Not found: {dataset_path}")

    try:
        runs = _get_dataset_runs(dataset_path)
        tasks = []
        skipped = 0
        cache_dir = os.path.join(dataset_path, ".gui_cache", "runs")
        for run_name, pt in runs:
            cache_file = os.path.join(cache_dir, f"{run_name}_{field or 'none'}.bin")
            if os.path.isfile(cache_file):
                skipped += 1
                continue
            if os.path.isfile(pt):
                tasks.append((pt, field))

        if not tasks:
            return jsonify(
                ok=True,
                job_id=None,
                total=len(runs),
                skipped=skipped,
                done=0,
                finished=True,
                message=f"All {skipped} runs already cached",
            )

        job_id = str(uuid.uuid4())
        _cache_jobs[job_id] = {
            "done": 0,
            "total": len(tasks),
            "skipped": skipped,
            "errors": 0,
            "finished": False,
            "error_details": [],
        }

        def _run_job(jid, tks):
            try:
                with Pool(processes=min(8, len(tks))) as pool:
                    for run_name, ok, err in pool.imap_unordered(_cache_worker, tks):
                        _cache_jobs[jid]["done"] += 1
                        if not ok:
                            _cache_jobs[jid]["errors"] += 1
                            _cache_jobs[jid]["error_details"].append(f"{run_name}: {err}")
            finally:
                _cache_jobs[jid]["finished"] = True

        threading.Thread(target=_run_job, args=(job_id, tasks), daemon=True).start()
        return jsonify(ok=True, job_id=job_id, total=len(tasks) + skipped, skipped=skipped)
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=str(e))


@app.route("/api/dataset/cache_progress/<job_id>")
def api_dataset_cache_progress(job_id):
    """Poll progress for a cache generation job."""
    job = _cache_jobs.get(job_id)
    if not job:
        return jsonify(ok=False, error="Unknown job")
    return jsonify(ok=True, **job)


@app.route("/api/dataset/field_range", methods=["POST"])
def api_dataset_field_range():
    """Compute global min/max for a scalar field across all runs, with caching."""
    data = request.get_json(force=True)
    dataset_path = data.get("dataset_path", "").strip()
    field = data.get("field", "cp")
    if not dataset_path or not os.path.isdir(dataset_path):
        return jsonify(ok=False, error=f"Not found: {dataset_path}")

    cache_dir = os.path.join(dataset_path, ".gui_cache")
    cache_file = os.path.join(cache_dir, f"field_range_{field}_{_GUI_CACHE_VERSION}.json")
    if os.path.isfile(cache_file):
        with open(cache_file) as f:
            cached = json_mod.load(f)
        # Invalidate old caches that used absolute min/max (no 'percentile' key)
        if "percentile" in cached:
            return jsonify(ok=True, **cached)

    try:
        global_min, global_max = float("inf"), float("-inf")
        count = 0
        for _run_name, pt in _get_dataset_runs(dataset_path):
            _, scalars, _ = _load_run_cached(pt, field)
            if scalars is None:
                continue
            # Use 1st/99th percentile per run to reject extreme outliers
            mn = float(np.percentile(scalars, 1.0))
            mx = float(np.percentile(scalars, 99.0))
            global_min = min(global_min, mn)
            global_max = max(global_max, mx)
            count += 1

        if count == 0:
            return jsonify(ok=False, error=f"No runs found with field '{field}'")

        result = {"field": field, "min": global_min, "max": global_max, "num_runs": count, "percentile": [1.0, 99.0]}
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "w") as f:
                json_mod.dump(result, f)
        except OSError:
            pass
        return jsonify(ok=True, **result)
    except Exception as e:
        return jsonify(ok=False, error=str(e))


@app.route("/api/dataset/thumbnails", methods=["POST"])
def api_dataset_thumbnails():
    """Return lightweight point cloud data for all runs (for thumbnail rendering).

    Returns JSON with per-run: positions (downsampled to thumb_points), cp scalars, bounds.
    Cached to disk for fast reload.
    """
    data = request.get_json(force=True)
    dataset_path = data.get("dataset_path", "").strip()
    field = data.get("field", "cp")
    thumb_points = int(data.get("thumb_points", 3000))
    if not dataset_path or not os.path.isdir(dataset_path):
        return jsonify(ok=False, error=f"Not found: {dataset_path}")

    cache_dir = os.path.join(dataset_path, ".gui_cache")
    cache_file = os.path.join(cache_dir, f"thumbs_{field}_{thumb_points}_{_GUI_CACHE_VERSION}.bin")
    if os.path.isfile(cache_file):
        with open(cache_file, "rb") as f:
            return Response(f.read(), mimetype="application/octet-stream")

    try:
        tasks = [(run_pt_path, field, thumb_points) for _, run_pt_path in _get_dataset_runs(dataset_path)]

        if not tasks:
            return jsonify(ok=False, error="No valid dataset .pt files found")

        with Pool(processes=min(8, len(tasks))) as pool:
            results = pool.map(_thumb_worker, tasks)

        run_datas = []
        for run_name, n, pos, scalars, err in results:
            if err is not None:
                continue  # skip failed runs silently
            run_datas.append({"name": run_name, "n": n, "pos": pos, "scalars": scalars})

        # Binary format:
        # uint32 num_runs
        # per run: uint16 name_len, name_bytes, uint32 n_points, uint8 has_scalar,
        #          float32[n*3] positions, float32[n] scalars (if has_scalar)
        parts = [struct.pack("<I", len(run_datas))]
        for rd in run_datas:
            name_bytes = rd["name"].encode("utf-8")
            has_s = 1 if rd["scalars"] is not None else 0
            parts.append(struct.pack("<H", len(name_bytes)))
            parts.append(name_bytes)
            parts.append(struct.pack("<IB", rd["n"], has_s))
            parts.append(rd["pos"].tobytes())
            if rd["scalars"] is not None:
                parts.append(rd["scalars"].tobytes())

        blob = b"".join(parts)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "wb") as f:
                f.write(blob)
        except OSError:
            pass
        return Response(blob, mimetype="application/octet-stream")
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset curation / export
# ─────────────────────────────────────────────────────────────────────────────


@app.route("/api/dataset/export_curated", methods=["POST"])
def api_dataset_export_curated():  # noqa: PLR0912
    """Copy a curated subset of runs to a new dataset directory.

    Copies run subdirs, norm_stats .pt files, and writes a new order_.txt.
    Also moves matching .gui_cache/runs/*.bin entries so the cache is preserved.
    """
    data = request.get_json(force=True)
    src_path = data.get("src_path", "").strip()
    dst_path = data.get("dst_path", "").strip()
    keep_runs = data.get("keep_runs", [])  # list of run names
    reject_runs = data.get("reject_runs", [])

    if not src_path or not os.path.isdir(src_path):
        return jsonify(ok=False, error=f"Source not found: {src_path}")
    if not dst_path:
        return jsonify(ok=False, error="Destination path required")
    if not keep_runs:
        return jsonify(ok=False, error="No runs marked as keep")
    if os.path.exists(dst_path):
        return jsonify(ok=False, error=f"Destination already exists: {dst_path}")

    try:
        os.makedirs(dst_path)

        # Copy run directories
        copied = []
        for run_name in keep_runs:
            src_run = os.path.join(src_path, run_name)
            dst_run = os.path.join(dst_path, run_name)
            if not os.path.isdir(src_run):
                continue
            shutil.copytree(src_run, dst_run)
            copied.append(run_name)

        # Write order_.txt  (same name as training code expects)
        with open(os.path.join(dst_path, "order_.txt"), "w") as f:
            f.writelines(name + "\n" for name in copied)

        # Write rejected list for reference
        with open(os.path.join(dst_path, "rejected_runs.txt"), "w") as f:
            f.writelines(name + "\n" for name in reject_runs)

        # Copy norm_stats .pt files (keep global statistics intact)
        for fname in os.listdir(src_path):
            if fname.endswith("norm_stats.pt"):
                shutil.copy2(os.path.join(src_path, fname), os.path.join(dst_path, fname))

        # Copy GUI cache entries for kept runs
        src_cache = os.path.join(src_path, ".gui_cache", "runs")
        dst_cache = os.path.join(dst_path, ".gui_cache", "runs")
        if os.path.isdir(src_cache):
            os.makedirs(dst_cache, exist_ok=True)
            for fname in os.listdir(src_cache):
                run_name = fname.split("_cp.bin")[0].split("_none.bin")[0]
                # Match any cache file whose prefix is a kept run name
                for kept in copied:
                    if fname.startswith(kept + "_"):
                        shutil.copy2(os.path.join(src_cache, fname), os.path.join(dst_cache, fname))
                        break

        return jsonify(ok=True, copied=len(copied), dst=dst_path, message=f"Created {dst_path} with {len(copied)} runs")
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=str(e))


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
