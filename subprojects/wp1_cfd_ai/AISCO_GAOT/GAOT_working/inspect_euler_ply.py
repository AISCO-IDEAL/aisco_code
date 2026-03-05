"""Show coordinate dimensions of all Euler PLY files and save to a text file."""

import glob
import os

import numpy as np

try:
    import pyvista as pv

    def read_ply(f):
        return np.array(pv.read(f).points)
except ImportError:
    from plyfile import PlyData

    def read_ply(f):
        d = PlyData.read(f)["vertex"]
        return np.stack([d["x"], d["y"], d["z"]], axis=1)


GEO_DIR = "/cluster/work/fuge/AISCO/wp1_cfd_ai/FLUENT/GEOMETRY"
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "euler_ply_dimensions.txt")

plys = sorted(glob.glob(os.path.join(GEO_DIR, "RUN_*.ply")))

header = f"{'Run':<10} {'x_min':>8} {'x_max':>8} {'y_min':>8} {'y_max':>8} {'z_min':>8} {'z_max':>8} {'x_span':>8} {'npts':>7}"
separator = "-" * 82

lines = [f"Found {len(plys)} PLY files", "", header, separator]

for fp in plys:
    name = os.path.basename(fp).replace(".ply", "")
    pts = read_ply(fp)
    xmn, xmx = pts[:, 0].min(), pts[:, 0].max()
    ymn, ymx = pts[:, 1].min(), pts[:, 1].max()
    zmn, zmx = pts[:, 2].min(), pts[:, 2].max()
    line = (
        f"{name:<10} {xmn:>8.1f} {xmx:>8.1f} {ymn:>8.1f} {ymx:>8.1f} {zmn:>8.1f} {zmx:>8.1f} {xmx - xmn:>8.1f} {pts.shape[0]:>7d}"
    )
    lines.append(line)
    print(line, flush=True)

output = "\n".join(lines)
with open(OUT_FILE, "w") as f:
    f.write(output + "\n")

print(f"\nSaved to {OUT_FILE}")
