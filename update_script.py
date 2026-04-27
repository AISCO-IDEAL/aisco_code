import sys

file_path = "subprojects/wp4_aeroshell_optimizer/v2/src/main_files/pretraining_latent.py"
with open(file_path) as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)
    if "raw_mesh = trimesh.load(ply_path)" in line:
        indent = line[: line.find("raw_mesh")]
        new_lines.append(f"{indent}if not raw_mesh.is_watertight:\n")
        new_lines.append(f"{indent}    raise ValueError(f\"Mesh loaded from '{{ply_path}}' is not watertight.\")\n")

with open(file_path, "w") as f:
    f.writelines(new_lines)
