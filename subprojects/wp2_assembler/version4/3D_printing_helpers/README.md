# 3D Printing Helpers

Post-processing tools for making printable artifacts from existing AISCO run
exports. These helpers do not modify the optimizer, assembler, or existing
export code.

## Honesty Rules

- Use existing exported meshes as the source of truth.
- Use current WP2 v4 configs only for deterministic repair/add-on geometry.
- Do not invent solar placement or masks. If no real solar artifact is found,
  the report says it is missing and no solar region is exported.
- Vision and overlap regions are computed only from real geometry. If the local
  mesh stack cannot perform the query, the report says so.

## Main Command

Run from the repository root, inside an environment with `cadquery`, `trimesh`,
`numpy`, and `pyyaml` installed:

```bash
python subprojects/wp2_assembler/version4/3D_printing_helpers/export_printables.py
```

The helper reads `print_config.yaml`, asks for missing values in the terminal,
and shows the final settings before exporting. You can either edit the YAML once
or type values interactively when prompted. The default config points at the
example `p2clwhx6_66994778_dummy` run and writes to its `printables_3d` folder.

All exported geometry is scaled by one constant `scale_factor`. The default is
`150 / 5800 = 0.02586206896551724`, so a full 5800 mm WP2 car-length domain
exports at about 150 mm.

The `epoch` setting controls the single mesh to export. Leave it blank or set it
to `latest` for the last available epoch, or set an integer like `1000` to export
only that epoch.

The `solar_source` setting chooses which real `solar/<source>/` PLY to include.
The default is `selection`, which is the actual saved solar placement/selection
export. Other useful options include `quality`, `penalty`, `shading`, and
`receiver_mask` when those folders exist.

Optional CLI overrides are still available for batch use:

```bash
python subprojects/wp2_assembler/version4/3D_printing_helpers/export_printables.py \
  --run-dir /cluster/work/fuge/AISCO/temp_/6166b4e953/p2clwhx6_66994778_dummy \
  --assembly-slot 0
```

Outputs are written to `<run-dir>/printables_3d/` unless `out_dir` is set in the
YAML or passed on the command line. The folder is split by print intent:

- `multicolor/` contains one STL per material/color: `mesh.stl`, `inner_assembly.stl`, `solar_placement.stl`, and `vision_cone_intersection.stl` when the corresponding real mask exists.
- `single_color/` contains one-material STL exports such as `removable_shell_source.stl`, `support_plate.stl`, `inner_assembly_full_wheels.stl`, `wheel_repair_full_wheels.stl`, `seat_support.stl`, and `shell_socket_cutters.stl`.
- `print_report.json` stays at the root of `printables_3d/`.

If a boolean engine is available through `trimesh`, the helper also attempts to
write `removable_shell_with_sockets.stl`. If the boolean fails, it keeps the
source shell and socket cutter geometry separate instead of silently faking the
result.


## Assembly Source

The helper does not use `assembly_id`. It reads the exported assembly meshes in
`mesh_assembly_v4`, for example `epoch_01000/slot_00_mesh.ply`, infers wheel
centers from the mesh geometry, adds full regulation wheels at those positions,
and builds the plate/support from the assembly bounds. Use `assembly_slot` to
choose which `slot_XX_mesh` file to use.
