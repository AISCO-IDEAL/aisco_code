# WP2 Assembler Version 4

`version4` builds no-build-zone geometry for WP2 solar car assemblies.

It uses:

- `config/regulations_and_parameters.yaml` for fixed dimensions, limits, and generation ranges
- `config/assembly_map.yaml` for precomputed seat and wheel placements

The assembled output also includes the driver vision box.

## Main Files

- `components/nbz.py`: main entry point for loading and post-processing an assembly
- `components/inner/seat.py`: seat NBZ geometry
- `components/inner/wheels.py`: wheel geometry
- `components/outer/outer_nbz.py`: outer NBZ geometry
- `components/vision/`: driver vision box and tracker box geometry
- `utils/assemblies_maker.py`: generation and filtering of valid assembly maps

## Main API

```python
from subprojects.wp2_assembler.version4.components.nbz import (
    call_assembly,
    call_assembly_solids,
    bounding_box_of_assembly,
    max_assembly_id,
)
```

- `call_assembly(id)`: returns a tuple of (inner assembly, outer solid, vision assembly)
- `call_assembly_solids(id)`: returns a tuple of (list of inner solids, outer solid) - useful for applications that need solids instead of assemblies
- `bounding_box_of_assembly(id)`: returns the bounding box of the inner assembly after processing
- `max_assembly_id()`: returns the highest available assembly ID

## `call_assembly` behavior

`call_assembly(id)` no longer returns the raw assembly. It returns a cut version that is ready to use with the ground-clearance constraints:

- The outer NBZ is cut at `no_build_zones.outer_prism.ground_clearance`
- The wheel solids are cut `1 mm` higher than that plane, so there is clearance for the aeroshell
- The seat is not cut
- If the seat would intersect that cut plane, the function raises an exception instead of trimming it

This keeps the seat geometry intact while enforcing that every returned assembly has enough space below the seat and around the wheels.

## Examples

```python
from subprojects.wp2_assembler.version4.components.nbz import call_assembly, call_assembly_solids

# Get assemblies as CadQuery Assembly objects
inner_assembly, outer_solid, vision_assembly = call_assembly(0)

# Get just the solids for visualization or analysis
inner_solids, outer_solid = call_assembly_solids(0)
```

## Configuration Files

### `config/regulations_and_parameters.yaml`

This file contains the fixed dimensions and rules used to build the geometry:

- `no_build_zones.outer_prism`: global outer-envelope dimensions and `ground_clearance`
- `no_build_zones.wheel_based_rules`: overhang and angle constraints used for the outer NBZ
- `no_build_zones.vision_box`: parameters for the driver vision volume, including `eye_level_z`
- `tracker_box`: tracker box dimensions and cone definition
- `seat.dimension`: seat geometry parameters
- `seat.combinations`: the grid of seat positions used when generating possible assemblies
- `wheel`: tire dimensions
- `wheel.combinations`: the grid of wheel positions, plus the allowed wheel counts used during generation

In practice, this is the main file to edit when a regulation changes or when you want to change the parameter space used to generate assemblies.

### `config/assembly_map.yaml`

This file stores the generated assembly library. Each entry contains:

- the assembly ID
- one seat position
- one position for each wheel

`call_assembly(id)` reads this file to reconstruct a specific assembly. `utils/assemblies_maker.py` is responsible for generating, validating, and saving these entries.

## Notes

- `main.py` is currently just a placeholder.
- Most geometry is driven directly from the YAML config files.
- The driver vision box is defined in `components/vision/vision_box.py` and placed relative to the seat using `no_build_zones.vision_box.eye_level_z`.
- `utils/assemblies_maker.py` is used to generate and filter valid assemblies before saving them into `assembly_map.yaml`.
