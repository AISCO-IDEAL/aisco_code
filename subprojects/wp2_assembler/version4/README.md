# WP2 Assembler Version 4

`version4` builds no-build-zone geometry for WP2 solar car assemblies.

It uses:

- `config/regulations_and_parameters.yaml` for all fixed dimensions and rules
- `config/assembly_map.yaml` for precomputed seat and wheel positions

It now includes a driver vision box in the assembled output.

## Main Files

- `components/nbz.py`: main entry point for loading an assembly
- `components/inner/seat.py`: seat NBZ geometry
- `components/inner/wheels.py`: wheel geometry
- `components/outer/outer_nbz.py`: outer NBZ geometry
- `components/vision/`: driver vision box and tracker box geometry
- `utils/assemblies_maker.py`: generation and filtering of valid assembly maps

## Main API

```python
from subprojects.wp2_assembler.version4.components.nbz import (
    call_assembly,
    bounding_box_of_assembly,
    max_assembly_id,
)
```

- `call_assembly(id)`: returns inner geometry, outer NBZ, and vision geometry
- `bounding_box_of_assembly(id)`: returns the inner assembly bounding box
- `max_assembly_id()`: returns the highest available ID

## Example

```python
from subprojects.wp2_assembler.version4.components.nbz import call_assembly

inner, outer, vision = call_assembly(0)
```

## Notes

- `main.py` is currently just a placeholder.
- Most geometry is driven directly from the YAML config files.
- The driver vision box is defined in `components/vision/vision_box.py` and placed relative to the seat using `no_build_zones.vision_box.eye_level_z`.
- `utils/assemblies_maker.py` is used to generate and filter valid assemblies before saving them into `assembly_map.yaml`.
