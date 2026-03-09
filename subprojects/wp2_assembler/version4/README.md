# Assembler v2.0 - No-Build Zone Generator

Second version of the AISCO Assembler. Generates inner and outer no-build zones (NBZ) for solar car assemblies based on regulation constraints. Assemblies can be generated randomly or retrieved deterministically using an ID number mapped in `assembly_map.yaml`.

## What's New in v2

- Parametric no-build zone generation for solar car design
- Inner NBZ: triangle-based wheelbase with wheels and seat constraints
- Outer NBZ: regulatory bounding box dimensions
- Deterministic assembly retrieval via ID mapping
- Random assembly generation with configurable distributions
- CadQuery-based 3D visualization

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- CadQuery and OCP-vscode for visualization
- pip package manager

### Installation

1. **Navigate to the project root:**
```powershell
cd AISCO_code
```

2. **Install the AISCO package:**
```powershell
pip install -e .
```

   This will install required dependencies including:
   - `cadquery` - 3D CAD modeling
   - `pyyaml` - Configuration file parsing
   - `numpy` - Numerical operations
   - `scipy` - Statistical distributions
   - `ocp-vscode` - CadQuery visualization

3. **Verify installation:**
```powershell
python -c "from subprojects.wp2_assembler.version2.components import nbz; print('Setup successful!')"
```

## Usage

### Running the Visualizer

```powershell
cd subprojects/wp2_assembler/version2
python main.py
```

Interactive options:
- `i` - Visualize inner NBZ only
- `o` - Visualize outer NBZ only
- `b` - Visualize both (inner + outer)
- `m` - Visualize multiple assemblies (range)
- `g` - Generate new random assemblies
- `s` - Stop/exit

### Programmatic Usage

```python
from subprojects.wp2_assembler.version2.components import nbz
from subprojects.wp2_assembler.version2.utils.reg_imp import get_regulation

# Generate or retrieve an assembly by ID
inner_nbz, outer_nbz, vision_cones = nbz.call_assembly_solids(id_number=1)

# Access regulation values
tire_width = get_regulation("wheel.tire_width")  # Returns 96
seat_width = get_regulation("seat.dimension.width")  # Returns 450.0

# Get the smallest bounding box containing the inner nbz
bounding_box_coordinates = bounding_box_of_assembly(id_number=1) # returns tuple with two tuples: ((x_min, y_min, z_min),(x_max,y_max,z_max))

# Visualize the bounding box
bounding_box_solid = create_bb_visualization(coordinates = bounding_box_coordinates) # returns a solid of the bounding box, with the correct positioning and size.

```

## Project Structure

```
version2/
├── README.md                    # This file
├── main.py                      # Interactive visualization entry point
├── __init__.py
│
├── components/                  # NBZ component generation
│   ├── nbz.py                  # Main assembly orchestration
│   ├── inner/                  # Inner NBZ components
│   │   ├── base_triangle.py   # Wheelbase triangle geometry
│   │   ├── wheels.py          # Wheel generation
│   │   └── seat.py            # Seat NBZ generation
│   └── outer/
│       └── outer_box.py       # Outer bounding box
│
├── utils/                       # Helper utilities
│   ├── distributions.py        # Random distribution functions
│   └── reg_imp.py             # Regulation import utilities
│
└── config/                      # Configuration files
    ├── example_regulations.yaml # Competition regulations
    └── assembly_map.yaml       # ID-to-parameters mapping
```

## Configuration

### Regulations (`config/example_regulations.yaml`)

Key parameters from World Solar Challenge regulations:

- **No-Build Zones (NBZ)**:
  - Outer prism: 5800mm (L) × 2300mm (W) × 1650mm (H)
  - Front/rear overhang: 60% of wheelbase
  - Road apex incline: 15°

- **Wheel specifications**:
  - Minimum count: 3
  - Tire diameter: 557mm (Silvretta)
  - Tire width: 96mm

- **Seat dimensions**:
  - Width: 450mm
  - Backrest angles: -27° to +45°
  - Legroom: 1000mm × 350mm
  - Headroom: 280mm × 300mm

### Assembly Mapping (`config/assembly_map.yaml`)

Each ID maps to specific assembly parameters:
```yaml
1:
  wheel_base_width: 1000
  wheel_base_length: 2500
  tyre_diameter: 596
  seat_position: 400
```

Access regulations programmatically:
```python
from subprojects.wp2_assembler.version2.utils.reg_imp import get_regulation

tire_diameter = get_regulation("wheel.tire_diameter")
```

## Architecture

### Components (`components/`)
- **Purpose**: Domain-specific NBZ generation logic
- **`nbz.py`**: Orchestrates assembly creation and persistence
- **`inner/`**: Inner NBZ components (triangle, wheels, seat)
- **`outer/`**: Outer bounding box generation

### Utils (`utils/`)
- **Purpose**: Generic reusable functions
- **`distributions.py`**: Truncated gamma and uniform distributions
- **`reg_imp.py`**: YAML regulation parsing utilities

### Design Principles
1. Separation of concerns: components vs. utilities
2. Regulation-driven parametric design
3. Deterministic reproducibility via ID mapping
4. Type-safe with Python type hints

## Technical Details

### Inner NBZ Generation
1. Creates triangular wheelbase using `Triangle` class
2. Places wheels at triangle vertices with regulation margins
3. Positions seat based on wheelbase and clearance rules
4. Validates geometric constraints

### Outer NBZ Generation
- Simple box primitive based on competition outer dimensions
- Centered at origin with configurable length/width/height

### Random Assembly Generation
- Uses truncated gamma distribution for wheelbase dimensions
- Uniform distribution for seat positioning
- Validates against regulation constraints
- Saves to `assembly_map.yaml` with incremented ID

## Troubleshooting

### Import Errors
Ensure you installed the package from the project root:
```powershell
cd AISCO_code
pip install -e .
```

### YAML Loading Issues
Verify `config/example_regulations.yaml` exists and contains valid YAML syntax.

### CadQuery Visualization
Make sure `ocp-vscode` extension is installed in VS Code for 3D visualization.

## Status

This version (v3) is now **frozen** and will not receive further modifications.

## References

- Competition: World Solar Challenge
- Class: Cruiser
- Regulations: `config/example_regulations.yaml`
