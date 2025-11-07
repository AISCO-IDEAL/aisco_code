# WP3 Solar Estimator

## Overview
The Solar Estimator evaluates solar deck concepts for the aCentauri race car. Starting from a 3D mesh (`.stl` or `.ply`), it extracts the upward-facing surface, computes illumination angles for a given sun direction, applies incidence-angle modifiers, and selects the best-performing surface within a configurable area limit. The toolkit covers one-off analyses, full-route simulations, and batch processing for entire design iterations.

## Requirements
- Python 3.10 or later (tested on macOS/Linux).
- Python packages: `numpy`, `pandas`, `trimesh`, `pvlib`, `joblib`, `tqdm`, `pytz`, `matplotlib`.
- Install from the project root (`AISCO_code`):

```bash
pip install -e .
```

Alternatively install packages manually:

```bash
pip install numpy pandas trimesh pvlib joblib tqdm pytz matplotlib
```

## Typical Workflow
1. Export the vehicle mesh and place it in `data/stl_models/`.
2. Define a race route CSV with columns `Lat,Long,altitude,time` in `data/race_routes/`.
3. Run a simulation:
   - Quick exploration via `main.ipynb` or `pv_index_simulator.ipynb`.
   - Programmatic run with `simulate_stl.py` or `run_parallel_simulations`.
4. Inspect the outputs:
   - CSV summaries in `data/simulation_results.csv`.
   - Selected solar decks (STL) in `data/selected_solardecks/`.
   - Optional colouring with `helpers/visualize_weight.py`.

## File and Folder Guide

### Python Modules
| Path | Purpose |
| --- | --- |
| `__init__.py` | Marks the directory as a Python package for imports like `subprojects.wp3_solar_estimator.read_mesh`. |
| `read_mesh.py` | Mesh utilities: face-to-sun angle (`facenormal_angle`), incidence-angle modifier (`powerratio_iam`), upward-face filter (`sky_face`), and a ray-based shadow indicator (`shadowing_factor`). |
| `sun.py` | Converts latitude, longitude, altitude, and UTC time to a normalized sun vector using `pvlib` (`get_sun_direction`). |
| `pv_index.py` | Single-snapshot scoring. `pv_perf_index` isolates sky-facing faces, weights them, sorts by performance, and clips to the requested area, returning an index (0–1) plus the selected mesh. |
| `pv_index_for_run.py` | Aggregates multiple timesteps (e.g., an entire race). Accepts a `top_mesh` and a DataFrame of per-timestep weights, averages them, and applies the same selection logic as `pv_index.py`. |
| `simulate_stl.py` | Core automation script. Loads meshes and routes, computes sun vectors for each timestamp, builds the weight matrix, calls `pv_index_for_run.pv_perf_index`, and stores results. Includes helpers `process_stl_file` (single file) and `run_parallel_simulations` (batch processing with `joblib`). |

### Notebooks
| Notebook | Description |
| --- | --- |
| `main.ipynb` | Legacy scratchpad for quick manual checks of the angle and performance functions. |
| `pv_index_simulator.ipynb` | Guided workflow with markdown explanations: set parameters, load meshes, run a single simulation, inspect outputs. |
| `run_race.ipynb` | Demonstrates a full race-day pipeline from mesh loading to aggregated performance selection. |
| `helpers/generate_car.ipynb` | Generates simple test geometries (boxes, etc.) to validate the pipeline. |
| `helpers/mesh_division.ipynb` | Experiments with alternative mesh subdivision strategies. |

### Helper Scripts
| Path | Description |
| --- | --- |
| `helpers/visualize_weight.py` | Provides `color_faces_by_weight` to map per-face weights to RGBA colours (Matplotlib colormap + gamma correction), useful for qualitative result checks. |

### Data Layout
| Path | Contents |
| --- | --- |
| `data/race_routes/` | Example route CSVs (e.g., `points.csv`) with latitude, longitude, altitude, and local timestamps. |
| `data/stl_models/` | Reference meshes, including the `outer_silvretta_shells/` design iterations. |
| `data/ply_models/` | Sample geometries in PLY format (`SilvrettaAeroskin.ply`). |
| `data/selected_solardecks/` | Auto-generated solar deck selections (`solardeck_*.stl`). Created on demand. |
| `data/simulation_results.csv` | Aggregated batch results: filename, performance index, runtime. |

## Usage Examples

### Single Simulation (Python)
```python
from subprojects.wp3_solar_estimator.simulate_stl import simulate_stl

performance, deck = simulate_stl(
    "subprojects/wp3_solar_estimator/data/race_routes/points.csv",
    "subprojects/wp3_solar_estimator/data/stl_models/car.stl",
    max_area=6.0,
)

print(f"Performance index: {performance:.3f}")
deck.export("solardeck_car.stl")
```

### Batch Run Over a Folder
```python
from subprojects.wp3_solar_estimator.simulate_stl import run_parallel_simulations

results = run_parallel_simulations(
    stl_folder="subprojects/wp3_solar_estimator/data/stl_models/outer_silvretta_shells",
    path_csv="subprojects/wp3_solar_estimator/data/race_routes/points.csv",
    output_csv="subprojects/wp3_solar_estimator/data/simulation_results.csv",
    max_area=6.0,
    n_jobs=8,
)

print(results.head())
```

### Visualising Face Weights
```python
import numpy as np
import trimesh
from subprojects.wp3_solar_estimator.read_mesh import sky_face, facenormal_angle
from subprojects.wp3_solar_estimator.sun import get_sun_direction
from subprojects.wp3_solar_estimator.helpers.visualize_weight import color_faces_by_weight

mesh = trimesh.load_mesh("subprojects/wp3_solar_estimator/data/stl_models/car.stl")
top_mesh = sky_face(mesh)
sun_dir = get_sun_direction(-12.45, 130.83, 100, np.datetime64("2025-09-15T10:00:00"))
angles = facenormal_angle(top_mesh, sun_dir)
weights = np.cos(np.radians(angles))
coloured = color_faces_by_weight(top_mesh.copy(), weights, gamma=1.5)
coloured.show()
```

## Notes and Best Practices
- Route timestamps should be provided in local time; `simulate_stl.py` converts them to UTC internally (example uses `Australia/Darwin`).
- The incidence-angle modifier defaults to `b0=0.05`. Increase the value for more conservative estimates on steep angles.
- When sharing results, ensure merge conflicts (e.g., in `pv_index.py`) are resolved so the README matches the code behaviour.
- Visualise the selected decks or the weight distribution in MeshLab or via `visualize_weight.py` to spot modelling artefacts or shadowing issues.
