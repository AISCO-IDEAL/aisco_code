# Changelog

All notable changes to AISCO_code will be documented in this file.

## [0.4.0] — 2026-04-17

### Changed
- `snappyHexMeshDict`: `nSurfaceLayers` 3 → 8 on `car_surface.*` and `ground` patches; `expansionRatio` 1.3 → 1.15 for smoother BL growth
- `controlDict`: `endTime` 1200 → 3000 (already in 0.3.1, confirmed for this dataset)
- Output directories: new cases in `OPENFOAM/CASES_V2/`, new .pt files in `processed_euler_openfoam_cp_v040/`

### Fixed
- `_recover_surface_mesh()`: removed broken STL fallback; boundary edge ratio > 1% now emits WARNING only and always returns VTK mesh
- `snappyHexMeshDict` template: three malformed negative literals (`- 1e30`, `- 1`, `- 1`) corrected to `-1e30`, `-1`, `-1`; previous template caused `FOAM FATAL IO ERROR` at line 57
- `snappyHexMeshDict` template: fixed indentation of `refinementSurfaces`, `patchInfo`, `levels`, and `locationInMesh` entries that caused OpenFOAM parse errors
- `openfoam_mesh_array.slurm`: added `rm -rf processor*` + `decomposePar -force` to prevent stale decomposition directories from blocking re-runs

### Added
- Extraction quality logging in `process_single_run()`: `n_stl_pts`, `stl_median_edge_mm`, `distinct_normal_dirs`, `p_min`, `p_max`, `p_zero_frac` logged per run
- `_STL_QUALITY_WARN_THRESHOLD = 10_000`: warns when input geometry has fewer than 10k surface points

### Data
- 23 OpenFOAM cases re-meshed (8 BL layers, expansionRatio 1.15) and re-solved to 3000 iterations; all 23 .pt files written to `processed_euler_openfoam_cp_v040/`
- RUN_212 and RUN_214 permanently excluded (box geometries)

## [0.3.1] — 2026-05-07

### Fixed
- Parallel MPI force file bug: `parse_openfoam_forces()` now reads `coefficient_0.dat` / `force_0.dat` (parallel output) in preference to serial variants when both exist, eliminating a silent fallback to stale serial data and correcting Cd extraction for all MPI-parallelised runs

### Added
- Cd stability statistics: `parse_cd_stats()` computes `cd_mean50` and `cd_std50` (mean and sample std over last min(50, N) Cd iterations); both fields stored as `float32[1]` tensors in every `.pt` file
- Cross-dataset validation test suite (`test_cross_dataset_validation.py`): 89 tests verifying schema compatibility, tensor shapes, physical sanity, and Cd sign conventions across OpenFOAM, DrivAer, Ahmed, and BWB datasets
- M3 time-averaging POC infrastructure: `m3_fieldaverage_poc_prepare.sh` and `m3_fieldaverage_poc.slurm` for evaluating fieldAverage on 3 representative cases (RUN_010, RUN_021, RUN_058); POC decision gate: ≥20% Cp MAE improvement for ≥2/3 cases required for adoption

### Changed
- `case_template/system/controlDict`: `endTime` and `writeInterval` updated from 1200 → 3000, ensuring all cases (including hard-to-converge geometries) complete a full 3000-iteration solve; time-averaging window begins at iteration 2700 if fieldAverage is enabled

### Known Issues (documented, not yet fixed)
- Euler Fluent Cd values are all negative (sign convention bug in `process_fluent_pressure.py`); DrivAer/Ahmed/OpenFOAM correctly store positive Cd — tracked in `TestKnownIssues`
- BWB dataset uses minimal schema (missing `face`, `pressure`, `wall_shear`, `cp`) — tracked in `TestKnownIssues`

## [0.3.0] — 2026-04-16

### Fixed
- Air density alignment: `rhoInf` in controlDict updated from 1.2041 to 1.225 (ISA standard), `nu` in transportProperties updated from 1.505e-5 to 1.480e-5 — eliminates ~1.7% systematic Cp/Cd offset
- Ground clearance: exact snap to 0.1 m (was minimum 0.2 m); geometries above target are now lowered

### Added
- Boundary layers: `addLayers true` with `nSurfaceLayers 3` on car_surface and ground patches
- Surface refinement: feature edges level 5→6, surface level (4,5)→(5,6) for better pressure gradient resolution
- Convergence telemetry: `converged` (bool) and `final_residuals` (float32[4]) fields in every `.pt` file, parsed from `log.simpleFoam`

## [0.2.0] — 2026-04-15

### Added
- **OpenFOAM pipeline**: Replaced Fluent with OpenFOAM v2306 for CFD data generation on Euler
- New headless Euler pipeline: PLY → STL → snappyHexMesh → simpleFoam → .pt
- `ply_to_stl.py`: Batch PLY→STL converter with unit detection (mm/m) and mesh repair
- `generate_cases.py`: Parameterized OpenFOAM case instantiation from template
- `process_openfoam_pressure.py`: OpenFOAM result extractor producing identical .pt schema
- OpenFOAM case template: k-ω SST, steady RANS, matching existing Fluent physics
- SLURM array job scripts for parallel batch generation of all 223+ runs
- New dataset option `euler_openfoam` in `preprocess_all_datasets.py`
- Pipeline orchestrator `submit_all.sh` with mesh→solve→extract dependency chain

### Changed
- `preprocess_all_datasets.py`: Added `euler_openfoam` dataset choice and `--euler-openfoam-workers` argument

## [0.1.0] — Initial release

- Initial AISCO monorepo with WP1-WP4 subprojects
