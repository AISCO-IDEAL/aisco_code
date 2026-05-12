# Changelog

All notable changes to AISCO are documented here.

## [0.7.1] — 2025-07-10

### WP7 GUI v3 cleanup (plan 020)

- **Deleted dead files**: `launcher/solar.html`, `launcher/testing.html`,
  `assembly/live/assembly.step`, `assembly/completed/assembly.step`,
  `core/stdout_stream.py` (renamed to `log_parser.py`).
- **Backend cleanup** (all routes, shared modules, settings):
  - Removed redundant `_storage_root()` helpers across 3 files; unified via
    `shared/run_paths.default_storage_root()`.
  - Removed `gui_override_prefix`, `copy_exclude_subfolders`,
    `mesh_assembly_dir_candidates` from `Settings` dataclass.
  - Renamed `_MESH_ASSEMBLY_CANDIDATES` → public `MESH_ASSEMBLY_CANDIDATES`.
  - Hardened `_assembly()` routing with explicit `raise ValueError` on unknown
    pipeline values.
  - Improved error handling in `gif_serving.py`, `config_overrides.py`,
    `process_manager.py`, `fs_poller.py`.
- **Frontend cleanup**:
  - Removed Alpine.js and htmx from `base.html`.
  - Extracted shared `bust(url)` and `makeConsoleLogger(el)` into
    `static/js/utils.js` (loaded globally); removed inline duplicates from 4
    templates.
  - Extracted Three.js viewer into `static/js/mesh_viewer.js` as an ES module;
    replaced ~35-line inline Three.js blocks in `output.html`,
    `assembly_selection.html`, and `viewer/index.html`.
  - Removed dead CSS classes `.run-name`, `.run-meta`, `.run-badges`; fixed
    `.run-item` layout from `display:grid` to `display:flex; flex-direction:
    column`.
  - Removed all `"ui": "number"` fields from `schema/gui_schema.json`.
  - Fixed output.html startup race: poll only starts when run is active.
  - Added session-absent notice to `optimization.html` toolbar.
