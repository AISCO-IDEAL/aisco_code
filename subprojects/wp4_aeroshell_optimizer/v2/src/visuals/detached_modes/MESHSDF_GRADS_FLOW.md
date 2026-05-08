# MeshSDFGrads — per-loss vertex-gradient visualisation
## What it shows
Each frame colours the surface mesh by `dL/dv · n` per vertex:
- **Green** = the loss wants to push the surface outward at this vertex.
- **Red** = the loss wants to push the surface inward at this vertex.
- **Grey** = the loss is neutral (orthogonal grad, or zero).

One `MeshSDFGrads` instance is created per `loss_key` in
`configs/neural_renderer_v0.yaml > detached_modes`. Use `loss_key: "__total__"`
for the weighted sum (what the optimiser actually sees) or a specific loss name
(e.g. `surface_loss_v0`) for a single component. Each block has its own
`enabled` flag so any subset of losses can be visualised independently.
## Ownership
Visualisation state lives entirely on the active `TotalLoss*` instance —
`TotalLossV2` for the diversity / concat path, and `TotalLoss` (V1) for the
legacy single-optimisation path used by non-concat models (which run through
`model_baseclass.optimise`). Both expose the same
`collect_vertex_grads_for_visualisation(weights_by_name)` API. **No loss has
any visualisation-specific code.** Any active loss can be turned off at any
time without breaking the visualiser; if no mesh-using loss is active at all,
the visualiser silently does nothing.
## Data flow
```
              ┌───────────────────────────────────────────────────┐
              │ TotalLoss{,V2}.{compute,_compute_raw_losses}      │
              │   1. _inject_shared_meshes(model)                 │
              │   2. _extract_primary_visu_mesh(model)            │
              │        - finds any active MeshSDFExtractor        │
              │        - reuses existing shared mesh, or extracts │
              │          one and pins it via set_shared_mesh()    │
              │        - stores tensor on self._visu_primary_v/_f │
              │   3. evaluates every active loss → raw tensors    │
              │   4. _capture_vertex_grads_via_autograd(...)      │
              │        for each active loss L_k:                  │
              │          torch.autograd.grad(L_k, primary_v,      │
              │                              allow_unused=True)   │
              │        - stores result on self._visu_per_loss_grads│
              │        - losses with no autograd path → absent    │
              └────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
              ┌───────────────────────────────────────────────────┐
              │ TotalLoss*.collect_vertex_grads_for_visualisation │
              │   reads self._visu_* and returns                  │
              │   {vertices, faces, per_loss, total}              │
              │   total = Σ_k w_k · dL/dv_k                       │
              └────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
              ┌───────────────────────────────────────────────────┐
              │ Renderer-update site (one of two):                │
              │  • concat / diversity:                            │
              │        GINN/diversity_manager.py :: _render        │
              │  • non-concat / single optimisation:               │
              │        losses/total_loss.py :: get_total_loss      │
              │   both inject the same payload into                │
              │   renderer_manager.update(...):                   │
              │     "vertices", "faces"  (= primary mesh)         │
              │     "vertice_grads_per_loss"  : {name: (V,3)}     │
              │     "vertice_grads_total"     :       (V,3)       │
              └────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
              ┌───────────────────────────────────────────────────┐
              │ visuals/visual_wrapper.py :: VisualisationManager │
              │   loads rendering_modes AND detached_modes from   │
              │   YAML (one MeshSDFGrads instance per block)      │
              │   dispatches optimization_state to every mode     │
              │   streams every appended frame to wandb           │
              └────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
              ┌───────────────────────────────────────────────────┐
              │ visuals/detached_modes/mesh_sdf_grads.py          │
              │   MeshSDFGrads(loss_key, intensity)               │
              │     update(): pick grad by loss_key,              │
              │               colour mesh, render frame           │
              │     finalize(): GIF + first/last PNGs to wandb    │
              │     log_single_image_to_wandb(): per-step image   │
              └───────────────────────────────────────────────────┘
```
## Files involved
| File | Role |
|---|---|
| [total_loss_v2.py](../../../src/losses/total_loss_v2.py) | Owns visu state for the **diversity / concat** path. Extracts primary mesh, captures per-loss `dL/dv` after every `compute()`, exposes `collect_vertex_grads_for_visualisation()`. |
| [total_loss.py](../../../src/losses/total_loss.py) | Same machinery for the **legacy single-optimisation** path (V1 `TotalLoss`, used by `has_optimise=True` non-concat models via `model_baseclass.optimise`). Capture happens inside `_compute_raw_losses`; injection into the renderer payload happens directly in `get_total_loss` since this path has no diversity manager. |
| [diversity_manager.py](../../../src/GINN/diversity_manager.py) | `_render` injects `vertice_grads_per_loss` and `vertice_grads_total` into `optimization_state` (diversity path only). |
| [visual_wrapper.py](../visual_wrapper.py) | Loads `detached_modes` block in addition to `rendering_modes`. |
| [mesh_sdf_grads.py](mesh_sdf_grads.py) | Config-driven by `loss_key` (or `"__total__"`); colours mesh by `dL/dv·n`, writes GIF + per-step image to wandb. |
| [neural_renderer_v0.yaml](../../../configs/neural_renderer_v0.yaml) | One `MeshSDF_grads_<loss>` block per loss + a `_total` block. Each `enabled` flag is independent. |
## Adding a new loss to the visualisation
Add a `MeshSDF_grads_<name>` block under `detached_modes` in the YAML:
```yaml
MeshSDF_grads_my_loss:
  enabled: True
  class_path: "detached_modes.mesh_sdf_grads:MeshSDFGrads"
  config:
    camera_position: [-2000, 0, 7000]
    look_at: [-1500, 0, 0]
    render_mode: "rgb"
    fps: 10
    loss_key: "my_loss_name"   # the loss_name from losses_v0_config.yaml
    intensity: 1.0
```
**Caveat — when does the panel show data?** The capture relies on `autograd.grad(L_k, primary_verts)`. A loss only contributes if it has an autograd path to the SDF mesh, which currently means it computes its loss as a function of the marching-cubes vertices (the surface_loss / developability pattern). Losses that sample independent points in space (eikonal, smoothness, …) have no path and their panel will be empty. This is a property of the loss, not a bug in the renderer.
## Cost
~1 ms × K losses per training step on a 2080 Ti. The expensive piece — `extract_mesh` — already runs as part of every active mesh-using loss. The visu pass adds one extra `set_shared_mesh` call when only one extractor exists, plus one `autograd.grad` per active loss against the same mesh tensor.
