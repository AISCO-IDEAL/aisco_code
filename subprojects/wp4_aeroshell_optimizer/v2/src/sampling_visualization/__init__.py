"""Sampling visualizer for SDF-based losses and pretraining.

Writes combined PLY files (dark-blue mesh + red sampled points) that can be
inspected in CloudCompare, MeshLab, or Blender.

Typical usage — post-training, static points::

    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization import SamplingVisualizer

    vis = SamplingVisualizer(output_dir=Path("/some/run/sampling_vis"), is_static=True)
    vis.save_snapshot(
        points_mm=pts_tensor,        # (N, 3) float32, in mm
        mesh_verts=verts_np,         # (V, 3) float32, in mm
        mesh_faces=faces_np,         # (F, 3) int64
        name="eikonal_points",
    )

Typical usage — per-epoch (diversity loss, only when epoch % cadence == 0)::

    vis = SamplingVisualizer(output_dir=run_dir / "sampling_vis", is_static=False)
    if epoch % vis_cadence == 0:
        vis.save_snapshot(points_mm=pts, mesh_verts=verts, mesh_faces=faces,
                          name=f"diversity_epoch_{epoch:05d}")
"""

from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.points_SDFmesh_to_ply import SamplingVisualizer

__all__ = ["SamplingVisualizer"]
