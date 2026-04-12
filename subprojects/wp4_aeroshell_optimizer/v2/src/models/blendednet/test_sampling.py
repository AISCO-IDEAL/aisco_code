"""Tests for new curvature/tip/extreme-weight sampling functions."""
import numpy as np
import pytest
import torch

# ── precompute_sdf helpers ──────────────────────────────────────────────

def _make_box_mesh():
    """Create a simple pyvista box mesh for testing."""
    import pyvista as pv
    return pv.Box(bounds=(-1, 1, -2, 2, -0.5, 0.5)).triangulate()


class TestComputeTriangleCurvature:
    def test_returns_per_triangle_array(self):
        from precompute_sdf import compute_triangle_curvature
        surf = _make_box_mesh()
        curv = compute_triangle_curvature(surf)
        n_tri = surf.n_faces
        assert curv.shape == (n_tri,)
        assert (curv >= 0).all()

    def test_nonzero_weights(self):
        from precompute_sdf import compute_triangle_curvature
        surf = _make_box_mesh()
        curv = compute_triangle_curvature(surf)
        # floor should ensure all > 0
        assert (curv > 0).all()


class TestSampleCurvatureWeighted:
    def test_output_shape(self):
        from precompute_sdf import sample_curvature_weighted
        surf = _make_box_mesh()
        pts, normals = sample_curvature_weighted(surf, 500, sigma=0.005)
        assert pts.shape == (500, 3)
        assert normals.shape == (500, 3)

    def test_points_near_surface(self):
        from precompute_sdf import sample_curvature_weighted
        surf = _make_box_mesh()
        pts, _ = sample_curvature_weighted(surf, 1000, sigma=0.005)
        # points should be close to the box (within ~3*sigma of surface)
        assert pts[:, 0].min() > -1.1
        assert pts[:, 0].max() < 1.1


class TestSampleWingtipPoints:
    def test_output_shape(self):
        from precompute_sdf import sample_wingtip_points
        surf = _make_box_mesh()
        pts, normals = sample_wingtip_points(surf, 500, sigma=0.003)
        assert pts.shape == (500, 3)
        assert normals.shape == (500, 3)

    def test_concentrated_at_high_y(self):
        from precompute_sdf import sample_wingtip_points
        surf = _make_box_mesh()  # y range [-2, 2]
        pts, _ = sample_wingtip_points(surf, 2000, sigma=0.001)
        # most points should have |y| > 80th percentile of range = 1.6
        high_y_frac = (np.abs(pts[:, 1]) > 1.2).mean()
        assert high_y_frac > 0.5, f"only {high_y_frac:.2f} of tip pts at high |y|"


class TestSampleEdgePoints:
    def test_output_shape(self):
        from precompute_sdf import sample_edge_points
        surf = _make_box_mesh()
        pts, normals = sample_edge_points(surf, 500, sigma=0.001)
        assert pts.shape == (500, 3)
        assert normals.shape == (500, 3)

    def test_concentrated_at_x_extremes(self):
        from precompute_sdf import sample_edge_points
        surf = _make_box_mesh()  # x range [-1, 1]
        pts, _ = sample_edge_points(surf, 2000, sigma=0.001)
        # edge sampler should favor x near -1 and +1
        abs_x = np.abs(pts[:, 0])
        # most points should have |x| > 0.5 (near edges, not center)
        edge_frac = (abs_x > 0.5).mean()
        assert edge_frac > 0.4, f"only {edge_frac:.2f} of edge pts at |x|>0.5"


# ── train_sdf helpers ──────────────────────────────────────────────────

class TestComputeExtremeWeights:
    def test_output_shape_and_sums_to_one(self):
        from train_sdf import compute_extreme_weights
        params = torch.rand(50, 9)
        pmin = params.min(dim=0).values
        pmax = params.max(dim=0).values
        w = compute_extreme_weights(params, pmin, pmax, max_weight=3.0)
        assert w.shape == (50,)
        assert abs(w.sum().item() - 1.0) < 1e-5

    def test_extreme_geom_gets_higher_weight(self):
        from train_sdf import compute_extreme_weights
        # 10 center geoms + 1 extreme
        params = torch.full((11, 9), 0.5)
        params[10] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pmin = torch.zeros(9)
        pmax = torch.ones(9)
        w = compute_extreme_weights(params, pmin, pmax, max_weight=3.0)
        # extreme geom should have highest weight
        assert w[10] > w[0]

    def test_max_weight_one_gives_uniform(self):
        from train_sdf import compute_extreme_weights
        params = torch.rand(20, 9)
        pmin = params.min(dim=0).values
        pmax = params.max(dim=0).values
        w = compute_extreme_weights(params, pmin, pmax, max_weight=1.0)
        # all weights should be equal
        assert torch.allclose(w, w[0].expand_as(w), atol=1e-5)


class TestSampleBatchWeighted:
    def test_weighted_sampling_runs(self):
        from train_sdf import sample_batch
        N, P = 20, 100
        points = torch.randn(N, P, 3)
        sdf = torch.randn(N, P)
        params = torch.randn(N, 9)
        weights = torch.ones(N) / N
        xyz, s, p = sample_batch(points, sdf, params, 8, 32, weights=weights)
        assert xyz.shape == (8, 32, 3)
        assert s.shape == (8, 32)
        assert p.shape == (8, 9)

    def test_none_weights_still_works(self):
        from train_sdf import sample_batch
        N, P = 20, 100
        points = torch.randn(N, P, 3)
        sdf = torch.randn(N, P)
        params = torch.randn(N, 9)
        xyz, s, p = sample_batch(points, sdf, params, 8, 32, weights=None)
        assert xyz.shape == (8, 32, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
