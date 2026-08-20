"""Tests for the precomputed direction grid and threaded warping."""

import numpy as np
import pytest

from src.config import CalibrationData, OutputConfig
from src.warp_sphere import (
    compute_warp_maps,
    create_equirectangular_grid,
    spherical_to_world_directions,
    warp_and_blend_sequential,
    warp_image_to_equirectangular,
    world_direction_grid,
)


@pytest.fixture
def grid():
    return create_equirectangular_grid(128, 64)


@pytest.fixture
def calib():
    return CalibrationData(fx=200.0, fy=200.0, cx=64.0, cy=48.0)


class TestWorldDirectionGrid:
    def test_matches_the_original_formulation(self, grid):
        theta, phi = grid
        new = world_direction_grid(theta, phi)
        old = spherical_to_world_directions(theta, phi)
        for a, b in zip(new, old):
            assert np.allclose(a, b, atol=1e-6)

    def test_is_float32(self, grid):
        for comp in world_direction_grid(*grid):
            assert comp.dtype == np.float32

    def test_directions_are_unit_length(self, grid):
        x, y, z = world_direction_grid(*grid)
        assert np.allclose(np.sqrt(x**2 + y**2 + z**2), 1.0, atol=1e-5)


class TestWarpMapsWithPrecomputedGrid:
    def test_passing_the_grid_changes_nothing(self, grid, calib):
        theta, phi = grid
        R = np.array([[0.9848, 0.0, 0.1736], [0.0, 1.0, 0.0], [-0.1736, 0.0, 0.9848]])
        a = compute_warp_maps(theta, phi, R, calib, 128, 96, None)
        b = compute_warp_maps(theta, phi, R, calib, 128, 96, world_direction_grid(theta, phi))
        assert np.allclose(a[0], b[0], atol=1e-4)
        assert np.allclose(a[1], b[1], atol=1e-4)
        assert np.array_equal(a[2], b[2])

    def test_mask_marks_only_pixels_inside_the_frame(self, grid, calib):
        theta, phi = grid
        map_x, map_y, valid = compute_warp_maps(theta, phi, np.eye(3), calib, 128, 96)
        assert valid.any() and not valid.all()
        assert (map_x[valid] >= 1.0).all() and (map_x[valid] < 127.0).all()
        assert (map_y[valid] >= 1.0).all() and (map_y[valid] < 95.0).all()

    def test_mask_dtype_is_boolean(self, grid, calib):
        _, _, valid = compute_warp_maps(theta_phi := grid[0], grid[1], np.eye(3), calib, 128, 96)
        assert valid.dtype == np.bool_

    def test_projection_matches_the_pinhole_formula(self, calib):
        """Near the optical axis the map must be exactly fx*tan(theta) + cx."""
        theta, phi = create_equirectangular_grid(1024, 512)
        map_x, map_y, valid = compute_warp_maps(theta, phi, np.eye(3), calib, 128, 96)
        r = int(np.argmin(np.abs(phi[:, 0])))          # closest row to the horizon
        c = int(np.argmin(np.abs(theta[0, :])))        # closest column to theta = 0
        assert valid[r, c]
        # the nearest grid column is a fraction of a degree off centre, so compare
        # against the angle that column actually represents
        assert map_x[r, c] == pytest.approx(calib.fx * np.tan(theta[r, c]) + calib.cx, abs=1e-3)
        assert map_y[r, c] == pytest.approx(calib.fy * np.tan(phi[r, c]) + calib.cy, abs=1e-3)


class TestThreadedWarping:
    @staticmethod
    def _infos(tmp_path, n):
        import cv2

        from src.io_utils import ImageInfo
        infos = []
        rng = np.random.default_rng(1)
        for i in range(n):
            p = tmp_path / f"f{i:02d}.jpg"
            cv2.imwrite(str(p), rng.integers(0, 255, (96, 128, 3), dtype=np.uint8))
            infos.append(ImageInfo(path=p, mtime=float(i)))
        return infos

    @staticmethod
    def _rots(n):
        out = []
        for i in range(n):
            a = np.radians(-3.0 * i)
            out.append(np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]))
        return out

    def _collect(self, tmp_path, workers, n=6):
        infos, rots = self._infos(tmp_path, n), self._rots(n)
        cfg = OutputConfig(pano_width=128, warp_workers=workers)
        seen, frames = [], []

        def cb(warped, mask, i):
            seen.append(i)
            frames.append((warped.copy(), mask.copy()))

        masks = warp_and_blend_sequential(infos, rots, CalibrationData(fx=120.0, fy=120.0, cx=64.0, cy=48.0),
                                         cfg, cb, 128, 96)
        return seen, frames, masks

    def test_callbacks_arrive_in_index_order(self, tmp_path):
        seen, _, _ = self._collect(tmp_path, workers=4)
        assert seen == sorted(seen) == list(range(len(seen)))

    def test_threaded_output_matches_serial(self, tmp_path):
        _, serial, m1 = self._collect(tmp_path, workers=1)
        _, threaded, m2 = self._collect(tmp_path, workers=4)
        assert len(serial) == len(threaded)
        for (wa, ma), (wb, mb) in zip(serial, threaded):
            assert np.array_equal(wa, wb)
            assert np.array_equal(ma, mb)
        assert all(np.array_equal(a, b) for a, b in zip(m1, m2))

    def test_returns_one_mask_per_frame(self, tmp_path):
        _, _, masks = self._collect(tmp_path, workers=3, n=5)
        assert len(masks) == 5
        assert all(m.shape == (64, 128) for m in masks)

    def test_auto_worker_count_runs(self, tmp_path):
        seen, _, masks = self._collect(tmp_path, workers=0)
        assert seen == list(range(len(masks)))

    def test_more_workers_than_frames_is_safe(self, tmp_path):
        seen, _, _ = self._collect(tmp_path, workers=32, n=3)
        assert seen == [0, 1, 2]
