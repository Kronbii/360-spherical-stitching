"""
Tests for the v2 pieces: gyro integration, guided rotation matching, bundle adjustment
and the simulator they are validated with.
"""

import cv2
import numpy as np
import pytest

from src.bundle_adjust import bundle_adjust, reprojection_rms
from src.config import CalibrationData
from src.guided_match import (detect_features, kabsch_rotation, match_pair, pixels_to_bearings,
                              project_bearings, ransac_rotation)
from src.imu import (GyroLog, integrate_gyro, predicted_overlap_pairs, relative_rotation,
                     rotation_error_degrees, so3_exp, so3_log)
from src.pipeline_v2 import V2Options, chain_adjacent, per_frame_errors, run_orientation_v2
from src.simulate import SweepSpec, render_frame, rot_y, simulate_sweep, true_orientation
from src.warp_sphere import warp_image_to_equirectangular, create_equirectangular_grid


def textured_world(width=1024, seed=3):
    """Random-blob equirectangular texture: plenty of ORB corners everywhere."""
    rng = np.random.default_rng(seed)
    h = width // 2
    img = np.full((h, width, 3), 90, np.uint8)
    for _ in range(1500):
        c = (int(rng.integers(0, width)), int(rng.integers(0, h)))
        r = int(rng.integers(3, 14))
        col = tuple(int(x) for x in rng.integers(20, 255, 3))
        if rng.random() < 0.5:
            cv2.circle(img, c, r, col, -1)
        else:
            cv2.rectangle(img, c, (c[0] + r, c[1] + r), col, -1)
    return img


def calib_for(width, height, hfov_deg):
    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    return CalibrationData(fx=f, fy=f, cx=(width - 1) / 2, cy=(height - 1) / 2)


class TestSO3:
    def test_exp_log_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            w = rng.normal(size=3) * rng.uniform(0, 3.0)
            R = so3_exp(w)
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
            assert np.isclose(np.linalg.det(R), 1.0)
            assert np.allclose(so3_exp(so3_log(R)), R, atol=1e-9)

    def test_log_near_pi(self):
        R = so3_exp(np.array([0.0, np.pi - 1e-7, 0.0]))
        assert np.allclose(so3_exp(so3_log(R)), R, atol=1e-6)

    def test_rotation_error(self):
        assert rotation_error_degrees(rot_y(0.1), rot_y(0.1)) == pytest.approx(0.0, abs=1e-9)
        assert rotation_error_degrees(rot_y(np.radians(3)), np.eye(3)) == pytest.approx(3.0)


class TestGyroIntegration:
    def test_noise_free_gyro_reproduces_truth(self):
        spec = SweepSpec(duration_s=6.0, yaw_span_deg=90.0)
        from src.simulate import body_rates
        t = np.arange(0.0, spec.duration_s + 1e-9, 1 / 400)
        gyro = GyroLog(t=t, omega=body_rates(spec, t))
        frame_t = np.linspace(0, spec.duration_s, 13)
        truth = [true_orientation(x, spec) for x in frame_t]
        est = integrate_gyro(gyro, frame_t, truth[0])
        errs = per_frame_errors(est, truth)
        assert errs.max() < 0.05  # zero-order hold at 400 Hz

    def test_constant_bias_gives_linear_drift(self):
        t = np.arange(0.0, 10.0 + 1e-9, 0.01)
        bias = np.radians(0.2)
        gyro = GyroLog(t=t, omega=np.tile([0.0, bias, 0.0], (len(t), 1)))
        est = integrate_gyro(gyro, np.array([0.0, 5.0, 10.0]), np.eye(3))
        assert rotation_error_degrees(est[1], np.eye(3)) == pytest.approx(1.0, rel=1e-3)
        assert rotation_error_degrees(est[2], np.eye(3)) == pytest.approx(2.0, rel=1e-3)
        # And subtracting a known bias removes it
        est_fixed = integrate_gyro(gyro, np.array([0.0, 10.0]), np.eye(3), bias=[0, bias, 0])
        assert rotation_error_degrees(est_fixed[1], np.eye(3)) < 1e-9

    def test_frames_between_samples(self):
        t = np.arange(0.0, 1.0 + 1e-9, 0.1)
        gyro = GyroLog(t=t, omega=np.tile([0.0, 1.0, 0.0], (len(t), 1)))
        est = integrate_gyro(gyro, np.array([0.03, 0.57]), np.eye(3))
        assert rotation_error_degrees(est[1], np.eye(3)) == pytest.approx(np.degrees(0.54), rel=1e-6)

    def test_rejects_short_log(self):
        gyro = GyroLog(t=np.array([0.0, 1.0]), omega=np.zeros((2, 3)))
        with pytest.raises(ValueError):
            integrate_gyro(gyro, np.array([0.0, 2.0]), np.eye(3))

    def test_overlap_pairs_find_loop_closure(self):
        yaws = np.radians(np.arange(0, 400, 10))
        rots = [rot_y(y).T for y in yaws]
        pairs = predicted_overlap_pairs(rots, max_angle_deg=25, min_gap=5)
        assert (0, 36) in pairs and (1, 37) in pairs
        assert all(j - i >= 5 for i, j in pairs)


class TestRotationModel:
    def test_kabsch_recovers_rotation(self):
        rng = np.random.default_rng(1)
        R = so3_exp(np.array([0.1, -0.3, 0.05]))
        a = rng.normal(size=(20, 3))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        assert np.allclose(kabsch_rotation(a, a @ R.T), R, atol=1e-10)

    def test_ransac_rejects_outliers(self):
        rng = np.random.default_rng(2)
        R = so3_exp(np.array([0.02, 0.12, -0.01]))
        K = calib_for(640, 480, 60).K
        pts = rng.uniform([0, 0], [640, 480], size=(200, 2))
        b_i = pixels_to_bearings(pts, K)
        b_j = b_i @ R.T
        b_j[:60] = pixels_to_bearings(rng.uniform([0, 0], [640, 480], size=(60, 2)), K)
        R_hat, mask = ransac_rotation(b_i, b_j, threshold_rad=1.5 / K[0, 0], rng=rng)
        assert rotation_error_degrees(R_hat, R) < 1e-3
        assert mask[60:].all() and mask[:60].sum() < 5

    def test_projection_roundtrip(self):
        K = calib_for(640, 480, 60).K
        pts = np.array([[0.0, 0.0], [320.0, 240.0], [639.0, 479.0]])
        uv, front = project_bearings(pixels_to_bearings(pts, K), K)
        assert front.all() and np.allclose(uv, pts)

    def test_guided_match_on_rendered_pair(self):
        world = textured_world()
        W, H = 480, 360
        calib = calib_for(W, H, 60)
        R_a = rot_y(0.0).T
        R_b = (rot_y(np.radians(12)) @ so3_exp(np.array([0.02, 0.0, 0.01]))).T
        img_a = render_frame(world, R_a, calib, W, H)
        img_b = render_frame(world, R_b, calib, W, H)
        fa, fb = detect_features(img_a, 2000), detect_features(img_b, 2000)
        R_true = relative_rotation(R_a, R_b)
        # A prior that is 1.5° off, as a drifting gyro would be
        R_prior = so3_exp(np.radians([0.0, 1.5, 0.0])) @ R_true
        pm = match_pair(0, 1, fa, fb, calib.K, R_prior, search_radius_px=40.0)
        assert pm.success and pm.inliers > 100
        assert rotation_error_degrees(pm.R_ij, R_true) < 0.1

    def test_prior_disagreement_rejects(self):
        world = textured_world()
        W, H = 480, 360
        calib = calib_for(W, H, 60)
        img_a = render_frame(world, np.eye(3), calib, W, H)
        img_b = render_frame(world, rot_y(np.radians(10)).T, calib, W, H)
        fa, fb = detect_features(img_a, 2000), detect_features(img_b, 2000)
        # Without a prior the pair fits; with a badly wrong prior and a window wide enough
        # to still find the matches, the fit must be refused
        pm = match_pair(0, 1, fa, fb, calib.K, None)
        assert pm.success
        wrong = rot_y(np.radians(-10)).T
        pm_bad = match_pair(0, 1, fa, fb, calib.K, wrong, search_radius_px=2000,
                            max_prior_disagreement_deg=2.0)
        assert not pm_bad.success


def synthetic_pairs(truth, K, n_pts=80, noise_px=0.3, seed=0, gaps=(1, 2), extra=()):
    """Exact correspondences between frames for known rotations, plus pixel noise."""
    from src.guided_match import PairMatch
    rng = np.random.default_rng(seed)
    W, H = 2 * K[0, 2] + 1, 2 * K[1, 2] + 1
    pairs = []
    n = len(truth)
    idx = [(i, i + g) for g in gaps for i in range(n - g)] + list(extra)
    for i, j in idx:
        R_ij = relative_rotation(truth[i], truth[j])
        pts_i = rng.uniform([0, 0], [W, H], size=(n_pts * 3, 2))
        uv, front = project_bearings(pixels_to_bearings(pts_i, K) @ R_ij.T, K)
        keep = front & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        pts_i, uv = pts_i[keep][:n_pts], uv[keep][:n_pts]
        uv = uv + rng.normal(0, noise_px, uv.shape)
        pairs.append(PairMatch(i, j, pts_i.astype(np.float32), uv.astype(np.float32),
                               R_ij, len(uv), len(uv)))
    return pairs


class TestBundleAdjustment:
    def make_ring(self, n=36, step_deg=11.0):
        return [(rot_y(np.radians(step_deg * k)) @ so3_exp(np.radians([2 * np.sin(k / 3), 0, np.cos(k / 4)]))).T
                for k in range(n)]

    def test_recovers_rotations_and_focal_with_loop_closure(self):
        truth = self.make_ring()
        calib = calib_for(640, 480, 60)
        pairs = synthetic_pairs(truth, calib.K, extra=[(0, 33), (0, 34), (1, 34), (1, 35), (2, 35)])
        rng = np.random.default_rng(5)
        # Start from a drifting prior: error grows along the sweep
        init = [so3_exp(np.radians(rng.normal(0, 0.3, 3) + [0, 0.1 * k, 0])) @ R
                for k, R in enumerate(truth)]
        init[0] = truth[0]
        res = bundle_adjust(init, pairs, focal=calib.fx * 0.96, cx=calib.cx, cy=calib.cy)
        errs = per_frame_errors(res.rotations, truth)
        assert errs.max() < 0.05
        assert res.focal == pytest.approx(calib.fx, rel=2e-3)
        assert res.rms_after_px < 0.6 < res.rms_before_px
        assert np.allclose(res.rotations[0], truth[0])

    def test_robust_to_bad_correspondences(self):
        truth = self.make_ring(n=12)
        calib = calib_for(640, 480, 60)
        pairs = synthetic_pairs(truth, calib.K, seed=1)
        rng = np.random.default_rng(9)
        for p in pairs[::3]:
            p.pts_j[:8] = rng.uniform([0, 0], [640, 480], size=(8, 2)).astype(np.float32)
        init = [so3_exp(np.radians(rng.normal(0, 0.5, 3))) @ R for R in truth]
        init[0] = truth[0]
        res = bundle_adjust(init, pairs, focal=calib.fx, cx=calib.cx, cy=calib.cy,
                            optimize_focal=False)
        assert per_frame_errors(res.rotations, truth).max() < 0.05

    def test_reprojection_rms_is_zero_for_truth(self):
        truth = self.make_ring(n=8)
        calib = calib_for(640, 480, 60)
        pairs = synthetic_pairs(truth, calib.K, noise_px=0.0)
        assert reprojection_rms(truth, calib.fx, calib.cx, calib.cy, pairs) < 1e-3

    def test_chain_falls_back_to_prior(self):
        truth = self.make_ring(n=5)
        calib = calib_for(640, 480, 60)
        pairs = synthetic_pairs(truth, calib.K, gaps=(1,))
        pairs[2].R_ij = None  # pair (2, 3) failed
        chained = chain_adjacent(truth[0], pairs, truth, len(truth))
        assert per_frame_errors(chained, truth).max() < 1e-6


class TestSimulator:
    def test_render_then_warp_lands_in_place(self):
        """A rendered frame warped back with the true rotation reproduces the world."""
        world = textured_world(width=1024)
        W, H = 400, 300
        calib = calib_for(W, H, 70)
        R = (rot_y(np.radians(40)) @ so3_exp(np.radians([5, 0, 3]))).T
        frame = render_frame(world, R, calib, W, H)
        theta, phi = create_equirectangular_grid(1024, 512)
        warped, mask = warp_image_to_equirectangular(frame, R, calib, theta, phi, (512, 1024))
        inner = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
        diff = np.abs(warped.astype(float) - world.astype(float))[inner]
        assert inner.sum() > 10_000
        assert np.median(diff) < 12

    def test_sweep_ground_truth_and_gyro_drift(self):
        world = textured_world(width=1024)
        spec = SweepSpec(duration_s=4.0, frame_rate_hz=2.0, yaw_span_deg=60.0,
                         image_width=160, image_height=120)
        sim = simulate_sweep(world, spec)
        assert len(sim.frames) == 9 and sim.frames[0].shape == (120, 160, 3)
        assert np.allclose(sim.gt_rotations[0], true_orientation(0.0, spec))
        assert len(sim.gyro.t) == int(spec.duration_s * spec.gyro_rate_hz) + 1
        prior = integrate_gyro(sim.gyro, sim.frame_times, sim.gt_rotations[0])
        drift = per_frame_errors(prior, sim.gt_rotations)
        # Bias is 0.25°/s: after 4 s the prior is off by roughly a degree, not zero, not ten
        assert 0.3 < drift[-1] < 3.0

    @pytest.mark.slow
    def test_end_to_end_ba_beats_imu_and_chain(self):
        world = textured_world(width=2048)
        spec = SweepSpec(duration_s=12.0, frame_rate_hz=3.0, yaw_span_deg=380.0,
                         image_width=400, image_height=300, hfov_deg=70.0)
        sim = simulate_sweep(world, spec)
        f0 = sim.calib.fx * 0.97
        init = CalibrationData(fx=f0, fy=f0, cx=sim.calib.cx, cy=sim.calib.cy)
        res = run_orientation_v2(sim.frames, sim.frame_times, sim.gyro, init,
                                 sim.gt_rotations[0], V2Options(orb_nfeatures=1500))
        e_imu = per_frame_errors(res.imu_rotations, sim.gt_rotations)
        e_chain = per_frame_errors(res.chained_rotations, sim.gt_rotations)
        e_ba = per_frame_errors(res.ba_rotations, sim.gt_rotations)
        assert e_ba.mean() < 0.2
        assert e_ba.mean() < e_imu.mean() and e_ba.mean() < e_chain.mean()
        assert res.focal_ba == pytest.approx(sim.calib.fx, rel=0.01)
